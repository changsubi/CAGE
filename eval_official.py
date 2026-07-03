"""
공식 REC-8K 프로토콜과 정렬된 평가 (train_test.py의 calc_loc_metric/distance_threshold_func 포팅)
================================================================================================
배경:
  공식 GroundingREC 평가(train_test.py:247-289)는 localization F1을 다음으로 계산한다.
    - 예측점 vs GT점 유클리드 cost → Hungarian 1:1 매칭 → cost < thr 이면 TP
    - thr = sqrt(w^2 + h^2) / 2  (median-area '예측 박스'의 w,h), 하한 없음
  그런데 REGraph는 박스를 예측하지 않아(점만 출력) 공식 thr을 직접 계산할 수 없다.
  → 본 스크립트는 공식 매칭/지표 로직을 '그대로' 쓰되, REGraph용 임계값을 명시적으로
    선택(--thr_mode)하게 하여 '어떤 임계값에서 우리 F1이 얼마인지'를 정직하게 측정한다.

임계값 정책(--thr_mode):
  fixed   : 모든 샘플에 고정 거리 임계값(--thr_value, 정규화 좌표). baseline 재평가와 동일값을
            쓰면 공정 비교가 된다. (권장: 먼저 공식 코드로 baseline의 평균 thr을 구해 맞춤)
  gtspace : 이미지별 GT 최근접거리 중앙값/2 를 객체 scale proxy로 사용(박스 없는 모델용 대안,
            '객체 크기 적응적'이라는 공식 thr의 정신을 점-only로 근사). 모든 모델에 동일 적용 시 공정.
  grid    : 기존 우리 방식(cell_diag 기반, 0.05 하한) — 과거 수치 재현·디버깅용.

주의: 출판용 비교표는 반드시 (a) REGraph에 size head를 달아 공식 thr을 쓰거나,
      (b) baseline 체크포인트를 이 스크립트의 동일 --thr_mode/--thr_value로 재평가해야 한다.
      서로 다른 thr 정책의 숫자를 같은 표에 넣으면 무효다.

사용법:
  python eval_official.py --checkpoint ./reg_f1_ds0p55_ow1p0_s42/best.pth --split test --thr_mode fixed --thr_value 0.05
  python eval_official.py --checkpoint <ckpt> --split val --thr_mode gtspace
"""

import os, sys, argparse
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import load_model

from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import (
    REGraphDecoder, build_grid_graph, count_sum_guided, count_graph_guided,
)
from train_regraph import run_feature_enhancer, setup_freeze

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def official_calc_loc_metric(pred_points, gt_points, dist_threshold):
    """train_test.py:261-289 의 calc_loc_metric 과 동일한 매칭/집계 로직.
    단, dist_threshold를 인자로 받는다(공식은 박스에서 계산, 여기선 정책별로 주입)."""
    n_pred = len(pred_points)
    n_gt = len(gt_points)
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    pred_np = np.asarray(pred_points, dtype=np.float64)
    gt_np = np.asarray(gt_points, dtype=np.float64)
    cost = cdist(pred_np, gt_np, metric='euclidean')
    pred_idx, gt_idx = linear_sum_assignment(cost)

    TP = int(sum(1 for pi, gi in zip(pred_idx, gt_idx) if cost[pi, gi] < dist_threshold))
    FP = n_pred - TP
    FN = n_gt - TP
    return TP, FP, FN


def gtspace_threshold(gt_points_norm):
    """GT 최근접거리 중앙값의 절반을 객체 scale proxy로 (박스 없는 모델용 적응 임계값)."""
    k = len(gt_points_norm)
    if k <= 1:
        return 0.05
    d = cdist(gt_points_norm, gt_points_norm)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.median(nn) / 2.0)


def evaluate_official(model, step_a, decoder, loader, annotations,
                      thr_mode='fixed', thr_value=0.05, nms_dist=None,
                      loc_mode='sum', edge_tau=0.5):
    model.eval(); step_a.eval(); decoder.eval()
    mae_sum = rmse_sum = 0.0
    tot_tp = tot_fp = tot_fn = 0
    counter = 0
    graph_cache = {}
    # over-count 진단: K=round(sum)이 gt보다 크면 잉여점은 dedup으로 못 없애는 강제 FP
    overcount_imgs = 0          # round(sum) > gt 인 이미지 수
    forced_fp = 0               # sum(max(0, round(sum)-gt)) — dedup 불가 FP 총량
    tot_gt = 0

    with torch.no_grad():
        for images, caps_list, shapes, img_caps_list in loader:
            anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
            shapes_exp = [shapes[i] for i, cl in enumerate(caps_list) for _ in cl]
            images_exp = torch.stack(
                [images[i] for i, cl in enumerate(caps_list) for _ in cl]).to(device)
            captions = [c for cl in caps_list for c in cl]

            feat = run_feature_enhancer(model, images_exp, captions, training=False)
            nodes, full_emb, class_emb, attr_emb, H2, W2 = step_a(
                feat['img_memory'], feat['txt_memory'],
                feat['spatial_shapes'], feat['level_start_index'],
                feat['text_token_mask'], feat['text_class_mask'], feat['text_attr_mask'])

            key = (H2, W2)
            if key not in graph_cache:
                ei, pos = build_grid_graph(H2, W2)
                graph_cache[key] = (ei.to(device), pos.to(device))
            edge_index, positions = graph_cache[key]

            for b in range(len(captions)):
                gt_pts = anno_b[b]['points']
                gt_count = len(gt_pts)
                out = decoder(nodes[b], class_emb[b], attr_emb[b], edge_index, positions)

                # counting (sum) — MAE/RMSE
                ps = out['combined_scores'].sum().item()
                mae_sum += abs(ps - gt_count)
                rmse_sum += (ps - gt_count) ** 2

                # over-count 진단 (dedup 천장)
                K = max(0, int(round(ps)))
                if K > gt_count:
                    overcount_imgs += 1
                    forced_fp += (K - gt_count)
                tot_gt += gt_count

                # localization 점 예측 (공식과 동일하게 '점' 사용)
                if loc_mode == 'graph':
                    # 학습된 inter_edges로 같은-객체 중복 제거 (N1 활용)
                    gd = nms_dist if nms_dist is not None else 1.0
                    _, centers, _ = count_graph_guided(
                        out['combined_scores'].cpu(), positions.cpu(),
                        edge_index.cpu(), out['inter_edges'].cpu(),
                        H=H2, W=W2, offsets=out['offsets'].cpu(),
                        edge_tau=edge_tau, min_dist_cells=gd)
                else:
                    _, centers, _ = count_sum_guided(
                        out['combined_scores'].cpu(), positions.cpu(),
                        H=H2, W=W2, offsets=out['offsets'].cpu(), min_dist_cells=nms_dist)

                # GT 정규화
                if gt_count > 0:
                    gt_norm = np.array(gt_pts, dtype=np.float64)
                    gt_norm[:, 0] /= shapes_exp[b][1]
                    gt_norm[:, 1] /= shapes_exp[b][0]
                else:
                    gt_norm = np.zeros((0, 2))

                # 임계값 정책
                if thr_mode == 'fixed':
                    thr = thr_value
                elif thr_mode == 'gtspace':
                    thr = gtspace_threshold(gt_norm) if gt_count > 0 else thr_value
                elif thr_mode == 'grid':
                    cell_diag = np.sqrt((1.0 / W2) ** 2 + (1.0 / H2) ** 2)
                    thr = max(cell_diag / 2, 0.05)
                else:
                    raise ValueError(f"unknown thr_mode {thr_mode}")

                tp, fp, fn = official_calc_loc_metric(centers, gt_norm, thr)
                tot_tp += tp; tot_fp += fp; tot_fn += fn
                counter += 1

            if counter % 200 < len(captions):
                p = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
                r = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
                f = 2 * p * r / (p + r) if p + r else 0
                print(f"\r  ({counter}) MAE={mae_sum/counter:.2f}, F1={f:.3f}", end='', flush=True)
    print()

    prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
    rec = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {
        'mae': mae_sum / counter, 'rmse': (rmse_sum / counter) ** 0.5,
        'precision': prec, 'recall': rec, 'f1': f1,
        'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn,
        'overcount_img_ratio': overcount_imgs / counter,
        'forced_fp': forced_fp, 'tot_gt': tot_gt,
        # 강제 FP가 전체 GT 대비 차지하는 비율 ≈ dedup으로 못 넘는 precision 손실 하한
        'forced_fp_ratio': forced_fp / max(tot_gt, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--thr_mode', type=str, default='fixed',
                        choices=['fixed', 'gtspace', 'grid'],
                        help='임계값 정책 (파일 상단 설명 참조)')
    parser.add_argument('--thr_value', type=float, default=0.05,
                        help='thr_mode=fixed 일 때의 고정 거리 임계값(정규화 좌표)')
    parser.add_argument('--nms_dist', type=float, default=-1.0)
    # localization 방식: sum=공간 NMS(기존), graph=학습된 inter_edges로 중복제거(N1)
    parser.add_argument('--loc_mode', type=str, default='sum', choices=['sum', 'graph'])
    parser.add_argument('--edge_tau', type=float, default=0.5,
                        help='graph 모드에서 same-object로 볼 inter_edge 임계값')
    # 디코더 아키텍처(체크포인트와 일치)
    parser.add_argument('--score_from_gat', type=int, default=1, choices=[0, 1])
    parser.add_argument('--class_heads', type=int, default=4)
    parser.add_argument('--attr_heads', type=int, default=4)
    parser.add_argument('--class_strength', type=float, default=1.0)
    parser.add_argument('--attr_strength', type=float, default=1.0)
    args = parser.parse_args()

    processor = DataProcessor()
    annotations = processor.annotations
    loader = get_loader(processor, args.split, 1)

    CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG_PATH, "./groundingdino_swint_ogc.pth").to(device)
    setup_freeze(model)

    step_a = StepA(d_model=256).to(device)
    decoder = REGraphDecoder(
        d_model=256,
        class_heads=args.class_heads, attr_heads=args.attr_heads,
        class_strength=args.class_strength, attr_strength=args.attr_strength,
        score_from_gat=bool(args.score_from_gat),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    step_a.load_state_dict(ckpt['step_a'])
    decoder.load_state_dict(ckpt['decoder'])
    print(f"Loaded: {args.checkpoint}")

    nms_dist = None if args.nms_dist <= 0 else args.nms_dist
    thr_desc = (f"fixed({args.thr_value})" if args.thr_mode == 'fixed'
                else args.thr_mode)
    print(f"Protocol: official calc_loc_metric | thr_mode={thr_desc} | "
          f"nms_dist={'adaptive' if nms_dist is None else nms_dist} | "
          f"loc_mode={args.loc_mode}" + (f"(tau={args.edge_tau})" if args.loc_mode == 'graph' else ""))

    res = evaluate_official(model, step_a, decoder, loader, annotations,
                            thr_mode=args.thr_mode, thr_value=args.thr_value,
                            nms_dist=nms_dist, loc_mode=args.loc_mode,
                            edge_tau=args.edge_tau)
    print("=" * 56)
    print(f"  [{args.split}] OFFICIAL-aligned  (thr={thr_desc})")
    print(f"  MAE={res['mae']:.2f}, RMSE={res['rmse']:.2f}")
    print(f"  F1={res['f1']:.3f}  (P={res['precision']:.3f}, R={res['recall']:.3f})")
    print(f"  TP={res['tp']}, FP={res['fp']}, FN={res['fn']}")
    print(f"  [over-count 천장] over-count 이미지 비율={res['overcount_img_ratio']*100:.1f}%, "
          f"강제FP={res['forced_fp']} (전체GT의 {res['forced_fp_ratio']*100:.1f}%)")
    print("=" * 56)


if __name__ == '__main__':
    main()
