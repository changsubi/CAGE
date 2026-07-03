"""
Inference-time ablations from a SINGLE trained checkpoint (no re-training).
================================================================================
REGraph's decoder returns class/attribute/combined scores, offsets, and
intersection edges, so several core design choices can be ablated purely at
evaluation. This reuses the official-aligned matching/metric of eval_official.py
and only swaps which quantity drives counting + localization.

Ablations exposed (all on the same reg_v2_s42/best.pth):
  --score_source {combined,class,attr,class_ref,attr_ref}
        combined = soft-AND (sigma_c * sigma_a)  [full model]
        class    = class graph only  (no conjunction -> over-counts the class)
        attr     = attribute graph only
        *_ref    = post-message-passing refined scores
  --use_offset {1,0}          : sub-cell offset head on/off (localization)
  --score_from_gat {1,0}      : combined from GAT-refined vs raw node scores
  --loc_mode {sum,graph}      : density-integration vs learned-graph decoding
  --thr_mode {fixed,gtspace,grid}, --thr_value, --nms_dist  (as in eval_official)

Usage (full model = defaults):
  python3 eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test \
      --thr_mode fixed --thr_value 0.05 --loc_mode sum --nms_dist 1.5
  # soft-AND ablation:
  python3 eval_ablation.py ... --score_source class
"""
import argparse, sys
import numpy as np
import torch

sys.path.append('GroundingDINO')  # must precede any utils/* import that needs groundingdino
from groundingdino.util.base_api import load_model

from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import (
    REGraphDecoder, build_grid_graph, count_sum_guided, count_graph_guided,
)
from train_regraph import run_feature_enhancer, setup_freeze
from eval_official import official_calc_loc_metric, gtspace_threshold, device

SCORE_KEY = {
    'combined': 'combined_scores',
    'class': 'class_scores',
    'attr': 'attr_scores',
    'class_ref': 'class_scores_refined',
    'attr_ref': 'attr_scores_refined',
}


def evaluate_ablation(model, step_a, decoder, loader, annotations,
                      thr_mode='fixed', thr_value=0.05, nms_dist=None,
                      loc_mode='sum', edge_tau=0.5,
                      score_source='combined', use_offset=True):
    model.eval(); step_a.eval(); decoder.eval()
    key_name = SCORE_KEY.get(score_source)  # None for combine modes (mean/min/max)
    mae_sum = rmse_sum = 0.0
    tot_tp = tot_fp = tot_fn = 0
    counter = 0
    graph_cache = {}
    overcount_imgs = 0; forced_fp = 0; tot_gt = 0

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

            k = (H2, W2)
            if k not in graph_cache:
                ei, pos = build_grid_graph(H2, W2)
                graph_cache[k] = (ei.to(device), pos.to(device))
            edge_index, positions = graph_cache[k]

            for b in range(len(captions)):
                gt_pts = anno_b[b]['points']; gt_count = len(gt_pts)
                out = decoder(nodes[b], class_emb[b], attr_emb[b], edge_index, positions)

                if score_source in ('mean', 'min', 'max'):
                    # combine the SAME refined scores the product (combined) uses,
                    # to isolate the choice of t-norm/aggregation (product vs mean/min/max).
                    # NOTE: do not name locals 'b' here — 'b' is the batch-loop index.
                    s_c, s_a = out['class_scores_refined'], out['attr_scores_refined']
                    if score_source == 'mean':
                        score = 0.5 * (s_c + s_a)
                    elif score_source == 'min':
                        score = torch.minimum(s_c, s_a)
                    else:
                        score = torch.maximum(s_c, s_a)
                else:
                    score = out[key_name]                   # <-- ablated quantity
                offs = out['offsets'] if use_offset else None

                ps = score.sum().item()
                mae_sum += abs(ps - gt_count)
                rmse_sum += (ps - gt_count) ** 2
                K = max(0, int(round(ps)))
                if K > gt_count:
                    overcount_imgs += 1; forced_fp += (K - gt_count)
                tot_gt += gt_count

                if loc_mode == 'graph':
                    gd = nms_dist if nms_dist is not None else 1.0
                    _, centers, _ = count_graph_guided(
                        score.cpu(), positions.cpu(), edge_index.cpu(),
                        out['inter_edges'].cpu(), H=H2, W=W2,
                        offsets=(offs.cpu() if offs is not None else None),
                        edge_tau=edge_tau, min_dist_cells=gd)
                else:
                    _, centers, _ = count_sum_guided(
                        score.cpu(), positions.cpu(), H=H2, W=W2,
                        offsets=(offs.cpu() if offs is not None else None),
                        min_dist_cells=nms_dist)

                if gt_count > 0:
                    gt_norm = np.array(gt_pts, dtype=np.float64)
                    gt_norm[:, 0] /= shapes_exp[b][1]
                    gt_norm[:, 1] /= shapes_exp[b][0]
                else:
                    gt_norm = np.zeros((0, 2))

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
    prec = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else 0
    rec = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {'mae': mae_sum / counter, 'rmse': (rmse_sum / counter) ** 0.5,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tot_tp, 'fp': tot_fp, 'fn': tot_fn,
            'overcount_img_ratio': overcount_imgs / counter,
            'forced_fp_ratio': forced_fp / max(tot_gt, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--thr_mode', default='fixed', choices=['fixed', 'gtspace', 'grid'])
    p.add_argument('--thr_value', type=float, default=0.05)
    p.add_argument('--nms_dist', type=float, default=-1.0)
    p.add_argument('--loc_mode', default='sum', choices=['sum', 'graph'])
    p.add_argument('--edge_tau', type=float, default=0.5)
    p.add_argument('--score_source', default='combined',
                   choices=['combined', 'class', 'attr', 'class_ref', 'attr_ref',
                            'mean', 'min', 'max'])
    p.add_argument('--use_offset', type=int, default=1, choices=[0, 1])
    p.add_argument('--score_from_gat', type=int, default=1, choices=[0, 1])
    p.add_argument('--class_heads', type=int, default=4)
    p.add_argument('--attr_heads', type=int, default=4)
    p.add_argument('--class_strength', type=float, default=1.0)
    p.add_argument('--attr_strength', type=float, default=1.0)
    args = p.parse_args()

    processor = DataProcessor()
    annotations = processor.annotations
    loader = get_loader(processor, args.split, 1)

    CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG_PATH, "./groundingdino_swint_ogc.pth").to(device)
    setup_freeze(model)
    step_a = StepA(d_model=256).to(device)
    decoder = REGraphDecoder(
        d_model=256, class_heads=args.class_heads, attr_heads=args.attr_heads,
        class_strength=args.class_strength, attr_strength=args.attr_strength,
        score_from_gat=bool(args.score_from_gat)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    step_a.load_state_dict(ckpt['step_a'])
    decoder.load_state_dict(ckpt['decoder'])

    nms_dist = None if args.nms_dist <= 0 else args.nms_dist
    res = evaluate_ablation(model, step_a, decoder, loader, annotations,
                            thr_mode=args.thr_mode, thr_value=args.thr_value,
                            nms_dist=nms_dist, loc_mode=args.loc_mode,
                            edge_tau=args.edge_tau, score_source=args.score_source,
                            use_offset=bool(args.use_offset))
    tag = (f"score={args.score_source} offset={args.use_offset} gat={args.score_from_gat} "
           f"loc={args.loc_mode} nms={'adaptive' if nms_dist is None else nms_dist} "
           f"thr={args.thr_mode}({args.thr_value if args.thr_mode=='fixed' else ''})")
    print("=" * 60)
    print(f"  [{args.split}] ABLATION | {tag}")
    print(f"  MAE={res['mae']:.2f}, RMSE={res['rmse']:.2f}")
    print(f"  F1={res['f1']:.3f}  (P={res['precision']:.3f}, R={res['recall']:.3f})")
    print(f"  overcount_img={res['overcount_img_ratio']*100:.1f}%, "
          f"forced_FP_ratio={res['forced_fp_ratio']*100:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
