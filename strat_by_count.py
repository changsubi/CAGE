"""Stratified error analysis by ground-truth count (for the 'no query ceiling' claim).
Bins the REC-8K test set by GT instance count and reports per-bin MAE for CAGE,
plus the maximum GT count (to check whether any scene exceeds the 900-query budget
of the detector baselines). Inference-only, uses reg_v2_s42/best.pth.

Usage:
  python3 strat_by_count.py --checkpoint reg_v2_s42/best.pth --split test --nms_dist 1.5
"""
import argparse, sys
import numpy as np
import torch

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import load_model
from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import REGraphDecoder, build_grid_graph
from train_regraph import run_feature_enhancer, setup_freeze

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BINS = [(1, 10), (11, 50), (51, 100), (101, 10 ** 9)]
BIN_NAMES = ['1-10', '11-50', '51-100', '>100']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='reg_v2_s42/best.pth')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--nms_dist', type=float, default=1.5)  # unused for counting; kept for parity
    args = p.parse_args()

    processor = DataProcessor()
    annotations = processor.annotations
    loader = get_loader(processor, args.split, 1)
    CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG, "./groundingdino_swint_ogc.pth").to(device); setup_freeze(model)
    step_a = StepA(256).to(device); decoder = REGraphDecoder(256).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model']); step_a.load_state_dict(ck['step_a']); decoder.load_state_dict(ck['decoder'])
    model.eval(); step_a.eval(); decoder.eval()

    # per-bin accumulators: [abs_err_sum, sq_err_sum, n, gt_sum, pred_sum]
    acc = {i: [0.0, 0.0, 0, 0.0, 0.0] for i in range(len(BINS))}
    max_gt = 0
    cache = {}
    with torch.no_grad():
        for images, caps_list, shapes, img_caps_list in loader:
            anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
            captions = [c for cl in caps_list for c in cl]
            images_exp = torch.stack([images[i] for i, cl in enumerate(caps_list) for _ in cl]).to(device)
            feat = run_feature_enhancer(model, images_exp, captions, training=False)
            nodes, fe, ce, ae, H2, W2 = step_a(
                feat['img_memory'], feat['txt_memory'], feat['spatial_shapes'],
                feat['level_start_index'], feat['text_token_mask'], feat['text_class_mask'], feat['text_attr_mask'])
            key = (H2, W2)
            if key not in cache:
                ei, pos = build_grid_graph(H2, W2); cache[key] = (ei.to(device), pos.to(device))
            edge_index, positions = cache[key]
            for b in range(len(captions)):
                gt = len(anno_b[b]['points'])
                out = decoder(nodes[b], ce[b], ae[b], edge_index, positions)
                pred = float(out['combined_scores'].sum().item())
                max_gt = max(max_gt, gt)
                for i, (lo, hi) in enumerate(BINS):
                    if lo <= gt <= hi:
                        e = abs(pred - gt)
                        acc[i][0] += e; acc[i][1] += e * e; acc[i][2] += 1
                        acc[i][3] += gt; acc[i][4] += pred
                        break

    print("=" * 74)
    print(f"  Stratified counting error by GT count  |  {args.split}  |  {args.checkpoint}")
    print("=" * 74)
    print(f"  {'bin':>8} {'#img':>6} {'meanGT':>8} {'meanPred':>9} {'MAE':>8} {'RMSE':>8}")
    for i in range(len(BINS)):
        s, sq, n, gsum, psum = acc[i]
        if n == 0:
            print(f"  {BIN_NAMES[i]:>8} {0:>6}      --        --       --       --")
            continue
        print(f"  {BIN_NAMES[i]:>8} {n:>6} {gsum/n:>8.1f} {psum/n:>9.1f} {s/n:>8.2f} {(sq/n)**0.5:>8.2f}")
    print("-" * 74)
    print(f"  max GT count in {args.split} = {max_gt}  "
          f"({'EXCEEDS' if max_gt > 900 else 'within'} the 900-query budget of detector baselines)")
    print("=" * 74)


if __name__ == '__main__':
    main()
