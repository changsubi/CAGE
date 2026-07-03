"""Introduction teaser (Fig 1) on a REAL REC-8K image, GroundingREC/CAD-GD style.
Shows our matcher-free mechanism: class graph score (sigma^c) INTERSECTED with the
attribute graph score (sigma^a) = combined soft-AND (sigma^c*sigma^a) -> the referred
sub-population only, decoded to points WITHOUT any object queries or Hungarian matching.

Run several indices, pick the clearest image (a class with a discriminative attribute,
e.g. people sitting vs standing) for the paper:
  python3 make_teaser_real.py --split test --idx 40
  python3 make_teaser_real.py --split test --scan 0 60   # render a range to browse
"""
import sys, os, argparse
sys.path.append('GroundingDINO')
import torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from groundingdino.util.base_api import load_model
from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import REGraphDecoder, build_grid_graph, count_sum_guided
from train_regraph import setup_freeze, run_feature_enhancer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def heat(ax, img, score2d, H2, W2, cmap, title, tcolor):
    W_, H_ = img.size
    ax.imshow(img)
    ax.imshow(score2d.reshape(H2, W2), alpha=0.60, cmap=cmap,
              extent=[0, W_, H_, 0], aspect='auto')
    ax.set_title(title, fontsize=12, color=tcolor, fontweight='bold')
    ax.axis('off')


def render(img, cap, gt_n, pred_n, cls2d, att2d, comb2d, centers, H2, W2, out_png,
           tagline=True, k_cls=None, k_att=None):
    W_, H_ = img.size
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.6))
    # 0: query image + GT
    ax[0].imshow(img)
    ax[0].set_title(f'Query: "{cap}"\nreferring expression counting (GT = {gt_n})',
                    fontsize=12, color='#1b1f24')
    ax[0].axis('off')
    # 1: class graph (broad -> would over-count the whole class)
    ct = r'class graph  $\sigma^{c}$' + (f'  (sum$\\approx${k_cls}, over-counts)' if k_cls is not None else '')
    heat(ax[1], img, cls2d, H2, W2, 'Blues', ct, '#2e6ca8')
    # 2: attribute graph (also broad)
    at = r'attribute graph  $\sigma^{a}$' + (f'  (sum$\\approx${k_att}, over-counts)' if k_att is not None else '')
    heat(ax[2], img, att2d, H2, W2, 'Oranges', at, '#b5651d')
    # 3: combined soft-AND + predicted points
    heat(ax[3], img, comb2d, H2, W2, 'Greens', r'soft-AND  $\sigma^{c}\!\cdot\!\sigma^{a}$', '#2e8b57')
    for c in centers:
        ax[3].plot(c[0]*W_, c[1]*H_, 'o', ms=6, mfc='none', mec='#c0202a', mew=1.8)
    ax[3].set_title(r'soft-AND  $\sigma^{c}\!\cdot\!\sigma^{a}$' + f'  (Pred = {pred_n}, correct)',
                    fontsize=12, color='#2e8b57', fontweight='bold')
    if tagline:
        fig.text(0.5, 0.005,
                 'REGraph counts by intersecting two text-conditioned graphs — '
                 'no object queries, no Hungarian matching.',
                 ha='center', fontsize=12.5, color='#2e8b57', fontweight='bold')
    plt.tight_layout(rect=[0, 0.04 if tagline else 0.0, 1, 1])
    plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='reg_v2_s42/best.pth')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--idx', type=int, default=-1)
    ap.add_argument('--scan', type=int, nargs=2, default=None, help='scan a [start end) range to browse')
    ap.add_argument('--contains', type=str, default='',
                    help='only render samples whose expression contains this substring '
                         '(e.g. green / red / grape / car) — great for finding COLOR-attribute examples')
    ap.add_argument('--max', type=int, default=30, help='stop after this many rendered figures')
    ap.add_argument('--nms_dist', type=float, default=1.5)
    ap.add_argument('--outdir', default='teaser_figs')
    ap.add_argument('--paper', action='store_true',
                    help='publication mode: drop the baked-in bottom tagline (LaTeX caption carries it)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    lo, hi = (args.scan if args.scan else (args.idx, args.idx + 1))

    processor = DataProcessor(); annotations = processor.annotations
    loader = get_loader(processor, args.split, 1)
    CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG, "./groundingdino_swint_ogc.pth").to(device); setup_freeze(model)
    step_a = StepA(256).to(device); decoder = REGraphDecoder(256).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck['model']); step_a.load_state_dict(ck['step_a']); decoder.load_state_dict(ck['decoder'])
    model.eval(); step_a.eval(); decoder.eval()

    cache = {}; rendered = 0
    with torch.no_grad():
        for idx, (images, caps_list, shapes, img_caps_list) in enumerate(loader):
            if idx < lo:
                continue
            if idx >= hi or rendered >= args.max:
                break
            anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
            captions = [c for cl in caps_list for c in cl]
            if args.contains and args.contains.lower() not in captions[0].lower():
                continue  # skip non-matching BEFORE the expensive forward pass
            images_exp = torch.stack([images[i] for i, cl in enumerate(caps_list) for _ in cl]).to(device)
            feat = run_feature_enhancer(model, images_exp, captions, training=False)
            nodes, fe, ce, ae, H2, W2 = step_a(
                feat['img_memory'], feat['txt_memory'], feat['spatial_shapes'],
                feat['level_start_index'], feat['text_token_mask'], feat['text_class_mask'], feat['text_attr_mask'])
            key = (H2, W2)
            if key not in cache:
                ei, pos = build_grid_graph(H2, W2); cache[key] = (ei.to(device), pos.to(device))
            edge_index, positions = cache[key]
            b = 0
            gt_pts = anno_b[b]['points']; cap = captions[b]
            img_id = [ic for icl in img_caps_list for ic in icl][b][0]
            out = decoder(nodes[b], ce[b], ae[b], edge_index, positions)
            cls = out['class_scores'].cpu().numpy()
            att = out['attr_scores'].cpu().numpy()
            comb = out['combined_scores']
            k_cls = int(round(float(cls.sum())))   # what the class graph alone would count
            k_att = int(round(float(att.sum())))   # what the attribute graph alone would count
            _, centers, _ = count_sum_guided(comb.cpu(), positions.cpu(), H=H2, W=W2,
                                             offsets=out['offsets'].cpu(), min_dist_cells=args.nms_dist)
            img = Image.open(f"{processor.get_image_path()}/{img_id}").convert('RGB')
            out_png = os.path.join(args.outdir, f"teaser_{args.split}_{idx:04d}_gt{len(gt_pts)}_pred{len(centers)}.png")
            render(img, cap, len(gt_pts), len(centers), cls, att, comb.cpu().numpy(), centers, H2, W2, out_png,
                   tagline=not args.paper, k_cls=k_cls, k_att=k_att)
            print(f"[{idx}] {img_id} | '{cap}' | GT={len(gt_pts)} Pred={len(centers)} -> {out_png}")
            rendered += 1
    print(f"\nSaved {rendered} figures to {args.outdir}/  (pick the clearest class+attribute example for Fig 1)")


if __name__ == '__main__':
    main()
