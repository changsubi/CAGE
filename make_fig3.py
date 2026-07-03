"""Build Fig. 3 (qualitative results) for the CAGE paper directly from the trained model.

Fig. 3 is a RESULTS grid (not the mechanism breakdown, which is Fig. 1): each panel is a
real REC-8K image with CAGE's predicted points (red) and a "GT / Pred" count inset. The
figure should contain (i) a same-image, different-attribute PAIR (the sub-class-discrimination
highlight), (ii) a few diverse examples, and (iii) one dense-scene under-counting FAILURE.
All in-figure text is English (the REC-8K expressions are English).

Two steps:

1) Browse candidates to choose panels (prints idx | image | each expression with GT and Pred):
     python3 make_fig3.py --browse --browse_max 120
   Look for: an image with two different-attribute expressions of one class (the pair),
   a few clean diverse examples, and a high-count image where Pred < GT (the failure).

2) Render the grid from your chosen (image_index:expression_index) panels, in order:
     python3 make_fig3.py --select "12:0,12:1,40:0,67:1,88:0,103:0" --cols 3 \
         --failure_idx 5 --out fig3.png
   (--failure_idx marks that 0-based panel with a red border + "failure"; --show_gt also
    plots ground-truth points in green.)
"""
import sys, os, argparse, math
sys.path.append('GroundingDINO')  # must precede utils/* imports
import torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from groundingdino.util.base_api import load_model
from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import REGraphDecoder, build_grid_graph, count_sum_guided
from train_regraph import setup_freeze, run_feature_enhancer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_all(checkpoint):
    processor = DataProcessor()
    annotations = processor.annotations
    loader = get_loader(processor, 'test', 1)
    CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG, "./groundingdino_swint_ogc.pth").to(device); setup_freeze(model)
    step_a = StepA(256).to(device); decoder = REGraphDecoder(256).to(device)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck['model']); step_a.load_state_dict(ck['step_a']); decoder.load_state_dict(ck['decoder'])
    model.eval(); step_a.eval(); decoder.eval()
    return processor, annotations, loader, model, step_a, decoder


def infer_image(model, step_a, decoder, images, caps_list, cache, nms_dist):
    """Run all expressions of one image; return list of dicts per expression."""
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
    outs = []
    for e in range(len(captions)):
        out = decoder(nodes[e], ce[e], ae[e], edge_index, positions)
        _, centers, _ = count_sum_guided(out['combined_scores'].cpu(), positions.cpu(), H=H2, W=W2,
                                          offsets=out['offsets'].cpu(), min_dist_cells=nms_dist)
        outs.append({'caption': captions[e], 'pred_xy': [(float(c[0]), float(c[1])) for c in centers],
                     'pred': len(centers)})
    return outs


def draw_panel(ax, img, query, gt, pred, pred_xy, gt_xy, is_failure, show_gt):
    W_, H_ = img.size
    ax.imshow(img)
    if show_gt and gt_xy is not None:
        for (x, y) in gt_xy:
            ax.plot(x, y, 'o', ms=5, mfc='none', mec='lime', mew=1.2)
    for (x, y) in pred_xy:
        ax.plot(x * W_, y * H_, 'o', ms=6, mfc='none', mec='red', mew=1.6)
    ax.set_title(f'"{query}"' + ('  (failure)' if is_failure else ''),
                 fontsize=10, color='#b00020' if is_failure else '#1b1f24')
    ax.text(0.03, 0.04, f'GT {gt} | Pred {pred}', transform=ax.transAxes, fontsize=10,
            color='white', ha='left', va='bottom',
            bbox=dict(facecolor='black', alpha=0.55, pad=2.5, edgecolor='none'))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(is_failure); s.set_color('#b00020'); s.set_linewidth(2.5)
    if not is_failure:
        ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='reg_v2_s42/best.pth')
    ap.add_argument('--nms_dist', type=float, default=1.5)
    ap.add_argument('--browse', action='store_true', help='save one preview IMAGE per candidate then exit')
    ap.add_argument('--browse_max', type=int, default=120)
    ap.add_argument('--browse_dir', default='fig3_candidates', help='folder to save preview panels for selection')
    ap.add_argument('--exclude', default='', help='comma substrings; skip expressions whose caption contains any (e.g. person,people,car)')
    ap.add_argument('--contains', default='', help='comma substrings; keep only expressions whose caption contains one')
    ap.add_argument('--max_rel_err', type=float, default=0.15,
                    help='browse: only save panels where |GT-Pred|/GT <= this (accurate predictions). Set 1.0 to keep all.')
    ap.add_argument('--min_gt', type=int, default=3, help='browse: skip panels with GT below this')
    ap.add_argument('--select', type=str, default='',
                    help='comma list of image_index:expression_index, e.g. "12:0,12:1,40:0,67:1,88:0,103:0"')
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--failure_idx', type=int, default=-1, help='0-based panel to mark as failure')
    ap.add_argument('--show_gt', action='store_true', help='also plot ground-truth points (green)')
    ap.add_argument('--out', default='fig3.png')
    args = ap.parse_args()

    processor, annotations, loader, model, step_a, decoder = load_all(args.checkpoint)
    cache = {}

    excl = [s.strip().lower() for s in args.exclude.split(',') if s.strip()]
    cont = [s.strip().lower() for s in args.contains.split(',') if s.strip()]

    def keep(caption):
        cl = caption.lower()
        if excl and any(s in cl for s in excl):
            return False
        if cont and not any(s in cl for s in cont):
            return False
        return True

    if args.browse or not args.select:
        os.makedirs(args.browse_dir, exist_ok=True)
        n_panels = 0
        with torch.no_grad():
            for idx, (images, caps_list, shapes, img_caps_list) in enumerate(loader):
                if idx >= args.browse_max:
                    break
                img_ids = [ic for icl in img_caps_list for ic in icl]
                anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
                img = Image.open(f"{processor.get_image_path()}/{img_ids[0][0]}").convert('RGB')
                outs = infer_image(model, step_a, decoder, images, caps_list, cache, args.nms_dist)
                for e, o in enumerate(outs):
                    if not keep(o['caption']):
                        continue
                    gt_pts = anno_b[e]['points']; gt = len(gt_pts)
                    if gt < args.min_gt or abs(gt - o['pred']) > args.max_rel_err * max(gt, 1):
                        continue  # only save accurate (GT ~= Pred) panels
                    fig, ax = plt.subplots(figsize=(4.2, 4.2))
                    # preview shows BOTH predicted (red) and GT (green) points so accuracy is visible
                    draw_panel(ax, img, o['caption'], gt, o['pred'], o['pred_xy'],
                               [(p[0], p[1]) for p in gt_pts], is_failure=False, show_gt=True)
                    fname = os.path.join(args.browse_dir,
                                         f'{idx:04d}_{e}__gt{gt}_pred{o["pred"]}.png')
                    plt.savefig(fname, dpi=110, bbox_inches='tight'); plt.close()
                    n_panels += 1
                if idx % 10 == 0:
                    print(f'  ...scanned {idx} images, saved {n_panels} preview panels')
        print(f'\nSaved {n_panels} preview images to {args.browse_dir}/')
        print('Open the folder, look at the predictions (red=pred, green=GT), and note the filenames.')
        print('Filename = IMAGEINDEX_EXPR__gtN_predM.png  ->  use IMAGEINDEX:EXPR in --select.')
        print('Two files with the SAME image index (e.g. 0012_0 and 0012_1) are a same-image pair.')
        print('Then:  python3 make_fig3.py --select "12:0,12:1,40:0,67:1,88:0,103:0" --cols 3 --failure_idx 5')
        return

    # ---- render selected panels ----
    order = [(int(t.split(':')[0]), int(t.split(':')[1])) for t in args.select.split(',') if t.strip()]
    want_imgs = set(i for i, _ in order)
    got = {}
    with torch.no_grad():
        for idx, (images, caps_list, shapes, img_caps_list) in enumerate(loader):
            if idx not in want_imgs:
                continue
            img_ids = [ic for icl in img_caps_list for ic in icl]
            anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
            img = Image.open(f"{processor.get_image_path()}/{img_ids[0][0]}").convert('RGB')
            outs = infer_image(model, step_a, decoder, images, caps_list, cache, args.nms_dist)
            for e in set(ex for i, ex in order if i == idx):
                gt_pts = anno_b[e]['points']
                got[(idx, e)] = dict(image=img, query=outs[e]['caption'], gt=len(gt_pts),
                                     pred=outs[e]['pred'], pred_xy=outs[e]['pred_xy'],
                                     gt_xy=[(p[0], p[1]) for p in gt_pts])
            if want_imgs.issubset(set(k[0] for k in got)):
                break

    panels = [got[k] for k in order if k in got]
    n = len(panels)
    cols = args.cols
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.1))
    axes = axes.flatten() if n > 1 else [axes]
    for i, p in enumerate(panels):
        draw_panel(axes[i], p['image'], p['query'], p['gt'], p['pred'], p['pred_xy'], p['gt_xy'],
                   is_failure=(i == args.failure_idx), show_gt=args.show_gt)
    for j in range(n, rows * cols):
        axes[j].axis('off')
    plt.tight_layout(pad=0.6)
    plt.savefig(args.out, dpi=170, bbox_inches='tight'); plt.close()
    print(f"Saved {args.out}  ({n} panels, {rows}x{cols})")
    for i, p in enumerate(panels):
        tag = '  <-- failure' if i == args.failure_idx else ''
        print(f'  panel {i}: "{p["query"]}"  GT {p["gt"]} | Pred {p["pred"]}{tag}')


if __name__ == '__main__':
    main()
