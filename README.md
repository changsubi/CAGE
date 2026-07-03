# CAGE: Matcher-Free Referring Expression Counting via Class-Attribute Graph Intersection

Official implementation of **CAGE**, the first *matcher-free* (query-free) model for
Referring Expression Counting (REC). CAGE keeps a frozen GroundingDINO backbone with a
fine-tuned cross-modal enhancer, but **discards the object-query decoder, box regressor,
and Hungarian matcher** used by every prior REC method. Instead it lays an 8-connected
lattice over the stride-16 image tokens and runs two text-conditioned graphs, a **class**
graph and an **attribute** graph, whose node scores are combined by a score-level product
(a soft logical AND). The count is a density integral, decoded with no object queries and
no matching.

On REC-8K, CAGE reaches the second-best counting error (test MAE **4.33**) with a task
head of only **1.37M** parameters (13.7x smaller than the detection decoder it replaces)
and no fixed count ceiling.

> **Naming.** This code base predates the paper title and uses the internal name
> **`REGraph`** (e.g. `REGraphDecoder`, the `regraph/` package, `reg_v2_s42/`). `REGraph`
> and `CAGE` refer to the same model; identifiers were left unchanged to keep the released
> checkpoint loadable.

> **Anonymity (double-blind review).** This repository is anonymized and is submitted
> directly as supplementary material (no external links). Trained weights are not included
> due to the supplementary size limit; all results reproduce from this code and the exact
> configuration in `reg_v2_s42/results.txt`. The checkpoint will be released publicly upon
> publication.

---

## 1. Repository structure

```
.
├── train_regraph.py          # training entry point (CAGE)
├── regraph/
│   ├── step_a.py             # StepA: text-conditioned multi-scale node construction
│   ├── modules.py            # dual class/attribute graphs, soft-AND, read-out, decoding
│   └── __init__.py
├── utils/
│   ├── processor.py          # REC-8K annotation / split loading (set REC8K_ROOT here)
│   ├── image_loader.py       # data loader (one image with all its expressions)
│   └── criterion.py          # matching / localization metrics
├── eval_official.py          # official REC-8K localization protocol (box-threshold F1)
├── eval_ablation.py          # inference-time ablations from one checkpoint (Table 2)
├── count_params.py           # parameter accounting (efficiency, Table in Sec. 4.3)
├── strat_by_count.py         # counting error stratified by GT count (supplement)
├── make_fig3.py              # qualitative figure (paper Fig. 3)
├── make_teaser_real.py       # teaser figure (paper Fig. 1)
├── GroundingDINO/            # vendored frozen backbone (third-party, Apache-2.0)
├── reg_v2_s42/
│   └── results.txt           # exact config + metrics of the released run
├── requirements.txt
└── README.md
```

The trained weights (`reg_v2_s42/best.pth`), the GroundingDINO backbone weights, and the
REC-8K images/annotations are **not** committed; see below for how to obtain them.

---

## 2. Installation

```bash
conda create -n cage python=3.9 -y
conda activate cage
# install a torch build matching your CUDA, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# build the vendored GroundingDINO ops
cd GroundingDINO && pip install -e . && cd ..
```

Download the GroundingDINO Swin-T backbone weights and place them at the repository root
as `groundingdino_swint_ogc.pth`:

```bash
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Training and evaluation were run on a single NVIDIA A100 (80 GB).

---

## 3. Data (REC-8K)

CAGE uses the public **REC-8K** benchmark introduced by *Referring Expression Counting*
(GroundingREC, CVPR 2024). We do not redistribute it. Obtain the images and annotations
from the official release:

- REC-8K / GroundingREC: https://github.com/sydai/referring-expression-counting

Expected layout:

```
<REC8K_ROOT>/
└── rec-8k/                 # REC-8K images
anno/
├── annotations.json        # REC-8K annotations (per image -> per expression)
└── splits.json             # official 10,555 / 3,336 / 3,231 train/val/test split
```

Point the code at your image directory (the parent of `rec-8k/`) via an environment
variable (no code edit needed):

```bash
export REC8K_ROOT=/path/to/rec8k_parent   # default is ./data
```

and place `annotations.json` / `splits.json` under `anno/` at the repository root.

---

## 4. Pretrained checkpoint

Trained weights are **not** bundled with this anonymized submission (they exceed the
supplementary size limit, and external links are not used for review). Reviewers can
reproduce every reported number by training with the command in Section 5.1;
`reg_v2_s42/results.txt` records the exact configuration and the metrics the released run
reproduces. To evaluate an existing checkpoint, place it at `reg_v2_s42/best.pth`. The
weights will be released publicly upon publication.

---

## 5. Reproducing the paper

### 5.1 Train CAGE (single fixed seed = 42)

```bash
python train_regraph.py \
    --seed 42 --epochs 30 --lr 1e-5 --lr_graph 5e-5 \
    --class_heads 4 --attr_heads 4 --class_strength 1.0 --attr_strength 1.0 \
    --score_from_gat 1 --density_sigma 0.55 --nms_dist 1.5 \
    --count_norm sqrt --overcount_w 2.0 --count_weight 1.0 \
    --offset_weight 1.0 --select_by f1 \
    --results_dir ./reg_v2_s42
```

Cross-graph hard-negative mining (CHNM) is **on** by default; add `--no_chnm 1` for the
*w/o CHNM* ablation.

### 5.2 Evaluate (Table 1, main results)

```bash
# official-protocol localization + threshold-free counting
python eval_official.py --checkpoint reg_v2_s42/best.pth --split test
python eval_official.py --checkpoint reg_v2_s42/best.pth --split val
```

### 5.3 Ablation study (Table 2)

All rows come from the **same** checkpoint by swapping the integrated score (no retraining):

```bash
python eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test                       # CAGE (full, product)
python eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test --score_source class   # class graph only
python eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test --score_source attr    # attribute graph only
python eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test --score_source mean    # mean conjunction
python eval_ablation.py --checkpoint reg_v2_s42/best.pth --split test --score_source min     # min conjunction
```

The *w/o CHNM* row requires the separately trained checkpoint from `--no_chnm 1`.

### 5.4 Efficiency (Sec. 4.3) and stratified counting (supplement)

```bash
python count_params.py                                                   # 1.37M task head vs 18.8M decoder (13.7x)
python strat_by_count.py --checkpoint reg_v2_s42/best.pth --split test    # MAE binned by GT count; max GT = 1004
```

### 5.5 Figures

```bash
python make_teaser_real.py                       # Fig. 1 (teaser)
python make_fig3.py --browse                     # save candidate panels, then:
python make_fig3.py --select "<img:expr,...>"    # render the qualitative grid (Fig. 3)
```

---

## 6. Expected results (REC-8K test)

| Metric | Value |
|---|---|
| Counting MAE (sum decoding) | **4.33** |
| Counting RMSE | 15.57 |
| Localization F1 (point threshold*) | 0.679 (P 0.714 / R 0.648) |

Ablation (test MAE / precision):

| Integrated score | MAE | P |
|---|---|---|
| class graph only | 411.5 | 0.04 |
| attribute graph only | 261.6 | 0.06 |
| mean conjunction | 95.8 | 0.14 |
| min conjunction | 22.5 | 0.38 |
| **product (CAGE, full)** | **4.33** | **0.71** |
| w/o CHNM | 4.70 | 0.70 |

\*Because CAGE predicts points, not boxes, its localization F1 uses a distance threshold
defined from the ground-truth object scale and is **not** directly comparable to the
box-derived-threshold F1 reported by detector baselines. Counting MAE/RMSE are
threshold-free and directly comparable. All numbers come from a single fixed-seed run.

---

## 7. Key hyperparameters

| Item | Value |
|---|---|
| Feature width `d` | 256 |
| Grid (stride-16) | 50 x 84 = 4,200 nodes, 8-connected |
| Message passing | GATv2, 1 round; class heads 4, attribute heads 4 |
| Residual ratio | 0.3 |
| Density bandwidth (sigma) | 0.55 x cell diagonal |
| Count normalization / over-count weight | sqrt / 2.0 |
| CHNM loss weight | 0.5 |
| Optimizer | AdamW, lr 1e-5 (enhancer) / 5e-5 (graph head) |
| Epochs / batch / seed | 30 / one image + all its expressions / 42 |
| Decoding | count = round(sum of combined scores); NMS radius 1.5 cells; sub-cell offset |

---

## 8. License and acknowledgments

This project is released under the MIT License (see `LICENSE`). It builds on:

- **GroundingDINO** (Apache-2.0), vendored under `GroundingDINO/` with its original license.
- **REC-8K / GroundingREC** (CVPR 2024), the benchmark and evaluation protocol.

We thank the authors of these works. Dataset copyrights remain with their original owners.
