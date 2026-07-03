"""
REGraph Full Training Script (v2 - Feature Enhancer fine-tuning)
=================================================================
GroundingREC와 동일하게 Backbone + BERT만 freeze.
Feature Enhancer, input_proj, feat_map → fine-tuning.
"""

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, copy, argparse, random
from datetime import datetime


def set_seed(seed):
    """재현성을 위한 시드 고정 (GPU에서 bit-exact는 아니지만 변동성 크게 감소)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

sys.path.append('GroundingDINO')
from groundingdino.util.base_api import load_model
from groundingdino.util.misc import nested_tensor_from_tensor_list, NestedTensor
from groundingdino.models.GroundingDINO.groundingdino import split_caption
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map

from utils.processor import DataProcessor
from utils.image_loader import get_loader
from regraph.step_a import StepA
from regraph.modules import (
    build_grid_graph, REGraphDecoder,
    generate_gt_labels, count_connected_components, count_peaks_nms,
    count_sum_guided, discriminative_loss,
)
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Freeze 설정 (GroundingREC과 동일)
# ============================================================

def setup_freeze(model):
    """Backbone + BERT만 freeze. 나머지(Feature Enhancer, input_proj, feat_map)는 fine-tuning."""
    # Backbone (Swin-T) freeze
    for param in model.backbone.parameters():
        param.requires_grad = False

    # BERT freeze
    for param in model.bert.parameters():
        param.requires_grad = False

    # 나머지는 fine-tuning:
    # - model.transformer.encoder (Feature Enhancer 6 layers)
    # - model.input_proj (level별 projection)
    # - model.feat_map (BERT → d_model projection)
    # - model.transformer.level_embed

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  GroundingDINO: {trainable:,} trainable, {frozen:,} frozen")
    return trainable


# ============================================================
# Feature Enhancer 실행 (fine-tuning 가능)
# ============================================================

def run_feature_enhancer(model, images, captions, training=False):
    """Feature Enhancer 실행. training=True이면 gradient 활성화."""

    # Text 전처리 (tokenize는 gradient 불필요)
    with torch.no_grad():
        subjects, contexts, attributes = [], [], []
        for cap in captions:
            s, c, a = split_caption(cap, model.anno)
            subjects.append(s); contexts.append(c); attributes.append(a)

        class_texts, attr_texts = [], []
        for cap in captions:
            cap_clean = cap.rstrip('.')
            found = False
            for c, items in model.anno.items():
                if c.lower() == cap_clean.lower():
                    class_texts.append(items['class'] + '.')
                    attr_texts.append(items['attribute'] + '.')
                    found = True
                    break
            if not found:
                class_texts.append(cap); attr_texts.append(cap)

        tokenized_class = model.tokenizer(class_texts, padding="longest", return_tensors="pt").to(images.device)
        tokenized_attr = model.tokenizer(attr_texts, padding="longest", return_tensors="pt").to(images.device)
        tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt").to(images.device)

        (text_masks, pos_ids, _) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, model.specical_tokens, model.tokenizer)

        if text_masks.shape[1] > model.max_text_len:
            text_masks = text_masks[:, :model.max_text_len, :model.max_text_len]
            pos_ids = pos_ids[:, :model.max_text_len]
            for k in ["input_ids", "attention_mask", "token_type_ids"]:
                tokenized[k] = tokenized[k][:, :model.max_text_len]

    # 이하는 gradient 필요 (training 시)
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        if model.sub_sentence_present:
            tok_enc = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tok_enc["attention_mask"] = text_masks; tok_enc["position_ids"] = pos_ids
        else:
            tok_enc = tokenized

        # BERT (frozen이므로 no_grad)
        with torch.no_grad():
            bert_out = model.bert(**tok_enc)

        # feat_map (fine-tuning)
        enc_text = model.feat_map(bert_out["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        # Masks
        mask = tokenized["attention_mask"].bool()
        text_class_mask = (tokenized["input_ids"].unsqueeze(2) == tokenized_class["input_ids"].unsqueeze(1)).any(dim=2) & mask
        text_attr_mask = (tokenized["input_ids"].unsqueeze(2) == tokenized_attr["input_ids"].unsqueeze(1)).any(dim=2) & mask

        special_ids = set(model.specical_tokens)
        for b in range(tokenized["input_ids"].shape[0]):
            for pos in range(tokenized["input_ids"].shape[1]):
                if tokenized["input_ids"][b, pos].item() in special_ids:
                    text_class_mask[b, pos] = False
                    text_attr_mask[b, pos] = False

        if enc_text.shape[1] > model.max_text_len:
            enc_text = enc_text[:, :model.max_text_len, :]
            text_token_mask = text_token_mask[:, :model.max_text_len]
            text_class_mask = text_class_mask[:, :model.max_text_len]
            text_attr_mask = text_attr_mask[:, :model.max_text_len]

        text_dict = {"encoded_text": enc_text, "text_token_mask": text_token_mask,
                     "position_ids": pos_ids, "text_self_attention_masks": text_masks}

        # Image — Backbone (frozen이므로 no_grad)
        samples = images
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        with torch.no_grad():
            features, poss = model.backbone(samples)

        # input_proj (fine-tuning)
        srcs, masks_list = [], []
        for l, feat in enumerate(features):
            src, mask_l = feat.decompose()
            srcs.append(model.input_proj[l](src)); masks_list.append(mask_l)
        if model.num_feature_levels > len(srcs):
            for l in range(len(srcs), model.num_feature_levels):
                if l == len(srcs):
                    src = model.input_proj[l](features[-1].tensors)
                else:
                    src = model.input_proj[l](srcs[-1])
                m = samples.mask
                mask_l = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = model.backbone[1](NestedTensor(src, mask_l)).to(src.dtype)
                srcs.append(src); masks_list.append(mask_l); poss.append(pos_l)

        src_flat, mask_flat_list, spatial_shapes = [], [], []
        for lvl, (src, mask_l, pos) in enumerate(zip(srcs, masks_list, poss)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))
            src_flat.append(src.flatten(2).transpose(1, 2))
            mask_flat_list.append(mask_l.flatten(1))
        src_flat = torch.cat(src_flat, 1)
        mask_flat = torch.cat(mask_flat_list, 1)
        ss_t = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flat.device)
        lsi = torch.cat((ss_t.new_zeros((1,)), ss_t.prod(1).cumsum(0)[:-1]))

        lvl_pos = []
        for lvl, (src, mask_l, pos) in enumerate(zip(srcs, masks_list, poss)):
            pf = pos.flatten(2).transpose(1, 2)
            if model.transformer.num_feature_levels > 1 and model.transformer.level_embed is not None:
                pf = pf + model.transformer.level_embed[lvl].view(1, 1, -1)
            lvl_pos.append(pf)
        lvl_pos = torch.cat(lvl_pos, 1)
        valid_ratios = torch.stack([model.transformer.get_valid_ratio(m) for m in masks_list], 1)

        # Feature Enhancer (fine-tuning)
        memory, memory_text = model.transformer.encoder(
            src_flat, pos=lvl_pos, level_start_index=lsi, spatial_shapes=ss_t,
            valid_ratios=valid_ratios, key_padding_mask=mask_flat,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )

    return {
        'img_memory': memory, 'txt_memory': memory_text,
        'spatial_shapes': spatial_shapes, 'level_start_index': lsi,
        'text_token_mask': text_token_mask,
        'text_class_mask': text_class_mask, 'text_attr_mask': text_attr_mask,
    }


# ============================================================
# Compute Loss
# ============================================================

def compute_loss(out, node_labels, edge_labels, node_assignments, gt_count,
                 nodes_final, gt_points_norm, positions, H, W,
                 density_sigma_scale=0.7, offset_weight=1.0,
                 edge_index=None, ba_inter=False, inter_bnd_w=3.0,
                 count_norm='linear', overcount_w=1.0, count_weight=1.0,
                 chnm_weight=0.5):
    dev = node_labels.device

    def weighted_bce(pred, target):
        pos = target.sum().clamp(min=1)
        neg = (1 - target).sum().clamp(min=1)
        w = torch.where(target > 0, neg / pos, torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=w)

    # class/attr sub-classifiers: binary BCE (각 토큰이 class/attribute에 해당하는지)
    loss_node_class = weighted_bce(out['class_scores'], node_labels)
    loss_node_attr = weighted_bce(out['attr_scores'], node_labels)

    # GAT 정제 score(refined)도 foreground/background로 보정 → combined_scores 안정화
    # (score_from_gat=True일 때만: combined_scores가 refined로부터 나오므로 calibration 필요.
    #  False일 때는 refined가 미사용이므로 손실 추가 안 해 baseline을 깨끗이 유지)
    if out.get('score_from_gat', False):
        loss_node_class = loss_node_class + weighted_bce(out['class_scores_refined'], node_labels)
        loss_node_attr = loss_node_attr + weighted_bce(out['attr_scores_refined'], node_labels)

    # === Gaussian density target: GT 중심에 peak, 주변은 감소 ===
    # 각 GT 포인트에 질량 1을 Gaussian 분포로 할당
    # → 중심 셀이 가장 높은 score → top-K localization 정밀도 향상
    cell_diag = ((1.0 / W) ** 2 + (1.0 / H) ** 2) ** 0.5
    sigma = cell_diag * density_sigma_scale
    density_target = torch.zeros_like(node_labels)
    if gt_count > 0:
        for gi in range(gt_count):
            m = (node_assignments == gi) & (node_labels > 0.5)
            if m.sum() == 0:
                continue
            gt_pt = gt_points_norm[gi]
            dists = torch.norm(positions[m] - gt_pt.unsqueeze(0), dim=1)
            weights = torch.exp(-dists ** 2 / (2 * sigma ** 2))
            weights = weights / weights.sum().clamp(min=1e-8)
            density_target[m] = weights
    # smooth_l1은 mean이라 크기 작음 → 노드 수를 곱해 BCE 스케일에 맞춤
    loss_node_density = F.smooth_l1_loss(
        out['combined_scores'], density_target, reduction='mean'
    ) * node_labels.shape[0]
    loss_node = loss_node_class + loss_node_attr + loss_node_density

    def weighted_edge_bce(pred, target):
        pos = target.sum().clamp(min=1)
        neg = (1 - target).sum().clamp(min=1)
        w = torch.where(target > 0, neg / pos, torch.ones_like(target))
        return F.binary_cross_entropy(pred, target, weight=w)

    loss_edge_class = weighted_edge_bce(out['class_edges'], edge_labels)
    loss_edge_attr = weighted_edge_bce(out['attr_edges'], edge_labels)

    # === BA-Inter: Boundary-Aware Intersection Edge Loss ===
    # inter_edges는 graph-dedup(count_graph_guided)이 읽는 신호. 인접하지만 서로 다른
    # GT 객체에 속하는 fg-fg "경계 edge"를 hard-negative로 강조해, dedup의 BFS suppress가
    # 옆 객체로 번지지 않고(=recall 보존) 같은 객체 내부는 강결합되도록(=precision) 한다.
    # edge_labels(N1 same-object supervision)는 불변 → N1/N2/CHNM 정의·score 경로 무변경.
    if ba_inter and edge_index is not None and node_assignments is not None:
        src, tgt = edge_index
        a_s, a_t = node_assignments[src], node_assignments[tgt]
        both_fg = (a_s >= 0) & (a_t >= 0)
        boundary_neg = (both_fg & (a_s != a_t)).float()  # 결정적 경계 hard-negative
        # 기존 positive 균형 유지(+회귀 방지) 위에 경계 가중만 추가
        pos = edge_labels.sum().clamp(min=1)
        neg = (1 - edge_labels).sum().clamp(min=1)
        base_w = torch.where(edge_labels > 0, neg / pos, torch.ones_like(edge_labels))
        w = base_w + boundary_neg * inter_bnd_w
        bce = F.binary_cross_entropy(out['inter_edges'], edge_labels, reduction='none')
        loss_edge_inter = (w * bce).sum() / w.sum().clamp(min=1)
    else:
        loss_edge_inter = weighted_edge_bce(out['inter_edges'], edge_labels)
    loss_edge = 0.5 * loss_edge_class + 0.5 * loss_edge_attr + loss_edge_inter

    # === L_count: over-count(K=round(sum)>gt) 억제 ===
    # 기존 'linear'(/gt_count)는 밀집 씬에서 count 손실을 1/gt로 죽여 over-count를 방치 →
    # over-count 이미지 63.7%, 강제FP가 전체GT의 18.9%로 측정됨(precision 천장).
    #   count_norm: linear(/gt, 기존) | sqrt(/sqrt(gt), 밀집 씬 count 신호 회복) | none(/1)
    #   overcount_w>1: 과예측(sum>gt)을 과소예측보다 강하게 페널티(비대칭) → K↓
    pred_count = out['combined_scores'].sum()
    gt_count_t = torch.tensor(gt_count, dtype=torch.float32, device=dev)
    if count_norm == 'sqrt':
        denom = max(float(gt_count), 1.0) ** 0.5
    elif count_norm == 'none':
        denom = 1.0
    else:  # linear (기존)
        denom = max(float(gt_count), 1.0)
    count_err = F.smooth_l1_loss(pred_count, gt_count_t)
    # 비대칭: 과예측이면 overcount_w 배 가중
    if overcount_w != 1.0 and pred_count.item() > gt_count:
        count_err = count_err * overcount_w
    loss_count = count_weight * count_err / denom

    loss_pull, loss_push = discriminative_loss(nodes_final, node_assignments, gt_count)
    loss_disc = loss_pull + loss_push

    loss_offset = torch.tensor(0.0, device=dev)
    if gt_count > 0 and gt_points_norm.shape[0] > 0:
        fg_mask = node_labels > 0
        if fg_mask.sum() > 0:
            fg_positions = positions[fg_mask]
            fg_offsets = out['offsets'][fg_mask]
            scale = torch.tensor([0.5 / W, 0.5 / H], device=dev)
            pred_pos = fg_positions + fg_offsets * scale
            fg_assignments = node_assignments[fg_mask]
            gt_targets = gt_points_norm[fg_assignments]
            loss_offset = F.smooth_l1_loss(pred_pos, gt_targets)

    # === CHNM: Cross-Graph Hard Negative Mining ===
    # Class Graph가 "같은 class"로 식별한 토큰 중 GT foreground가 아닌 것
    # = 같은 class이지만 다른 attribute를 가진 토큰 (hard negative)
    # → attr_scores를 낮추도록 학습 (soft weighting, threshold 불필요)
    #
    # 이것이 Dual Graph만의 고유 능력:
    #   class_scores(Module 1 출력)로 hard negative를 식별하여
    #   attr_scores(Module 2 출력)에 contrastive supervision 제공
    #   → Cross-graph feedback loop
    hard_neg_weight = out['class_scores'].detach() * (1.0 - node_labels)
    # class_score가 높고(person임) + foreground가 아님(walking이 아님) → 높은 weight
    if hard_neg_weight.sum() > 1e-8:
        loss_chnm_attr = (hard_neg_weight * F.binary_cross_entropy(
            out['attr_scores'],
            torch.zeros_like(out['attr_scores']),
            reduction='none',
        )).sum() / hard_neg_weight.sum().clamp(min=1)
        loss_chnm_combined = (hard_neg_weight * F.binary_cross_entropy(
            out['combined_scores'],
            torch.zeros_like(out['combined_scores']),
            reduction='none',
        )).sum() / hard_neg_weight.sum().clamp(min=1)
        loss_chnm = loss_chnm_attr + loss_chnm_combined
    else:
        loss_chnm = torch.tensor(0.0, device=dev)

    total = (loss_node + loss_edge + loss_count
             + 0.5 * loss_disc + offset_weight * loss_offset + chnm_weight * loss_chnm)
    return total, {}


# ============================================================
# Training
# ============================================================

def train_epoch(model, step_a, decoder, optimizer, loader, annotations, epoch,
                density_sigma_scale=0.7, offset_weight=1.0,
                ba_inter=False, inter_bnd_w=3.0,
                count_norm='linear', overcount_w=1.0, count_weight=1.0,
                chnm_weight=0.5):
    model.train()  # Feature Enhancer train mode (dropout 등 활성화)
    step_a.train(); decoder.train()
    total_loss, count = 0, 0
    graph_cache = {}

    for images, caps_list, shapes, img_caps_list in loader:
        anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
        shapes_exp = [shapes[i] for i, cl in enumerate(caps_list) for _ in cl]
        images_exp = torch.stack([images[i] for i, cl in enumerate(caps_list) for _ in cl]).to(device)
        captions = [c for cl in caps_list for c in cl]

        # Feature Enhancer (gradient 활성화, fine-tuning)
        feat = run_feature_enhancer(model, images_exp, captions, training=True)

        # Step A (gradient 연결 — .detach() 제거!)
        nodes, full_emb, class_emb, attr_emb, H2, W2 = step_a(
            feat['img_memory'], feat['txt_memory'],
            feat['spatial_shapes'], feat['level_start_index'],
            feat['text_token_mask'], feat['text_class_mask'], feat['text_attr_mask'],
        )

        key = (H2, W2)
        if key not in graph_cache:
            ei, pos = build_grid_graph(H2, W2)
            graph_cache[key] = (ei.to(device), pos.to(device))
        edge_index, positions = graph_cache[key]

        bs = len(captions)
        batch_loss = torch.tensor(0.0, device=device)

        for b in range(bs):
            out = decoder(nodes[b], class_emb[b], attr_emb[b], edge_index, positions)

            gt_pts = anno_b[b]['points']
            gt_count = len(gt_pts)
            if gt_count == 0:
                gt_pts_norm = torch.zeros(0, 2, device=device)
            else:
                gt_pts_norm = torch.tensor(gt_pts, dtype=torch.float32, device=device)
                gt_pts_norm[:, 0] /= shapes_exp[b][1]
                gt_pts_norm[:, 1] /= shapes_exp[b][0]

            node_labels, edge_labels, node_assignments = generate_gt_labels(
                gt_pts_norm, positions, edge_index, H2, W2)

            loss, _ = compute_loss(out, node_labels, edge_labels, node_assignments,
                                   gt_count, out['nodes_final'],
                                   gt_pts_norm, positions, H2, W2,
                                   density_sigma_scale=density_sigma_scale,
                                   offset_weight=offset_weight,
                                   edge_index=edge_index,
                                   ba_inter=ba_inter, inter_bnd_w=inter_bnd_w,
                                   count_norm=count_norm, overcount_w=overcount_w,
                                   chnm_weight=chnm_weight,
                                   count_weight=count_weight)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / bs
        optimizer.zero_grad()
        batch_loss.backward()
        # gradient clipping — 모든 학습 파라미터에 적용
        all_params = [p for p in model.parameters() if p.requires_grad] + \
                     list(step_a.parameters()) + list(decoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        count += 1
        if count % 50 == 0:
            print(f'\r  [Train] Ep {epoch} ({count}/{len(loader.dataset)}), '
                  f'loss: {total_loss/count:.4f}', end='', flush=True)
    print()
    return total_loss / max(count, 1)


# ============================================================
# Evaluation
# ============================================================

def calc_loc_metrics(pred_centers, gt_points_norm, comp_sizes=None,
                     H=25, W=42, default_thr=0.05):
    n_pred = len(pred_centers)
    n_gt = len(gt_points_norm)

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, 1.0, 1.0, 1.0
    if n_pred == 0:
        return 0, 0, n_gt, 0.0, 0.0, 0.0
    if n_gt == 0:
        return 0, n_pred, 0, 0.0, 0.0, 0.0

    pred_np = np.array(pred_centers)
    cost = cdist(pred_np, gt_points_norm, metric='euclidean')
    pred_idx, gt_idx = linear_sum_assignment(cost)

    cell_diag = np.sqrt((1.0 / W) ** 2 + (1.0 / H) ** 2)
    if comp_sizes is not None and len(comp_sizes) > 0:
        median_size = sorted(comp_sizes)[len(comp_sizes) // 2]
        dist_thr = max(np.sqrt(median_size) * cell_diag / 2, default_thr)
    else:
        dist_thr = default_thr

    TP = sum(1 for pi, gi in zip(pred_idx, gt_idx) if cost[pi, gi] < dist_thr)
    FP = n_pred - TP
    FN = n_gt - TP

    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return TP, FP, FN, prec, rec, f1


def visualize_sample(img_id, caption, gt_pts, centers, combined_scores,
                     H2, W2, pred_count, cc, processor, epoch, results_dir='./regraph_results_v6'):
    """랜덤 1장 시각화 저장"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    img_path = f"{processor.get_image_path()}/{img_id}"
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return
    img_w, img_h = img.size

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # GT
    axes[0].imshow(img)
    for pt in gt_pts:
        axes[0].plot(pt[0], pt[1], 'g+', markersize=12, markeredgewidth=2)
    axes[0].set_title(f'GT: "{caption}" count={len(gt_pts)}', fontsize=10)
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(img)
    for ct in centers:
        axes[1].plot(ct[0] * img_w, ct[1] * img_h, 'r+', markersize=12, markeredgewidth=2)
    axes[1].set_title(f'Pred: count={pred_count} (cc={cc})', fontsize=10)
    axes[1].axis('off')

    # Heatmap
    scores_2d = combined_scores.reshape(H2, W2)
    axes[2].imshow(img)
    axes[2].imshow(scores_2d, alpha=0.5, cmap='jet',
                   extent=[0, img_w, img_h, 0], aspect='auto')
    for pt in gt_pts:
        axes[2].plot(pt[0], pt[1], 'g+', markersize=8, markeredgewidth=1)
    axes[2].set_title('Combined scores', fontsize=10)
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/ep{epoch}_vis.png', dpi=120, bbox_inches='tight')
    plt.close()


def evaluate(model, step_a, decoder, loader, annotations, split='val', epoch=0,
             processor=None, results_dir='./regraph_results_v6', nms_dist=None):
    model.eval(); step_a.eval(); decoder.eval()
    mae_sum, rmse_sum, mae_cc, rmse_cc = 0, 0, 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0
    counter = 0
    graph_cache = {}
    vis_done = False
    vis_target = np.random.randint(0, min(50, len(loader.dataset)))

    with torch.no_grad():
        for images, caps_list, shapes, img_caps_list in loader:
            anno_b = [annotations[ic] for icl in img_caps_list for ic in icl]
            shapes_exp = [shapes[i] for i, cl in enumerate(caps_list) for _ in cl]
            images_exp = torch.stack([images[i] for i, cl in enumerate(caps_list) for _ in cl]).to(device)
            captions = [c for cl in caps_list for c in cl]

            feat = run_feature_enhancer(model, images_exp, captions, training=False)
            nodes, full_emb, class_emb, attr_emb, H2, W2 = step_a(
                feat['img_memory'], feat['txt_memory'],
                feat['spatial_shapes'], feat['level_start_index'],
                feat['text_token_mask'], feat['text_class_mask'], feat['text_attr_mask'],
            )

            key = (H2, W2)
            if key not in graph_cache:
                ei, pos = build_grid_graph(H2, W2)
                graph_cache[key] = (ei.to(device), pos.to(device))
            edge_index, positions = graph_cache[key]

            for b in range(len(captions)):
                gt_pts = anno_b[b]['points']
                gt_count = len(gt_pts)
                out = decoder(nodes[b], class_emb[b], attr_emb[b], edge_index, positions)

                # Counting: sum 기반 (density map — Fix #1)
                ps = out['combined_scores'].sum().item()
                pred_count = max(0, int(round(ps)))
                mae_sum += abs(ps - gt_count)
                rmse_sum += (ps - gt_count) ** 2

                # === Sum-Guided Top-K + Adaptive NMS ===
                cc, all_centers, comp_sizes = count_sum_guided(
                    out['combined_scores'].cpu(), positions.cpu(),
                    H=H2, W=W2, offsets=out['offsets'].cpu(),
                    min_dist_cells=nms_dist)
                mae_cc += abs(cc - gt_count)
                rmse_cc += (cc - gt_count) ** 2

                centers = all_centers
                sizes = comp_sizes

                # F1
                if gt_count > 0:
                    gt_norm = np.array(gt_pts, dtype=np.float64)
                    gt_norm[:, 0] /= shapes_exp[b][1]
                    gt_norm[:, 1] /= shapes_exp[b][0]
                    tp, fp, fn, _, _, _ = calc_loc_metrics(
                        centers, gt_norm, comp_sizes=sizes, H=H2, W=W2)
                else:
                    tp, fp, fn = 0, len(centers), 0
                total_tp += tp; total_fp += fp; total_fn += fn

                # 랜덤 1장 시각화
                if not vis_done and counter == vis_target and processor is not None and gt_count > 0:
                    img_id = [ic for icl in img_caps_list for ic in icl][b][0]
                    visualize_sample(
                        img_id, captions[b], gt_pts, centers,
                        out['combined_scores'].cpu().numpy(),
                        H2, W2, pred_count, cc, processor, epoch,
                        results_dir=results_dir)
                    vis_done = True

                counter += 1

            if counter % 200 < len(captions):
                cur_p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
                cur_r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
                cur_f1 = 2*cur_p*cur_r/(cur_p+cur_r) if cur_p+cur_r > 0 else 0
                print(f'\r  [{split}] ep {epoch} ({counter}) '
                      f'MAE(sum)={mae_sum/counter:.2f}, MAE(cc)={mae_cc/counter:.2f}, '
                      f'F1={cur_f1:.3f}', end='', flush=True)

    print()
    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    rec = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    return {
        'sum_mae': mae_sum / counter, 'sum_rmse': (rmse_sum / counter) ** 0.5,
        'cc_mae': mae_cc / counter, 'cc_rmse': (rmse_cc / counter) ** 0.5,
        'precision': prec, 'recall': rec, 'f1': f1,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_graph', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='./groundingdino_swint_ogc.pth')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (노이즈 검증용 multi-seed 재실행에 사용)')
    parser.add_argument('--select_by', type=str, default='cc_mae', choices=['cc_mae', 'f1'],
                        help='best 체크포인트 선택 기준 (cc_mae=낮을수록, f1=높을수록). '
                             'F1 목표 실험이면 f1 권장')
    # === Multi-head Ablation 인자 ===
    parser.add_argument('--class_heads', type=int, default=4,
                        help='Module 1 (Class Graph) GATv2 multi-head 수')
    parser.add_argument('--attr_heads', type=int, default=4,
                        help='Module 2 (Attribute Graph) GATv2 multi-head 수')
    # === Attention 집중 강도(temperature^-1) Ablation 인자 ===
    parser.add_argument('--class_strength', type=float, default=1.0,
                        help='Class Graph attention 집중 강도 (s>1 뾰족, s<1 평평)')
    parser.add_argument('--attr_strength', type=float, default=1.0,
                        help='Attribute Graph attention 집중 강도 (s>1 뾰족, s<1 평평)')
    # === GAT 정제 feature를 scoring에 연결할지 (F1 직접 영향 경로) ===
    parser.add_argument('--score_from_gat', type=int, default=1, choices=[0, 1],
                        help='1=combined_scores를 GAT 정제 노드에서 계산(attention→F1 직접), '
                             '0=기존 raw 노드 scoring')
    # === 진짜 F1 레버 ===
    parser.add_argument('--density_sigma', type=float, default=0.7,
                        help='density target Gaussian sigma 스케일(셀 대각선 배수). '
                             '작을수록 score peak가 뾰족 → localization↑ (기본 0.7)')
    parser.add_argument('--offset_weight', type=float, default=1.0,
                        help='offset(위치 보정) loss 가중치. 클수록 sub-cell 위치 정밀↑ (기본 1.0)')
    parser.add_argument('--no_chnm', type=int, default=0, choices=[0, 1],
                        help='1이면 CHNM(L_chnm)을 제거하고 학습 (ablation: 명명된 기여 검증용)')
    parser.add_argument('--nms_dist', type=float, default=-1.0,
                        help='eval 시 top-K 선택 NMS 최소거리(셀 단위). '
                             '<=0이면 K기반 적응형(기본). 밀집 씬이면 작게(예 1.5), 희소면 크게(예 2.5)')
    # === BA-Inter: Boundary-Aware Intersection Edge Loss (graph-dedup 증폭) ===
    parser.add_argument('--ba_inter', type=int, default=0, choices=[0, 1],
                        help='1=inter_edges에 경계 hard-negative 가중 적용(graph dedup 증폭)')
    parser.add_argument('--inter_bnd_w', type=float, default=3.0,
                        help='경계 edge(fg-fg 다른 객체) hard-negative 추가 가중')
    # === over-count(K) 억제: precision 천장(강제FP 18.9%) 직접 공략 ===
    parser.add_argument('--count_norm', type=str, default='linear',
                        choices=['linear', 'sqrt', 'none'],
                        help='L_count 정규화. linear=/gt(기존,밀집씬 신호 죽음), '
                             'sqrt=/sqrt(gt)(밀집씬 count 신호 회복), none=/1')
    parser.add_argument('--overcount_w', type=float, default=1.0,
                        help='과예측(sum>gt) 비대칭 페널티 배수(>1이면 over-count 강하게 억제)')
    parser.add_argument('--count_weight', type=float, default=1.0,
                        help='L_count 전체 가중')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='결과 저장 디렉토리 (미지정 시 설정값 기반 자동 생성)')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print("  REGraph Full (v2: Feature Enhancer fine-tuning)")
    print(f"  seed={args.seed}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    processor = DataProcessor()
    annotations = processor.annotations
    train_loader = get_loader(processor, 'train', args.batch_size)
    val_loader = get_loader(processor, 'val', args.batch_size)
    test_loader = get_loader(processor, 'test', args.batch_size)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model = load_model(CONFIG_PATH, args.checkpoint)
    model = model.to(device)

    # Freeze: Backbone + BERT만 (GroundingREC과 동일)
    fe_trainable = setup_freeze(model)

    step_a = StepA(d_model=256).to(device)
    decoder = REGraphDecoder(
        d_model=256,
        class_heads=args.class_heads,
        attr_heads=args.attr_heads,
        class_strength=args.class_strength,
        attr_strength=args.attr_strength,
        score_from_gat=bool(args.score_from_gat),
    ).to(device)
    print(f"  Multi-head config: class_heads={args.class_heads}, attr_heads={args.attr_heads}")
    print(f"  Attn-strength config: class_strength={args.class_strength}, attr_strength={args.attr_strength}")
    print(f"  score_from_gat: {bool(args.score_from_gat)} "
          f"(combined_scores 출처: {'GAT 정제 노드' if args.score_from_gat else 'raw 노드'})")

    sa_params = sum(p.numel() for p in step_a.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Step A:   {sa_params:,} params")
    print(f"  Decoder:  {dec_params:,} params")
    print(f"  Total trainable: {fe_trainable + sa_params + dec_params:,} params")

    # Optimizer: Feature Enhancer(낮은 lr) + Step A/Decoder(높은 lr)
    fe_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': fe_params, 'lr': args.lr},                    # Feature Enhancer: 1e-5
        {'params': step_a.parameters(), 'lr': args.lr_graph},    # Step A: 5e-5
        {'params': decoder.parameters(), 'lr': args.lr_graph},   # Graph Decoder: 5e-5
    ], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        cs = str(args.class_strength).replace('.', 'p')
        as_ = str(args.attr_strength).replace('.', 'p')
        sg = 'gat' if args.score_from_gat else 'raw'
        results_dir = (f'./regraph_results_v6_c{args.class_heads}_a{args.attr_heads}'
                       f'_cs{cs}_as{as_}_{sg}_s{args.seed}')
    os.makedirs(results_dir, exist_ok=True)
    print(f"  Results dir: {results_dir}")

    # nms_dist: <=0 이면 적응형(None)
    nms_dist = None if args.nms_dist <= 0 else args.nms_dist
    print(f"  F1 levers: density_sigma={args.density_sigma}, "
          f"offset_weight={args.offset_weight}, "
          f"nms_dist={'adaptive' if nms_dist is None else nms_dist}")
    print(f"  BA-Inter: ba_inter={bool(args.ba_inter)}, inter_bnd_w={args.inter_bnd_w}")
    print(f"  Count: count_norm={args.count_norm}, overcount_w={args.overcount_w}, "
          f"count_weight={args.count_weight}")
    print(f"  Selection metric: {args.select_by}")
    # 선택 기준: cc_mae는 낮을수록, f1은 높을수록 좋음
    best_score = float('inf') if args.select_by == 'cc_mae' else -float('inf')
    best_val_mae = float('inf')
    best_val_f1 = 0.0
    best_state = None
    patience = 8
    no_improve = 0

    for epoch in range(args.epochs):
        loss = train_epoch(model, step_a, decoder, optimizer, train_loader, annotations, epoch,
                           density_sigma_scale=args.density_sigma,
                           offset_weight=args.offset_weight,
                           ba_inter=bool(args.ba_inter), inter_bnd_w=args.inter_bnd_w,
                           count_norm=args.count_norm, overcount_w=args.overcount_w,
                           count_weight=args.count_weight,
                           chnm_weight=(0.0 if args.no_chnm else 0.5))
        scheduler.step()

        val = evaluate(model, step_a, decoder, val_loader, annotations, 'val', epoch,
                       processor=processor, results_dir=results_dir, nms_dist=nms_dist)
        print(f'  Ep {epoch}: loss={loss:.4f} | '
              f'val MAE(sum)={val["sum_mae"]:.2f}, MAE(cc)={val["cc_mae"]:.2f}, '
              f'F1={val["f1"]:.3f} (P={val["precision"]:.3f}, R={val["recall"]:.3f})')

        # Save checkpoint (모델 전체 포함)
        state = {
            'model': model.state_dict(),
            'step_a': step_a.state_dict(),
            'decoder': decoder.state_dict(),
        }
        torch.save(state, f'{results_dir}/latest.pth')

        cur = val['cc_mae'] if args.select_by == 'cc_mae' else val['f1']
        improved = (cur < best_score) if args.select_by == 'cc_mae' else (cur > best_score)
        if improved:
            best_score = cur
            best_val_mae = val['cc_mae']
            best_val_f1 = val['f1']
            no_improve = 0
            best_state = {
                'model': copy.deepcopy(model.state_dict()),
                'step_a': copy.deepcopy(step_a.state_dict()),
                'decoder': copy.deepcopy(decoder.state_dict()),
            }
            torch.save(best_state, f'{results_dir}/best.pth')
            print(f'  -> New best! ({args.select_by}) '
                  f'MAE(cc)={best_val_mae:.2f}, F1={best_val_f1:.3f}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
                break

    # Test
    model.load_state_dict(best_state['model'])
    step_a.load_state_dict(best_state['step_a'])
    decoder.load_state_dict(best_state['decoder'])
    test = evaluate(model, step_a, decoder, test_loader, annotations, 'test', -1,
                    processor=processor, results_dir=results_dir, nms_dist=nms_dist)
    print(f"\n  Test MAE(sum)={test['sum_mae']:.2f}, MAE(cc)={test['cc_mae']:.2f}, "
          f"RMSE(cc)={test['cc_rmse']:.2f}, F1={test['f1']:.3f}")

    with open(f'{results_dir}/results.txt', 'w') as f:
        f.write(f"REGraph v6: Level1(50x84) + sharp Gaussian + CHNM + multi-head GATv2\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"Multi-head: class_heads={args.class_heads}, attr_heads={args.attr_heads}\n")
        f.write(f"Attn-strength: class_strength={args.class_strength}, attr_strength={args.attr_strength}\n")
        f.write(f"score_from_gat: {bool(args.score_from_gat)}\n")
        f.write(f"F1 levers: density_sigma={args.density_sigma}, "
                f"offset_weight={args.offset_weight}, "
                f"nms_dist={'adaptive' if nms_dist is None else nms_dist}\n")
        f.write(f"BA-Inter: ba_inter={bool(args.ba_inter)}, inter_bnd_w={args.inter_bnd_w}\n")
        f.write(f"Count: count_norm={args.count_norm}, overcount_w={args.overcount_w}, "
                f"count_weight={args.count_weight}\n")
        f.write(f"LR: FE={args.lr}, Graph={args.lr_graph}\n")
        f.write(f"Trainable: FE={fe_trainable:,}, StepA={sa_params:,}, Decoder={dec_params:,}\n")
        f.write(f"Selection metric: {args.select_by}\n")
        f.write(f"Best val MAE(cc): {best_val_mae:.2f}\n")
        f.write(f"Best val F1: {best_val_f1:.3f}\n")
        f.write(f"Test MAE(sum)={test['sum_mae']:.2f}, MAE(cc)={test['cc_mae']:.2f}, "
                f"RMSE(cc)={test['cc_rmse']:.2f}\n")
        f.write(f"Test F1={test['f1']:.3f} (P={test['precision']:.3f}, R={test['recall']:.3f})\n")

    print(f"\nDone. Results in {results_dir}/")


if __name__ == '__main__':
    main()
