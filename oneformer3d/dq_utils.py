from oneformer3d.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from mmengine.structures import InstanceData
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS
import numpy as np
from contextlib import contextmanager
from sklearn.cluster import AgglomerativeClustering
def match_for_indices(matcher, pred, insts, insts_pts, matched_dict):
    inst_gts = []
    n = 200

    for i in range(len(pred['masks'])): #4*[27,20000]
        # point-level gt_mask
        inst_gt = InstanceData()
        inst_gt.p_masks = insts_pts[i].p_masks
        inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :] #insts[i].sp_masks.shape:[223,N] 排除最后 201 行。 inst_gt.sp_masks:[22,56]
        if pred['cls_preds'][i].shape[1] == 2:
            # category agnostic 类别无关
            inst_gt.labels_3d = torch.zeros_like(insts[i].labels_3d[:-n - 1]) #inst_gt.labels_3d:[22] 创建一个与 insts[i].labels_3d 的前 labels_3d.shape[0] - 201 个元素具有相同形状和数据类型的全零张量
        else:
            inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
        if 'bboxes_3d' in insts[i].keys():
            inst_gt.bboxes_3d = insts[i].bboxes_3d[:-n - 1, :] #inst_gt.bboxes_3d:[22,7] 排除最后201行
        if insts[i].get('query_masks') is not None: #检查 insts[i] 对象是否具有名为 'query_masks' 的属性，并且该属性的值不为 None
            inst_gt.query_masks = insts[i].query_masks[:-n - 1, :] #inst_gt.query_masks:[22,37] 排除最后201行
        inst_gts.append(inst_gt)
                # match
    indices = []
    cls_preds = pred['cls_preds'] # [N_segment, 19]
    pred_masks = pred['masks'] # [N_segment, 20000]

    for i in range(len(inst_gts)): # batch_size
        pred_instances = InstanceData(
            scores=cls_preds[i],
            masks=pred_masks[i])
        gt_instances = InstanceData(
            labels=inst_gts[i].labels_3d,
            masks=inst_gts[i].p_masks) # mask_pred_mode[-1] is "P"
        if inst_gts[i].get('query_masks') is not None:
            gt_instances.query_masks = inst_gts[i].query_masks
        # All-False-gt_mask will not be matched 
        indices.append(matcher(pred_instances, gt_instances, matched_dict=matched_dict[i] if matched_dict is not None else None))
    return indices
class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = F.relu
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout1(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
    
class QueryInteractionX(nn.Module):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        dropout = kwargs.get('drop_rate', 0.0)
        self.with_att = kwargs.get('with_att', False)
        self.with_pos = kwargs.get('with_pos', False)
        self.mode = kwargs.get('mode', 'embed_only')
        if self.mode == 'with_pos':
            self.gen_tau = nn.Linear(in_channels, 8)
        self.self_attn = nn.MultiheadAttention(in_channels, 8, dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(in_channels, mid_channels, dropout)

    def forward(self, track_embed, obj_embed, pos_embed=None, pos=None):
        track_num = len(track_embed)
        query_embed = torch.cat([track_embed, obj_embed], dim=0)
        
        if self.mode == 'with_pos':
            pos_all = torch.cat([pos, pos], dim=0)  # [B, Q_max, 3]
            dist = self.calc_bbox_dists(pos_all)
            tau = self.gen_tau(query_embed)  # [B, Q, 8]

            tau = tau.permute(1, 0)  # [B, 8, Q]
            attn_mask = dist[None, :, :] * tau[..., None]  # [8, Q, Q]

        # add position embedding
        if self.with_pos and pos_embed is not None:
            pos_embed = torch.cat([pos_embed, pos_embed], dim=0)
            q = k = query_embed + pos_embed
        else:
            q = k = query_embed

        tgt = query_embed.clone()
        # attention
        if self.with_att:
            if self.mode == 'with_pos':
                tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None], attn_mask=attn_mask)[0][:, 0]
            else:
                tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        track_embed = tgt[:track_num]
        obj_embed = tgt[track_num:]

        return track_embed, obj_embed
    
    def calc_bbox_dists(self, pos1_xyz):
        """
        计算每个样本中查询点之间的距离。
        
        pos1_xyz: [N, 3]
        """
        # 计算 pairwise 欧几里得距离
        dist = torch.cdist(pos1_xyz, pos1_xyz, p=2)

        # 取负值，使得距离越近，值越大
        dist = -dist  # [B, Q_max, Q_max]

        return dist

class RefineQueryX(BaseModule):
    def __init__(self, in_channels, mid_channels, **kwargs):
        super().__init__()
        self.query_interaction = QueryInteractionX(in_channels, mid_channels, **kwargs)
        self.with_pos = kwargs.get('with_pos', False)
        self.with_att = kwargs.get('with_att', False)
        self.mode = kwargs.get('mode', 'embed_only')
        self.norm = nn.LayerNorm(in_channels)
from typing import List, Tuple

# def build_pairwise_mask(
#         merged_groups: List[List[int]],
#         compact: bool = False,
#         device: str = "cpu",
#         max_value: int = -1
#     ) -> Tuple[torch.Tensor, List[int]]:
#     """
#     Args:
#         merged_groups : 每个子列表是一簇（同一物体）的样本索引
#         compact       : True  →  仅生成出现索引的 NxN 紧凑矩阵
#                         False →  生成 (max_id+1)×(max_id+1) 矩阵
#         device        : 返回 Tensor 所在设备
#     Returns:
#         mask          : pairwise 0-1 Tensor
#         present_ids   : 列表，记录参与 mask 的原始样本索引顺序
#     """
#     # 1. 收集唯一样本并排序，保持列顺序稳定
#     present_ids = sorted({s for g in merged_groups for s in g})

#     if compact:                           # 紧凑矩阵，仅覆盖出现的样本
#         N = len(present_ids)
#         mask = torch.zeros((N, N), device=device)
#         id2row = {oid: i for i, oid in enumerate(present_ids)}
#         for group in merged_groups:
#             loc = torch.tensor([id2row[o] for o in group], device=device)
#             mask[loc.unsqueeze(1), loc] = 1.
#     else:                                 # full 矩阵
#         if max_value > 0:
#             max_id = max_value
#         else:
#             max_id = max(present_ids)
#         mask   = torch.zeros((max_id + 1, max_id + 1), device=device)
#         for group in merged_groups:
#             idx = torch.tensor(group, device=device)
#             mask[idx.unsqueeze(1), idx] = 1.

#     return mask.float(), present_ids


from typing import List, Tuple
import torch

def build_pairwise_mask(
        merged_groups: List[List[int]],
        compact: bool = False,
        device: str = "cpu",
        max_value: int = -1,
        scale: bool = False,
        low: float = 0.05,
        high: float = 0.95
    ) -> Tuple[torch.Tensor, List[int]]:
    """
    构造成对 mask，并可选地将 {0,1} 缩放到 [low, high]。

    Args:
        merged_groups : 每个子列表是一簇（同一物体）的样本索引
        compact       : True → 仅生成出现索引的 NxN 矩阵
                        False→ 生成 (max_id+1)x(max_id+1) 矩阵
        device        : 返回 Tensor 的设备
        max_value     : 在 full 模式下，mask 的大小为 (max_value+1)^2
                        若 <=0 则自动取 sample 索引中的最大值
        scale         : 是否对生成的 {0,1} mask 做线性缩放
        low, high     : 当 scale=True 时，将原来的 0 映射到 low，把 1 映射到 high

    Returns:
        mask        : pairwise mask，若 scale=True 则值在 [low, high]
        present_ids : 参与矩阵的原始样本索引列表
    """
    # 1. 收集参照的样本索引并排序
    present_ids = sorted({s for g in merged_groups for s in g})

    if compact:
        N = len(present_ids)
        mask = torch.zeros((N, N), device=device)
        id2row = {oid: i for i, oid in enumerate(present_ids)}
        for group in merged_groups:
            loc = torch.tensor([id2row[o] for o in group], device=device)
            mask[loc.unsqueeze(1), loc] = 1.
    else:
        if max_value > 0:
            max_id = max_value
        else:
            max_id = max(present_ids)
        mask = torch.zeros((max_id+1, max_id+1), device=device)
        for group in merged_groups:
            idx = torch.tensor(group, device=device)
            mask[idx.unsqueeze(1), idx] = 1.

    # 2. 可选的值缩放（label smoothing 风格）
    if scale:
        # 线性映射：0→low, 1→high，其它值（理论上是0/1以外）映射到 [low, high]
        mask = mask * (high - low) + low

    return mask.float(), present_ids



def cluster_with_threshold(sim_mat: torch.Tensor, thresh=0.75):
    """sim_mat: [N,N] 0-1 之间，相似度越大越像"""
    N = sim_mat.size(0)
    parent = list(range(N))
    def find(i):           # 并查集
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(N):
        for j in range(i+1, N):
            if sim_mat[i, j] >= thresh:
                pi, pj = find(i), find(j)
                if pi != pj: parent[pj] = pi
    # 收集簇
    clusters = {}
    for i in range(N):
        root = find(i)
        clusters.setdefault(root, []).append(i)
    return list(clusters.values())

def cluster_complete_link(sim_mat: torch.Tensor, thresh=0.75):
    """
    用完全链接（complete linkage）层次聚类保证簇内任意两点相似度 ≥ thresh。
    """
    # 1) 转成 NumPy，distance = 1 - similarity
    D = (1.0 - sim_mat.cpu().numpy())
    # 2) 完全链接，cutoff 在 distance_threshold
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=1 - thresh
    )
    labels = model.fit_predict(D)
    # 3) 收集结果
    clusters = []
    for c in np.unique(labels):
        clusters.append(list(np.where(labels == c)[0]))
    return clusters
def analyze_masked_stats(heatmap: torch.Tensor,
                         matrix: torch.Tensor,
                         mask_value: int = 1) -> dict:
    """
    统计 matrix 在 heatmap 中等于 mask_value 位置上的最大/最小/均值/方差并一行打印。

    参数:
        heatmap (Tensor): 与 matrix 同形状的整数型 Tensor，用于掩码。
        matrix (Tensor): 与 heatmap 同形状的浮点型 Tensor，用于计算统计量。
        mask_value (int): heatmap 中要选取的位置值，默认 1。

    返回:
        stats (dict): 包含 'count','max','min','mean','var' 五个统计量。
    """
    mask = (heatmap == mask_value)
    selected = matrix[mask]

    count = selected.numel()
    if count == 0:
        print(f"No elements where heatmap=={mask_value}")
        return {'count': 0, 'max': None, 'min': None, 'mean': None, 'var': None}

    max_val  = selected.max().item()
    min_val  = selected.min().item()
    mean_val = selected.mean().item()
    var_val  = selected.var(unbiased=False).item()

    # 一行打印所有统计量
    print(
        f"count={count}, "
        f"max={max_val:.6f}, min={min_val:.6f}, "
        f"mean={mean_val:.6f}, var={var_val:.6f}"
    )

    return {
        'count': count,
        'max':   max_val,
        'min':   min_val,
        'mean':  mean_val,
        'var':   var_val
    }

def find_optimal_threshold(tmp_heatmap, merge_det_heatmap, det_bboxes, threshold_range=(0, 1), step_size=0.01):
    """
    在给定的阈值范围内，搜索使得 error_mask 最小的最佳阈值。
    
    参数:
        tmp_heatmap (Tensor): 热力图，用于计算 tmp_heatmap_mask。
        merge_det_heatmap (Tensor): 目标检测热力图，用于比较。
        det_bboxes (Tensor): 边界框，大小为 (N, 4)，用于计算错误掩码的长度。
        threshold_range (tuple): 阈值范围，默认为 (0, 1)。
        step_size (float): 阈值搜索的步长，默认为 0.01。
    
    返回:
        best_threshold (float): 最佳阈值。
        best_error_mask (float): 对应的最小 error_mask 值。
    """
    best_threshold = threshold_range[0]
    best_error_mask = float('inf')
    
    # 遍历阈值范围，计算 error_mask
    for threshold in torch.arange(threshold_range[0], threshold_range[1], step_size):
        tmp_heatmap_mask = (tmp_heatmap > threshold).float()  # 计算 mask
        
        # 计算 error_mask
        error_mask = torch.abs(tmp_heatmap_mask - merge_det_heatmap).sum() / len(det_bboxes)
        
        # 更新最佳阈值和最小 error_mask
        if error_mask < best_error_mask:
            best_error_mask = error_mask
            best_threshold = threshold.item()
    
    return best_threshold, best_error_mask

def calc_mean_grouped(acc_dict, group_ranges=[(0, 50), (50, 100), (100, float('inf'))]):
    group_stats = {}
    for start, end in group_ranges:
        group_values = []
        for key, value in acc_dict.items():
            if start <= key <= end:
                group_values.extend(value)
        if group_values:
            mean_val = np.mean(group_values)
            p25 = np.percentile(group_values, 25)
            p50 = np.percentile(group_values, 50)  # Median
            p75 = np.percentile(group_values, 75)
            group_stats[(start, end)] = {'mean': mean_val, '25th_percentile': p25, '50th_percentile': p50, '75th_percentile': p75}
        else:
            group_stats[(start, end)] = {'mean': None, '25th_percentile': None, '50th_percentile': None, '75th_percentile': None}
    return group_stats

@contextmanager
def frozen_inference(module):
    """Temporarily set module to eval and disable grad, then restore original training state."""
    was_training = module.training
    module.eval()
    with torch.no_grad():
        yield
    if was_training:
        module.train()

def replace_bn_with_ln(module: nn.Module):
    """
    递归地将 module 及其子模块中的所有 BatchNorm 替换成 LayerNorm。
    对于 1d/2d/3d 的 BatchNorm，LayerNorm 都用 num_features 作为 normalized_shape。
    """
    for name, child in module.named_children():
        # 如果是 BatchNorm 层
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # 保留原来的 eps 和 affine 配置
            eps = child.eps
            affine = child.affine
            num_features = child.num_features

            # 用 LayerNorm(num_features) 来替代
            ln = nn.LayerNorm(normalized_shape=num_features,
                              eps=eps,
                              elementwise_affine=affine)
            setattr(module, name, ln)

        else:
            # 递归处理子模块
            replace_bn_with_ln(child)

class MergeFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # dim 是 appearance 和 geometry 向量的维度（比如 256）
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, appear_emb, geom_emb):
        # 都是 [N_det, N_det, dim]
        # 1) 拼接
        x = torch.cat([appear_emb, geom_emb], dim=-1)               # [N_det, N_det, 2*dim]
        # 2) 计算门控权重
        w = torch.sigmoid(self.gate(x))                             # [N_det, N_det, dim]
        # 3) 加权融合
        fused = w * appear_emb + (1.0 - w) * geom_emb               # [N_det, N_det, dim]
        return fused
def sigmoid_focal_loss(pred, tgt, alpha=0.25, gamma=2.0, reduction='mean'):
    # pred: 已经是 sigmoid 后的概率, tgt: 0/1 标签
    p_t = pred * tgt + (1 - pred) * (1 - tgt)          # p_t = p if y=1 else 1-p
    alpha_factor = alpha * tgt + (1 - alpha) * (1 - tgt)
    modulating_factor = (1 - p_t).pow(gamma)

    # 逐元素 BCE
    bce = F.binary_cross_entropy(pred, tgt, reduction='none')
    loss = alpha_factor * modulating_factor * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
    
import numpy as np
from sklearn.metrics import pair_confusion_matrix

def to_label_array(clusters, N):
    """
    Convert a list of clusters (each a list of indices) into a label array of size N.
    """
    labels = np.empty(N, dtype=int)
    for cid, members in enumerate(clusters):
        labels[members] = cid
    return labels

def evaluate_clustering_pairwise(cluster_gt, cluster_pred, N=None):
    """
    Evaluate clustering quality by pairwise confusion metrics.

    Args:
        cluster_gt (List[List[int]]): Ground-truth clusters, each a list of sample indices.
        cluster_pred (List[List[int]]): Predicted clusters, each a list of sample indices.
        N (int, optional): Total number of samples. If None, inferred as max index + 1.

    Returns:
        dict: {
            'tp': int,  # True positive pairs
            'tn': int,  # True negative pairs
            'fp': int,  # False positive pairs
            'fn': int,  # False negative pairs
            'precision': float,  # TP / (TP + FP)
            'false_positive_rate': float  # FP / (FP + TN)
        }
    """
    # Infer N if not provided
    if N is None:
        max_gt = max(idx for cl in cluster_gt for idx in cl) if cluster_gt else -1
        max_pred = max(idx for cl in cluster_pred for idx in cl) if cluster_pred else -1
        N = max(max_gt, max_pred) + 1

    labels_gt = to_label_array(cluster_gt, N)
    labels_pred = to_label_array(cluster_pred, N)

    # Compute pairwise confusion matrix: [[TN, FP], [FN, TP]]
    cm = pair_confusion_matrix(labels_gt, labels_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'precision': precision,
        'false_positive_rate': fpr
    }

# Example usage:
# cluster_gt = [[0,1], [2,3,4], [5]]
# cluster_pred = [[0], [1,2], [3,4,5]]
# results = evaluate_clustering_pairwise(cluster_gt, cluster_pred)
# print(results)
def cluster_with_per_class_threshold(iou_map, labels_mask, class_ids,
                                     thresh_per_class, base_thresh=1.0):
    """
    iou_map:    [N,N] similarity
    labels_mask:[N,N] 0/1 to mask out cross-class pairs (optional)
    class_ids:  [N]   integer class for each sample
    thresh_per_class: dict mapping {class_id: desired_thresh}
    base_thresh: single cutoff passed to cluster_complete_link
    """
    device = iou_map.device
    # build a per-sample threshold tensor
    per_samp = torch.tensor(
        [thresh_per_class[int(c)] for c in class_ids],
        device=device, dtype=iou_map.dtype
    )  # shape [N]
    alpha = base_thresh / per_samp       # [N]
    coef  = torch.sqrt(alpha[:,None] * alpha[None,:])  # [N,N], symmetric

    scaled_sim = iou_map * labels_mask.float() * coef
    return cluster_complete_link(scaled_sim, base_thresh)
