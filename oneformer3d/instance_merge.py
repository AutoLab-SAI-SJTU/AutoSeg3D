import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
import pdb
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

# This function is deprecated by OnlineMerge. No update anymore.
def ins_merge_mat(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature
    """
    weights = [0.4,0.4,0.2]
    threshold = 0.75
    frame_num = len(masks)
    points_per_mask = masks[0].shape[1]
    cur_masks, cur_labels, cur_scores, cur_queries, cur_query_feats, cur_sem_preds, cur_xyz = \
        masks[0], labels[0], scores[0], queries[0], query_feats[0], sem_preds[0], xyz_list[0]
    for i in range(1, frame_num):
        next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
            masks[i], labels[i], scores[i], queries[i], query_feats[i], sem_preds[i], xyz_list[i]
        query_feat_scores = (cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
        sem_pred_scores = F.cosine_similarity(cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
        xyz_dists = torch.cdist(cur_xyz, next_xyz, p=2)
        xyz_scores = 1 / (xyz_dists + 1e-6)
        
        mix_scores = weights[0] * query_feat_scores + weights[1] * sem_pred_scores + weights[2] * xyz_scores
        mix_scores = torch.where(mix_scores > threshold, mix_scores, torch.zeros_like(mix_scores))
        if mix_scores.shape[0] < mix_scores.shape[1]:
            mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                    - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
        # Hungarian assign
        row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
        row_ind = torch.tensor(row_ind).to(mix_scores.device)
        col_ind = torch.tensor(col_ind).to(mix_scores.device)
        mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
        row_ind = row_ind[mix_scores_mask]
        col_ind = col_ind[mix_scores_mask]

        temp = torch.zeros(cur_masks.shape[0]).bool().to(cur_masks.device)
        temp[row_ind] = True
        temp = temp.unsqueeze(1)
        temp_masks = torch.zeros((cur_masks.shape[0], points_per_mask)).bool().to(cur_masks.device)
        temp_masks[row_ind] = next_masks[col_ind]
        next_masks_ = torch.where(temp, temp_masks,
                                    torch.zeros((cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
        cur_masks = torch.cat((cur_masks, next_masks_), dim=1)
        no_merge_masks = torch.tensor(np.setdiff1d(np.arange(next_masks.shape[0]),
                col_ind.cpu())).to(next_masks.device)
        former_padding = torch.zeros((no_merge_masks.shape[0], points_per_mask * i)).bool().to(next_masks.device)
        new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
        cur_masks = torch.cat((cur_masks, new_masks), dim=0)
        
        cur_scores[row_ind] = (cur_scores[row_ind] * i + next_scores[col_ind]) / (i + 1)
        cur_scores = torch.cat((cur_scores, next_scores[no_merge_masks]), dim=0)
        cur_queries[row_ind] = (cur_queries[row_ind] * i + next_queries[col_ind]) / (i + 1)
        cur_queries = torch.cat((cur_queries, next_queries[no_merge_masks]), dim=0)
        cur_query_feats[row_ind] = (cur_query_feats[row_ind] * i + next_query_feats[col_ind]) / (i + 1)
        cur_query_feats = torch.cat((cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
        cur_sem_preds[row_ind] = (cur_sem_preds[row_ind] * i + next_sem_preds[col_ind]) / (i + 1)
        cur_sem_preds = torch.cat((cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
        cur_xyz[row_ind] = (cur_xyz[row_ind] * i + next_xyz[col_ind]) / (i + 1)
        cur_xyz = torch.cat((cur_xyz, next_xyz[no_merge_masks]), dim=0)
    
    if len(cur_scores) > inscat_topk_insts:
        _, kept_ins = cur_scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    cur_masks, cur_scores = cur_masks[kept_ins], cur_scores[kept_ins]
    cur_labels = torch.zeros_like(cur_scores).long()
    return cur_masks, cur_labels, cur_scores
       
def ins_cat(masks, labels, scores, inscat_topk_insts):
    """Directly stack multiview instances without mask merging"""
    frame_num = len(masks)
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    if len(scores) > inscat_topk_insts:
        _, kept_ins = scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    labels, scores = labels[kept_ins], scores[kept_ins]
    ins_num = [mask.shape[0] for mask in masks]
    frame_indicator = torch.cat([torch.ones(num)*i for i, num in enumerate(ins_num)])
    frame_indicator = frame_indicator.to(scores.device)[kept_ins]
    masks = torch.cat(masks, dim=0)[kept_ins]
    new_mask = masks.new_zeros(size=(masks.shape[0], frame_num*masks.shape[1]))
    for ids in range(len(ins_num)):
        this_frame = (frame_indicator == ids)
        new_mask[this_frame, ids*masks.shape[1]:(ids+1)*masks.shape[1]] = masks[this_frame]
    return new_mask, labels, scores

def ins_merge(points, masks, labels, scores, queries, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature"""
    frame_num = len(points)
    pts_per_frame = points[0].shape[0]
    cur_instances = [InstanceQuery(mask, label, score, query) for mask, label, score, query \
            in zip(masks[0], labels[0], scores[0], queries[0])]
    cur_points = points[0]
    for i in range(1, frame_num):
        for mask, label, score, query in zip(masks[i], labels[i], scores[i], queries[i]):
            is_merge = False
            for InsQ in cur_instances:
                # merged ins
                if InsQ.compare(cur_points, points[i], mask, label, score, query):
                    InsQ.merge(mask, label, score, query, i)
                    is_merge = True
                    break
            # new ins
            if not is_merge:
                mask = torch.cat([mask.new_zeros(pts_per_frame*i).bool(), mask])
                cur_instances.append(InstanceQuery(mask, label, score, query))
        cur_points = torch.cat([cur_points, points[i]])
        # not merged ins
        for InsQ in cur_instances:
            if len(InsQ.mask) < cur_points.shape[0]:
                InsQ.pad(pts_per_frame)
    merged_mask = torch.stack([InsQ.mask for InsQ in cur_instances], dim=0)
    merged_labels = torch.tensor([InsQ.label for InsQ in cur_instances]).to(merged_mask.device)
    merged_scores = torch.tensor([InsQ.score for InsQ in cur_instances]).to(merged_mask.device)
    if len(merged_scores) > inscat_topk_insts:
        _, kept_ins = merged_scores.topk(inscat_topk_insts)
    else:
        kept_ins = ...
    merged_mask, merged_labels, merged_scores = \
        merged_mask[kept_ins], merged_labels[kept_ins], merged_scores[kept_ins]
    return merged_mask, merged_labels, merged_scores

class GTMerge():
    def __init__(self):
        self.cur_queries = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_queries = None
        self.merge_counts = None
    
    # weighted sum according to count of merge, rather than frame
    def merge(self, queries, cls_preds, query_ins_masks):
        batch_size = len(queries)
        ins_query_list = []
        merge_count_list = []
        # Intra-frame merge: choose one with max score
        for i in range(batch_size):
            n_instances = len(query_ins_masks[i])
            if n_instances == 0:
                return None
            ins_query = []
            merge_count = []
            for j in range(n_instances):
                temp_idx = query_ins_masks[i][j]
                # ins_query.append(queries[i][temp_idx].mean(0) if
                #      len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                # merge_count.append(len(temp_idx))
                fg_scores = cls_preds[i][temp_idx].softmax(-1)[:,:-1].sum(-1)
                ins_query.append(queries[i][temp_idx][fg_scores.argmax()] if
                     len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                merge_count.append(1 if len(temp_idx) != 0 else 0)
            ins_query_list.append(torch.stack(ins_query, dim=0))
            merge_count_list.append(torch.tensor(merge_count, device=temp_idx.device).unsqueeze(-1))
        if self.cur_queries is None:
            self.cur_queries = ins_query_list
            self.merge_counts = merge_count_list
        else:
            # Inter-frame merge: mean across frame
            for i in range(batch_size):
                # self.cur_queries[i] = (self.cur_queries[i] * self.fi + ins_query_list[i]) / (self.fi + 1)
                self.cur_queries[i] = (self.cur_queries[i] * self.merge_counts[i] + ins_query_list[i]
                     * merge_count_list[i]) / (self.merge_counts[i] + merge_count_list[i] + 1e-6)
                self.merge_counts[i] = self.merge_counts[i] + merge_count_list[i]
        output_queries = []
        for i in range(batch_size):
            output_queries.append(self.cur_queries[i][self.cur_queries[i].sum(-1) != 0])
        self.fi += 1
        return output_queries


class OnlineMerge():
    def __init__(self, inscat_topk_insts, use_bbox=False, merge_type="count"):
        assert merge_type in ['count', 'frame']
        self.merge_type = merge_type
        self.inscat_topk_insts = inscat_topk_insts
        self.use_bbox = use_bbox
        if self.use_bbox:
            self.iou_calculator = AxisAlignedBboxOverlaps3D()
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.merge_counts = None
    
    def merge(self, masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, det2track_mat=None):
        points_per_mask = masks.shape[1]
        # masks, labels, scores, queries, query_feats, sem_preds, xyz_list = \
        #     self.intra_frame_merge(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, q)
        if self.cur_masks is None:
            self.cur_masks = masks
            self.cur_labels = labels
            self.cur_scores = scores
            # self.cur_queries = queries
            self.cur_query_feats = query_feats
            self.cur_sem_preds = sem_preds
            self.cur_xyz = self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            self.merge_counts = torch.zeros_like(scores).long()
        else:
            self.fi += 1
            next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
                masks, labels, scores, queries, query_feats, sem_preds, \
                self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            query_feat_scores = (self.cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
            # sem_pred_scores = F.cosine_similarity(self.cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
            if self.use_bbox:
                xyz_scores = self.iou_calculator(self.cur_xyz, next_xyz, is_aligned=False)
            else:
                xyz_dists = torch.cdist(self.cur_xyz, next_xyz, p=2)
                xyz_scores = 1 / (xyz_dists + 1e-6)
                        
            mix_scores = query_feat_scores * xyz_scores
            inst_label_scores = torch.where(self.cur_labels.unsqueeze(1) == next_labels.unsqueeze(0), torch.ones((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device), torch.zeros((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device))
            
            mix_scores = torch.where(mix_scores > 0, mix_scores, torch.zeros_like(mix_scores))
            mix_scores = mix_scores * inst_label_scores
            if mix_scores.shape[0] < mix_scores.shape[1]:
                mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                     - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
            # Hungarian assign
            row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
            row_ind = torch.tensor(row_ind).to(mix_scores.device)
            col_ind = torch.tensor(col_ind).to(mix_scores.device)
            mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
            row_ind = row_ind[mix_scores_mask]
            col_ind = col_ind[mix_scores_mask]

            temp = torch.zeros(self.cur_masks.shape[0]).bool().to(self.cur_masks.device) # [N_obj_previous]
            temp[row_ind] = True
            temp = temp.unsqueeze(1) # [N_obj_previous, 1]
            temp_masks = torch.zeros((self.cur_masks.shape[0], points_per_mask)).bool().to(self.cur_masks.device)
            temp_masks[row_ind] = next_masks[col_ind]
            # 更新已经存在的物体
            next_masks_ = torch.where(temp, temp_masks, # 使用 temp 作为条件，决定在哪些行使用更新后的掩码
                                     torch.zeros((self.cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
            self.cur_masks = torch.cat((self.cur_masks, next_masks_), dim=1)
            # 匹配新物体
            no_merge_masks = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
            no_merge_masks[col_ind] = False
            former_padding = torch.zeros((no_merge_masks.nonzero().shape[0], points_per_mask * self.fi)).bool().to(next_masks.device)
            new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
            self.cur_masks = torch.cat((self.cur_masks, new_masks), dim=0)
            # 更新合并次数 merge_counts 匹配到的老实例（row_ind）合并次数+1。
            self.merge_counts[row_ind] += 1
            if len(no_merge_masks) > 0: # 对于新加入的实例（未匹配的那些），其合并次数初始化为 0，并拼到 merge_counts 的尾部
                self.merge_counts = torch.cat([self.merge_counts,
                     torch.zeros(no_merge_masks.shape[0]).long().to(self.merge_counts.device)], dim=0)
            
            if self.merge_type == 'count':
                count = self.merge_counts[row_ind]
            else: count = self.fi
            # 被合并的老实例，用“加权平均”更新分数：
            self.cur_scores[row_ind] = (self.cur_scores[row_ind] * count + next_scores[col_ind]) / (count + 1)
            self.cur_scores = torch.cat((self.cur_scores, next_scores[no_merge_masks]), dim=0)
            if self.merge_type == 'count':
                count = count.unsqueeze(-1)
            self.cur_labels = torch.cat((self.cur_labels, next_labels[no_merge_masks]), dim=0) # 更新标签
            # self.cur_queries[row_ind] = (self.cur_queries[row_ind] * count + next_queries[col_ind]) / (count + 1)
            # self.cur_queries = torch.cat((self.cur_queries, next_queries[no_merge_masks]), dim=0)
            self.cur_query_feats[row_ind] = (self.cur_query_feats[row_ind] * count + next_query_feats[col_ind]) / (count + 1)
            self.cur_query_feats = torch.cat((self.cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
            # self.cur_sem_preds[row_ind] = (self.cur_sem_preds[row_ind] * count + next_sem_preds[col_ind]) / (count + 1)
            # self.cur_sem_preds = torch.cat((self.cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
            self.cur_xyz[row_ind] = (self.cur_xyz[row_ind] * count + next_xyz[col_ind]) / (count + 1)
            self.cur_xyz = torch.cat((self.cur_xyz, next_xyz[no_merge_masks]), dim=0)
            
        if len(self.cur_scores) > self.inscat_topk_insts:
            _, kept_ins = self.cur_scores.topk(self.inscat_topk_insts)
        else:
            kept_ins = ...
        cur_masks, cur_scores = self.cur_masks[kept_ins], self.cur_scores[kept_ins]
        cur_labels = self.cur_labels[kept_ins]
        # cur_queries = self.cur_queries[kept_ins]
        cur_bboxes = self.cur_xyz[kept_ins] if self.use_bbox else None
        # cur_labels = torch.zeros_like(self.cur_scores).long()
        return cur_masks, cur_labels, cur_scores, cur_bboxes # , cur_queries
    
    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


class DQ_Track_OnlineMerge():
    def __init__(self, inscat_topk_insts, use_bbox=False, merge_type="count", use_buffer=False):
        self.merge_type = merge_type
        self.inscat_topk_insts = inscat_topk_insts
        self.use_bbox = use_bbox
        if self.use_bbox:
            self.iou_calculator = AxisAlignedBboxOverlaps3D()
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.fi = 0
        self.merge_counts = None
        # Optional revival buffer (disabled by default)
        self.use_buffer = True
        # Buffer stores only global ids and ages; snapshots optional
        self.buffer_ids = []   # list[int]
        self.buffer_age = []   # list[int]
        # Snapshots used by external det2buffer_mat computation (Scheme A)
        self.buffer_queries = []  # list[Tensor[C]]
        self.buffer_bboxes = []   # list[Tensor[6]]
        self.buffer_cats = []     # list[Tensor[]]
        # Additional snapshots for strict revival
        self.buffer_scores = []       # list[Tensor[1]] or scalar tensor
        self.buffer_track_age = []    # list[Tensor[1]] or scalar tensor
        self.buffer_long_track = []   # list[Tensor[1]] bool
        self.buffer_active = []       # list[Tensor[1]] bool
    
    def clean(self):
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.merge_counts = None
        # clear buffer
        self.buffer_ids = []
        self.buffer_age = []
        self.buffer_queries = []
        self.buffer_bboxes = []
        self.buffer_cats = []
        self.buffer_scores = []
        self.buffer_track_age = []
        self.buffer_long_track = []
        self.buffer_active = []

    def get_buffer_snapshots(self, device=None):
        """Expose buffer snapshots for external det2buffer_mat computation.
        Returns (buf_ids, buf_queries, buf_bboxes, buf_cats) or (None, None, None, None) if empty.
        """
        if len(self.buffer_ids) == 0:
            return None, None, None, None
        buf_ids = torch.as_tensor(self.buffer_ids, dtype=torch.long,
                                  device=device if device is not None else None)
        if len(self.buffer_queries) > 0:
            buf_q = torch.stack(self.buffer_queries, dim=0)
            buf_q = buf_q.to(device) if device is not None else buf_q
        else:
            buf_q = None
        if len(self.buffer_bboxes) > 0:
            buf_b = torch.stack(self.buffer_bboxes, dim=0)
            buf_b = buf_b.to(device) if device is not None else buf_b
        else:
            buf_b = None
        if len(self.buffer_cats) > 0:
            buf_c = torch.stack(self.buffer_cats, dim=0)
            buf_c = buf_c.to(device) if device is not None else buf_c
        else:
            buf_c = None
        return buf_ids, buf_q, buf_b, buf_c
  
    def merge(self, masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, det2track_mat, det2buffer_mat, track_instances, track_embedding_for_update, current_max_track_id, det_category, ema_decay_rate=0.5, asso_thres=0.1, miss_thres=50):
        points_per_mask = masks.shape[1]
        # masks, labels, scores, queries, query_feats, sem_preds, xyz_list = \
        #     self.intra_frame_merge(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, q)
        if self.cur_masks is None:
            self.cur_masks = masks # [N_obj, 20000]
            self.cur_labels = labels # [N_obj]
            self.cur_scores = scores # [N_obj]
            # self.cur_queries = queries
            self.cur_query_feats = query_feats # [N_obj, 256]
            # self.cur_sem_preds = sem_preds # [N_obj, 2]
            self.cur_xyz = self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list # [N_obj, 6]
            self.merge_counts = torch.zeros_like(scores).long() # [N_obj]
            # batch_idx = 0
            # valid_track = track_instances.valid_track[batch_idx]
            # track_instances.queries[valid_track] = track_embedding_for_update
        else:
            self.fi += 1
            batch_idx = 0
            next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
                masks, labels, scores, queries, query_feats, sem_preds, \
                self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            # Aging & pruning the buffer (drop if age > 2*miss_thres)
            if self.use_buffer and len(self.buffer_ids) > 0:
                # increase ages
                self.buffer_age = [a + 1 for a in self.buffer_age]
                # keep those with age <= 2*miss_thres
                keep_mask = [a <= (miss_thres * 2) for a in self.buffer_age]
                if not all(keep_mask):
                    self.buffer_ids = [gid for gid, k in zip(self.buffer_ids, keep_mask) if k]
                    self.buffer_age = [a for a, k in zip(self.buffer_age, keep_mask) if k]
                    self.buffer_queries = [q for q, k in zip(self.buffer_queries, keep_mask) if k]
                    self.buffer_bboxes = [bb for bb, k in zip(self.buffer_bboxes, keep_mask) if k]
                    self.buffer_cats = [c for c, k in zip(self.buffer_cats, keep_mask) if k]
                    self.buffer_scores = [s for s, k in zip(self.buffer_scores, keep_mask) if k]
                    self.buffer_track_age = [ta for ta, k in zip(self.buffer_track_age, keep_mask) if k]
                    self.buffer_long_track = [lt for lt, k in zip(self.buffer_long_track, keep_mask) if k]
                    self.buffer_active = [ac for ac, k in zip(self.buffer_active, keep_mask) if k]
            valid_track_idx = track_instances.valid_track[batch_idx].nonzero(as_tuple=True)[0]
            track_category = track_instances.category[batch_idx][valid_track_idx]
            invalid_mask = torch.ones((valid_track_idx.shape[0], next_labels.shape[0])).to(self.cur_labels.device)
            for i in range(valid_track_idx.shape[0]):
                for j in range(next_labels.shape[0]):
                    if track_category[i] == det_category[j]:
                        invalid_mask[i][j] = 2
            # Original
            # query_feat_scores = (self.cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
            query_feat_scores = det2track_mat.T
            # mask掉小于0.1的
            # query_feat_scores = torch.where(query_feat_scores > asso_thres, query_feat_scores, torch.zeros_like(query_feat_scores))
            # sem_pred_scores = F.cosine_similarity(self.cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
            if self.use_bbox:
                # xyz_scores = self.iou_calculator(self.cur_xyz, next_xyz, is_aligned=False)
                cur_xyz = track_instances.bboxes[batch_idx][valid_track_idx]
                xyz_scores = self.iou_calculator(bbox_pred_to_bbox(cur_xyz), next_xyz, is_aligned=False)
            else:
                xyz_dists = torch.cdist(self.cur_xyz, next_xyz, p=2)
                xyz_scores = 1 / (xyz_dists + 1e-6)
            # xyz_scores_mask = torch.where(xyz_scores > 0.1, 1, 0)
            mix_scores = query_feat_scores  * xyz_scores # * invalid_mask
            # mix_scores = xyz_scores
            # query_feat_scores = torch.where(query_feat_scores > 0.05, query_feat_scores, torch.zeros_like(query_feat_scores))
            # mix_scores = query_feat_scores * xyz_scores
            # inst_label_scores = torch.where(self.cur_labels.unsqueeze(1) == next_labels.unsqueeze(0), torch.ones((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device), torch.zeros((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device))
            
            mix_scores = torch.where(mix_scores > 0.00, mix_scores, torch.zeros_like(mix_scores))
            # mix_scores = mix_scores * inst_label_scores
            if mix_scores.shape[0] < mix_scores.shape[1]:
                mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                     - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
            # Hungarian assign
            row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
            row_ind = torch.tensor(row_ind).to(mix_scores.device)
            col_ind = torch.tensor(col_ind).to(mix_scores.device)
            mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
            row_ind = row_ind[mix_scores_mask]
            col_ind = col_ind[mix_scores_mask]
            match_dets = col_ind
            match_tracks = valid_track_idx[row_ind]
            # DQ_Track
            # det_category = next_labels
            # track_category = track_instances.obj_labels[batch_idx][valid_track_idx]
            # invalid = (det2track_mat < asso_thres)
            # assign_mat = det2track_mat - invalid * 10
            # match_dets, match_tracks = linear_sum_assignment(-assign_mat.cpu().numpy())
            # DQ_Track assign
            # update info for matched tracklets
            if len(match_dets) > 0:
                track_instances.valid_track[batch_idx][match_tracks] = True
                track_instances.long_track[batch_idx][match_tracks] = True
                track_instances.active[batch_idx][match_tracks] = True
                track_instances.track_age[batch_idx][match_tracks] += 1
                before_age = track_instances.track_age[batch_idx][match_tracks].unsqueeze(-1)
                track_instances.disappear_time[batch_idx][match_tracks] = 0
                
                cur_bboxes = bboxes[match_dets]
                cur_bboxes[:, :3] += xyz_list[match_dets][:, :3]
                if self.merge_type == 'count':
                    track_instances.bboxes[batch_idx][match_tracks] = (track_instances.bboxes[batch_idx][match_tracks] * before_age + cur_bboxes) / (before_age + 1)
                    track_instances.queries[batch_idx][match_tracks] = (track_instances.queries[batch_idx][match_tracks] * before_age + track_embedding_for_update[match_dets]) / (before_age + 1)
                else:
                    track_instances.bboxes[batch_idx][match_tracks] = track_instances.bboxes[batch_idx][match_tracks] * ema_decay_rate + (1 - ema_decay_rate) * cur_bboxes
                    track_instances.queries[batch_idx][match_tracks] = track_instances.queries[batch_idx][match_tracks] * ema_decay_rate + (1 - ema_decay_rate) * track_embedding_for_update[match_dets]
                track_instances.scores[batch_idx][match_tracks] = next_scores[match_dets]
            # 更新已经存在的物体
            global_match_tracks = track_instances.global_track_id[batch_idx][match_tracks]
            temp = torch.zeros(self.cur_masks.shape[0]).bool().to(self.cur_masks.device)
            temp[global_match_tracks] = True
            temp = temp.unsqueeze(1) # [N_track, 1]
            temp_masks = torch.zeros((self.cur_masks.shape[0], points_per_mask)).bool().to(self.cur_masks.device)
            temp_masks[global_match_tracks] = next_masks[match_dets]
            next_masks_ = torch.where(temp, temp_masks, # 使用 temp 作为条件，决定在哪些行使用更新后的掩码
                                     torch.zeros((self.cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
            self.cur_masks = torch.cat((self.cur_masks, next_masks_), dim=1) # 给每个物体增加新的点
            # 匹配新物体
            no_merge_masks = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
            no_merge_masks[match_dets] = False
            former_padding = torch.zeros((no_merge_masks.nonzero().shape[0], points_per_mask * self.fi)).bool().to(next_masks.device)
            new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
            self.cur_masks = torch.cat((self.cur_masks, new_masks), dim=0)
            self.merge_counts[global_match_tracks] += 1
            if len(no_merge_masks) > 0:
                self.merge_counts = torch.cat([self.merge_counts,
                        torch.zeros(no_merge_masks.shape[0]).long().to(self.merge_counts.device)], dim=0)
            # ?是不是根本就没约束这个self.cur_masks的长度
            # # 更新其余全局参数 需要进行映射
            count = self.merge_counts[global_match_tracks]
            self.cur_scores[global_match_tracks] = (self.cur_scores[global_match_tracks] * count + next_scores[match_dets]) / (count + 1)
            self.cur_scores = torch.cat((self.cur_scores, next_scores[no_merge_masks]), dim=0)
            count = count.unsqueeze(-1)
            self.cur_labels = torch.cat((self.cur_labels, next_labels[no_merge_masks]), dim=0) # 更新标签
            self.cur_query_feats[global_match_tracks] = (self.cur_query_feats[global_match_tracks] * count + next_query_feats[match_dets]) / (count + 1)
            self.cur_query_feats = torch.cat((self.cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
            self.cur_xyz[global_match_tracks] = (self.cur_xyz[global_match_tracks] * count + next_xyz[match_dets]) / (count + 1)
            self.cur_xyz = torch.cat((self.cur_xyz, next_xyz[no_merge_masks]), dim=0)
            
            # 进行更新
            if len(match_dets) > 0:
                unmatched_dets = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
                unmatched_dets[match_dets] = False
                unmatched_dets = unmatched_dets.nonzero(as_tuple=True)[0]
                unmatched_tracks = torch.zeros(self.cur_masks.shape[0]).bool().to(self.cur_masks.device)
                unmatched_tracks[valid_track_idx] = True
                unmatched_tracks[match_tracks] = False
                unmatched_tracks = unmatched_tracks.nonzero(as_tuple=True)[0]
            else:
                unmatched_dets = torch.arange(0, next_masks.shape[0]).to(next_masks.device)
                unmatched_tracks = torch.arange(0, self.cur_masks.shape[0]).to(self.cur_masks.device)

            # Try to revive from buffer by matching unmatched detections to buffered tracks
            if self.use_buffer and (unmatched_dets.numel() > 0) and (len(self.buffer_ids) > 0):
                buf_ids = torch.as_tensor(self.buffer_ids, device=self.cur_labels.device, dtype=torch.long)
                # Prefer externally computed det2buffer_mat when available
                if det2buffer_mat is not None:
                    # det2buffer_mat shape: [N_det, N_buf] in buffer_ids order
                    query_feat_scores = det2buffer_mat[unmatched_dets].T  # [B, D]
                else:
                    # fallback to pure dot-product similarity
                    # buf_q = self.cur_query_feats[buf_ids]  # [B, C]
                    # det_q = next_query_feats[unmatched_dets]  # [D, C]
                    # query_feat_scores = buf_q @ det_q.t()  # [B, D]
                    raise NotImplementedError("det2buffer_mat must be provided for buffer revival.")
                # spatial similarity (align with main path)
                if self.use_bbox:
                    # Use buffered snapshot (center+size) for IoU, aligned with main path
                    if len(self.buffer_bboxes) == len(self.buffer_ids) and len(self.buffer_bboxes) > 0:
                        buf_param = torch.stack(self.buffer_bboxes, dim=0).to(next_xyz.device, dtype=next_xyz.dtype)
                        buf_boxes = bbox_pred_to_bbox(buf_param)  # convert to corners
                    else:
                        # Fallback to global cache (already corners)
                        raise NotImplementedError("Buffer bbox snapshots must be provided for buffer revival with bbox IoU.")
                    det_boxes = next_xyz[unmatched_dets]  # axis-aligned [D, 6]
                    xyz_scores = self.iou_calculator(buf_boxes, det_boxes, is_aligned=False)
                else:
                    buf_centers = self.cur_xyz[buf_ids]
                    det_centers = next_xyz[unmatched_dets]
                    d = torch.cdist(buf_centers, det_centers, p=2)
                    xyz_scores = 1.0 / (d + 1e-6)

                mix_scores = query_feat_scores * xyz_scores
                mix_scores = torch.where(mix_scores > 0.00, mix_scores, torch.zeros_like(mix_scores))
                if mix_scores.size(0) < mix_scores.size(1):
                    pad = mix_scores.new_zeros(mix_scores.size(1) - mix_scores.size(0), mix_scores.size(1))
                    mix_scores = torch.cat([mix_scores, pad], dim=0)

                r_row, r_col = linear_sum_assignment((-mix_scores).cpu())
                r_row = torch.as_tensor(r_row, device=mix_scores.device)
                r_col = torch.as_tensor(r_col, device=mix_scores.device)
                valid = mix_scores[r_row, r_col] > 0
                r_row = r_row[valid]
                r_col = r_col[valid]

                if r_row.numel() > 0:
                    # allocate empty slots
                    empty_mask = ~track_instances.valid_track[batch_idx]
                    need = r_row.numel()
                    free = int(empty_mask.sum().item())
                    assert need <= free, f"Need {need} slots to revive but only {free} free"
                    empty_idx = empty_mask.nonzero(as_tuple=True)[0][:need]

                    # revive mapping
                    revived_global_ids = buf_ids[r_row]
                    revived_det_ids = unmatched_dets[r_col]

                    # gather buffered snapshots for strict revival
                    device = self.cur_masks.device
                    buf_qsnap = torch.stack(self.buffer_queries, dim=0)[r_row].to(device)
                    buf_bbox = torch.stack(self.buffer_bboxes, dim=0)[r_row].to(device)
                    buf_score = (torch.stack(self.buffer_scores, dim=0)[r_row].to(device)
                                 if len(self.buffer_scores) == len(self.buffer_ids) else None)
                    buf_age = (torch.stack(self.buffer_track_age, dim=0)[r_row].to(device)
                               if len(self.buffer_track_age) == len(self.buffer_ids) else None)
                    buf_cat = torch.stack(self.buffer_cats, dim=0)[r_row].to(device)

                    # current detection (absolute center + size)
                    cur_bboxes = bboxes[revived_det_ids]
                    cur_bboxes[:, :3] += xyz_list[revived_det_ids][:, :3]
                    cur_queries = track_embedding_for_update[revived_det_ids]

                    # blend according to merge_type (keep consistency with matched update)
                    if self.merge_type == 'count' and buf_age is not None:
                        w = (buf_age.unsqueeze(-1).clamp(min=0).to(cur_bboxes.dtype))
                        new_bboxes = (buf_bbox * w + cur_bboxes) / (w + 1)
                        new_queries = (buf_qsnap * w + cur_queries) / (w + 1)
                        new_age = (buf_age + 1)
                    else:
                        # EMA or missing age -> use EMA as fallback
                        new_bboxes = buf_bbox * ema_decay_rate + (1 - ema_decay_rate) * cur_bboxes
                        new_queries = buf_qsnap * ema_decay_rate + (1 - ema_decay_rate) * cur_queries
                        new_age = (buf_age + 1) if buf_age is not None else torch.ones_like(revived_det_ids, dtype=torch.long, device=device)

                    # write track slots
                    track_instances.valid_track[batch_idx][empty_idx] = True
                    track_instances.long_track[batch_idx][empty_idx] = True
                    track_instances.active[batch_idx][empty_idx] = True
                    track_instances.track_age[batch_idx][empty_idx] = new_age
                    track_instances.disappear_time[batch_idx][empty_idx] = 0
                    track_instances.queries[batch_idx][empty_idx] = new_queries
                    track_instances.bboxes[batch_idx][empty_idx] = new_bboxes
                    # scores: keep consistent with matched path -> use current detection score
                    track_instances.scores[batch_idx][empty_idx] = next_scores[revived_det_ids]
                    # restore original category
                    track_instances.category[batch_idx][empty_idx] = buf_cat
                    track_instances.global_track_id[batch_idx][empty_idx] = revived_global_ids

                    # update global caches: write revived masks into the last frame block
                    self.cur_masks[revived_global_ids, -points_per_mask:] = next_masks[revived_det_ids]

                    self.merge_counts[revived_global_ids] += 1
                    cnt = self.merge_counts[revived_global_ids]
                    self.cur_scores[revived_global_ids] = (self.cur_scores[revived_global_ids] * cnt + next_scores[revived_det_ids]) / (cnt + 1)
                    cnt = cnt.unsqueeze(-1)
                    self.cur_query_feats[revived_global_ids] = (self.cur_query_feats[revived_global_ids] * cnt + next_query_feats[revived_det_ids]) / (cnt + 1)
                    self.cur_xyz[revived_global_ids] = (self.cur_xyz[revived_global_ids] * cnt + next_xyz[revived_det_ids]) / (cnt + 1)

                    # remove matched dets from unmatched_dets
                    keep_mask = torch.ones(unmatched_dets.size(0), dtype=torch.bool, device=unmatched_dets.device)
                    keep_mask[r_col] = False
                    unmatched_dets = unmatched_dets[keep_mask]

                    # remove revived ids from buffer
                    keep = torch.ones(len(self.buffer_ids), dtype=torch.bool)
                    keep[r_row.cpu()] = False
                    keep_list = keep.tolist()
                    self.buffer_ids = [gid for gid, k in zip(self.buffer_ids, keep_list) if k]
                    self.buffer_age = [a for a, k in zip(self.buffer_age, keep_list) if k]
                    self.buffer_queries = [q for q, k in zip(self.buffer_queries, keep_list) if k]
                    self.buffer_bboxes = [bb for bb, k in zip(self.buffer_bboxes, keep_list) if k]
                    self.buffer_cats = [c for c, k in zip(self.buffer_cats, keep_list) if k]
                    self.buffer_scores = [s for s, k in zip(self.buffer_scores, keep_list) if k]
                    self.buffer_track_age = [ta for ta, k in zip(self.buffer_track_age, keep_list) if k]
                    self.buffer_long_track = [lt for lt, k in zip(self.buffer_long_track, keep_list) if k]
                    self.buffer_active = [ac for ac, k in zip(self.buffer_active, keep_list) if k]
            if len(unmatched_dets) > 0: # 增加新的track
                # empty_track = (track_instances.valid_track[batch_idx] == False)
                # assert empty_track.sum() > 0
                # tmp_mask = empty_track[empty_track]
                # tmp_mask[len(unmatched_dets):] = False
                # empty_track[empty_track.clone()] = tmp_mask
                empty_track = ~track_instances.valid_track[batch_idx]
                needed = unmatched_dets.numel()
                free = empty_track.sum().item()
                assert needed <= free, f"Need {needed} new slots but only {free} available"

                empty_track = empty_track.nonzero(as_tuple=True)[0][:needed]
                assert track_instances.valid_track[batch_idx][empty_track].sum() == 0, "track_instances.valid_track[batch_idx][empty_track].sum() != 0"
                track_instances.valid_track[batch_idx][empty_track] = True
                track_instances.long_track[batch_idx][empty_track] = False
                track_instances.active[batch_idx][empty_track] = True
                track_instances.track_age[batch_idx][empty_track] = 0
                track_instances.disappear_time[batch_idx][empty_track] = 0
                track_instances.queries[batch_idx][empty_track] = track_embedding_for_update[unmatched_dets]
                cur_bboxes = bboxes[unmatched_dets]
                cur_bboxes[:, :3] += xyz_list[unmatched_dets][:, :3]
                track_instances.bboxes[batch_idx][empty_track] = cur_bboxes
                track_instances.scores[batch_idx][empty_track] = next_scores[unmatched_dets]
                # track_instances.obj_labels[batch_idx][empty_track] = det_category[unmatched_dets]
                track_instances.global_track_id[batch_idx][empty_track] = current_max_track_id + torch.arange(0, len(unmatched_dets)).to(self.cur_masks.device)
                track_instances.category[batch_idx][empty_track] = det_category[unmatched_dets]
                current_max_track_id += len(unmatched_dets)
            if len(unmatched_tracks) > 0:
                track_instances.disappear_time[batch_idx][unmatched_tracks] += 1
            # assert track_instances.disappear_time[batch_idx].max() <= miss_thres + 1, "track_instances.disappear_time[batch_idx].max() > miss_thres + 1"
            track_mask = track_instances.disappear_time[batch_idx] > miss_thres
            dead_track = track_instances.valid_track[batch_idx] & track_mask
            if dead_track.sum() > 0:
                # push to buffer before clearing, if enabled
                if self.use_buffer:
                    dead_ids = track_instances.global_track_id[batch_idx][dead_track]
                    dead_q = track_instances.queries[batch_idx][dead_track]
                    dead_bb = track_instances.bboxes[batch_idx][dead_track]
                    dead_cat = track_instances.category[batch_idx][dead_track]
                    dead_sc = track_instances.scores[batch_idx][dead_track]
                    dead_age = track_instances.track_age[batch_idx][dead_track]
                    dead_lt = track_instances.long_track[batch_idx][dead_track]
                    dead_ac = track_instances.active[batch_idx][dead_track]
                    # only keep valid ids (>= 0)
                    valid_mask = dead_ids >= 0
                    if valid_mask.any():
                        for gid, q, bb, c, s, ta, lt, ac in zip(
                                dead_ids[valid_mask], dead_q[valid_mask], dead_bb[valid_mask], dead_cat[valid_mask],
                                dead_sc[valid_mask], dead_age[valid_mask], dead_lt[valid_mask], dead_ac[valid_mask]):
                            self.buffer_ids.append(int(gid.item()))
                            self.buffer_age.append(0)
                            self.buffer_queries.append(q.detach())
                            self.buffer_bboxes.append(bb.detach())
                            self.buffer_cats.append(c.detach())
                            self.buffer_scores.append(s.detach())
                            self.buffer_track_age.append(ta.detach())
                            self.buffer_long_track.append(lt.detach())
                            self.buffer_active.append(ac.detach())

                # free slots as original behavior
                track_instances.valid_track[batch_idx][dead_track] = False
                track_instances.long_track[batch_idx][dead_track] = False
                track_instances.active[batch_idx][dead_track] = False
                track_instances.track_age[batch_idx][dead_track] = 0
                track_instances.disappear_time[batch_idx][dead_track] = 0
                track_instances.queries[batch_idx][dead_track] = torch.zeros_like(track_instances.queries[batch_idx][dead_track])
                track_instances.bboxes[batch_idx][dead_track] = torch.zeros_like(track_instances.bboxes[batch_idx][dead_track])
                track_instances.scores[batch_idx][dead_track] = torch.zeros_like(track_instances.scores[batch_idx][dead_track])
                track_instances.obj_labels[batch_idx][dead_track] = torch.zeros_like(track_instances.obj_labels[batch_idx][dead_track])
                track_instances.global_track_id[batch_idx][dead_track] = -1 * torch.ones_like(track_instances.global_track_id[batch_idx][dead_track])
        if len(self.cur_scores) > self.inscat_topk_insts:
            _, kept_ins = self.cur_scores.topk(self.inscat_topk_insts)
        else:
            kept_ins = ...
        cur_masks, cur_scores = self.cur_masks[kept_ins], self.cur_scores[kept_ins]
        cur_labels = self.cur_labels[kept_ins]
        # cur_queries = self.cur_queries[kept_ins]
        # cur_bboxes = self.cur_xyz[kept_ins] if self.use_bbox else None
        # assert len(self.cur_masks) == sum(track_instances.valid_track[0]), "len(self.cur_masks) != sum(track_instances.valid_track[batch_idx])"
        # batch_idx = 0
        # new_valid_track_idx = track_instances.valid_track[batch_idx].nonzero(as_tuple=True)[0]
        # new_bboxes = track_instances.bboxes[batch_idx][new_valid_track_idx]
        # new_bboxes_de = bbox_pred_to_bbox(new_bboxes)
        # assert torch.allclose(new_bboxes_de, self.cur_xyz, rtol=1e-3, atol=1e-6), "new_bboxes_de != self.cur_xyz"
        return cur_masks, cur_labels, cur_scores, current_max_track_id
    
    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)
    
class InstanceQuery():
    def __init__(self, mask, label, score, query):
        self.mask = mask
        self.label = label
        self.score = score
        self.query = query
        self.merge_count = 1
    
    def pad(self, pts_num):
        self.mask = torch.cat([self.mask, self.mask.new_zeros(pts_num).bool()])
    
    def compare(self, cur_points, points, mask, label, score, query, pts_thr=0.05, thr=0.1):
        if cur_points.shape[0] != len(self.mask):
            return False
        if self.label != label:
            return False
        cur_xyz = cur_points[self.mask, :3].unsqueeze(1) # Mx3
        if cur_xyz.shape[0] > 10000:
            sample_idx = torch.randperm(cur_xyz.shape[0])[:10000]
            cur_xyz = cur_xyz[sample_idx]
        xyz = points[mask, :3].unsqueeze(0) # Nx3
        if xyz.shape[0] > 10000:
            sample_idx = torch.randperm(xyz.shape[0])[:10000]
            xyz = xyz[sample_idx]
        dist_mat = cur_xyz - xyz # MxNx3
        dist_mat = (dist_mat ** 2).sum(-1).sqrt() # MxN
        min_dist1 = dist_mat.min(-1).values # M
        min_dist2 = dist_mat.min(0).values # N
        ratio1 = (min_dist1 < pts_thr).sum() / len(min_dist1)
        ratio2 = (min_dist2 < pts_thr).sum() / len(min_dist2)
        if max(ratio1, ratio2) > thr:
            return True
        else:
            return False
    
    def merge(self, mask, label, score, query, frame_i):
        self.mask = torch.cat([self.mask, mask])
        self.score = (self.score * frame_i + score) / (frame_i + 1)
        self.query = (self.query * frame_i + query) / (frame_i + 1)
        self.merge_count += 1
def bbox_pred_to_bbox(bbox_pred):
    """Transform predicted bbox parameters to bbox.
    """
    if bbox_pred.shape[0] == 0:
        return bbox_pred
    bbox = bbox_pred
    # x_center = points[:, 0] + bbox_pred[:, 0]
    # y_center = points[:, 1] + bbox_pred[:, 1]
    # z_center = points[:, 2] + bbox_pred[:, 2]
    # bbox = torch.stack([
    #     x_center,
    #     y_center,
    #     z_center,
    #     bbox_pred[:, 3],
    #     bbox_pred[:, 4],
    #     bbox_pred[:, 5]], -1)

    # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)
