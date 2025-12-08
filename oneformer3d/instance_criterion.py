import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmengine.structures import InstanceData
from mmdet3d.registry import MODELS, TASK_UTILS
from torch_scatter import scatter_min 

def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets) # 计算交集
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :] # 计算并集
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss # 重叠越大（Dice 系数越高），损失越小；重叠越小，损失越大。


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


@MODELS.register_module()
class InstanceCriterion:
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(self, matcher, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss=False):
        self.matcher = TASK_UTILS.build(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        return loss

    # todo: refactor pred to InstanceData
    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred['cls_preds']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData(
                labels=insts[i].labels_3d,
                masks=insts[i].sp_masks)
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss)

        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}


@MODELS.register_module()
class MixedInstanceCriterion:
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(self, matcher, bbox_loss, loss_weight, non_object_weight, num_classes,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss=False, first_layer_one2many=False):
        self.matcher = TASK_UTILS.build(matcher)
        self.bbox_loss = MODELS.build(bbox_loss)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.first_layer_one2many = first_layer_one2many

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, mode, indices=None, inst_dict=None, mot_type=None, top_k=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs['cls_preds']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        pred_bboxes = aux_outputs['bboxes']
        centers = aux_outputs['centers']
        if centers is None: centers = [None] * len(pred_masks)

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData(
                    scores=cls_preds[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks if mode == "SP" else insts[i].p_masks)
                if insts[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts[i].query_masks
                if top_k is not None:
                    indices.append(self.matcher(pred_instances, gt_instances, top_k=top_k))
                else:
                    indices.append(self.matcher(pred_instances, gt_instances))
        if inst_dict is not None:
            inst_dict['aux_indices'].append(indices)

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, bbox_losses, mask_bce_losses, mask_dice_losses = [], [], [], []
        for mask, score, bbox, center, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, 
                                                      pred_bboxes, centers, insts, indices):
            if len(inst) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt] if mode == "SP" else inst.p_masks[idx_gt]
            if len(idx_q) == 0:
                mask_bce_losses.append(torch.tensor(0.0).to(pred_mask.device))
                mask_dice_losses.append(torch.tensor(0.0).to(pred_mask.device))
            else:
                mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                pred_mask, tgt_mask.float()))
                mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip bbox loss
            if bbox is not None:
                pred_bbox = bbox[idx_q]
                sp_center = center[idx_q]
                tgt_bbox = inst.bboxes_3d[idx_gt, :6]
                if len(tgt_bbox) == 0:
                    bbox_losses.append(torch.tensor(0.0).to(pred_mask.device))
                else:
                    bbox_loss = self.bbox_loss(
                        self._bbox_to_loss(
                            self._bbox_pred_to_bbox(sp_center, pred_bbox)),
                        self._bbox_to_loss(tgt_bbox))
                    bbox_losses.append(bbox_loss)
            
            # check if skip objectness loss
            if score is not None:
                pred_score = score[idx_q]
                with torch.no_grad():
                    tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

                filter_id, _ = torch.where(tgt_score > 0.5)
                if filter_id.numel():
                    tgt_score = tgt_score[filter_id]
                    pred_score = pred_score[filter_id]
                    score_losses.append(F.mse_loss(pred_score, tgt_score))

        # todo: actually .mean() should be better
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).sum() / len(pred_masks)
        else:
            bbox_loss = 0
        
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss +
            self.loss_weight[4] * bbox_loss)
        if mot_type is None:
            return loss
        elif mot_type == 'motr':
            return loss, indices
    
    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)

    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        return base_bbox

    # todo: refactor pred to InstanceData
    def __call__(self, pred, insts, mask_pred_mode, use_temporal_loss=False, mot_type=None, track_instances=None, matched_dict=None, use_one2many=False):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData with
                    `sp_masks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).
        
        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred['cls_preds'] # [N_segment, 19]
        pred_scores = pred['scores'] # ? 这个都是None
        pred_masks = pred['masks'] # [N_segment, 20000]
        pred_bboxes = pred['bboxes'] # [N_segment, 6]
        centers = pred['centers'] # [N_segment, 3]
        if centers is None: centers = [None] * len(pred_masks)
        if use_temporal_loss:
            inst_dict = {'indices':[],
                         'aux_indices':[]}

        # match
        indices = []
        for i in range(len(insts)): # batch_size
            pred_instances = InstanceData(
                scores=cls_preds[i],
                masks=pred_masks[i])
            gt_instances = InstanceData(
                labels=insts[i].labels_3d,
                masks=insts[i].p_masks) # mask_pred_mode[-1] is "P"
            if insts[i].get('query_masks') is not None:
                gt_instances.query_masks = insts[i].query_masks
            # All-False-gt_mask will not be matched 
            indices.append(self.matcher(pred_instances, gt_instances, matched_dict=matched_dict[i] if matched_dict is not None else None))
            # assert len(indices[i][-1].unique()) == len(indices[i][-1]),  f"Indices {indices[-1]} are not unique, please check the matcher."
        if use_temporal_loss:
            inst_dict['indices'] = indices
        # class loss 分类损失
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss = torch.mean(torch.stack(cls_losses))
        if mot_type is not None:
            matched_indices = indices
        # 3 other losses 掩码损失、BBox损失、得分损失
        score_losses, bbox_losses, mask_bce_losses, mask_dice_losses = [], [], [], []
        for mask, score, bbox, center, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, 
                                                      pred_bboxes, centers, insts, indices):
            if len(inst) == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst.p_masks[idx_gt]
            if len(idx_q) == 0:
                mask_bce_losses.append(torch.tensor(0.0).to(pred_mask.device))
                mask_dice_losses.append(torch.tensor(0.0).to(pred_mask.device))
            else:
                mask_bce_losses.append(F.binary_cross_entropy_with_logits(
                    pred_mask, tgt_mask.float()))
                mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip bbox loss
            if bbox is not None: # 检测框损失
                pred_bbox = bbox[idx_q]
                sp_center = center[idx_q]
                tgt_bbox = inst.bboxes_3d[idx_gt, :6]
                if len(tgt_bbox) == 0:
                    bbox_losses.append(torch.tensor(0.0).to(pred_mask.device))
                else:
                    bbox_loss = self.bbox_loss(
                        self._bbox_to_loss(
                            self._bbox_pred_to_bbox(sp_center, pred_bbox)),
                        self._bbox_to_loss(tgt_bbox))
                    bbox_losses.append(bbox_loss)

            # check if skip objectness loss
            if score is not None: # 这个没有用到
                pred_score = score[idx_q]
                with torch.no_grad():
                    tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

                filter_id, _ = torch.where(tgt_score > 0.5)
                if filter_id.numel():
                    tgt_score = tgt_score[filter_id]
                    pred_score = pred_score[filter_id]
                    score_losses.append(F.mse_loss(pred_score, tgt_score))

        # todo: actually .mean() should be better
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).sum() / len(pred_masks)
        else:
            bbox_loss = 0
        
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0
        
        if len(mask_bce_losses): # 4
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight: # True
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss: # True
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss +
            self.loss_weight[1] * mask_bce_loss +
            self.loss_weight[2] * mask_dice_loss +
            self.loss_weight[3] * score_loss +
            self.loss_weight[4] * bbox_loss)

        if 'aux_outputs' in pred: # True
            if mot_type is None:
                if self.iter_matcher: # True # ?为啥要重新匹配
                    indices = None
                for i, aux_outputs in enumerate(pred['aux_outputs']): # 计算每一层的损失
                    if use_temporal_loss:
                        loss += self.get_layer_loss(aux_outputs, insts, mask_pred_mode[i], indices, inst_dict)
                    else:
                        if i == 0 and self.first_layer_one2many:
                            loss += self.get_layer_loss(aux_outputs, insts, mask_pred_mode[i], indices, top_k=4)
                        else:
                            loss += self.get_layer_loss(aux_outputs, insts, mask_pred_mode[i], indices)

            elif mot_type == 'motr':
                for i, aux_outputs in enumerate(pred['aux_outputs']):
                    # loss += self.get_layer_loss(aux_outputs, insts, mask_pred_mode[i], indices)
                    loss += self.get_layer_loss(aux_outputs, insts, mask_pred_mode[i], indices)
            else:
                raise NotImplementedError(f"mot_type {mot_type} not implemented")

        if mot_type is None:
            if use_temporal_loss:
                return {'inst_loss': loss}, inst_dict
            else:
                return {'inst_loss': loss}
        elif mot_type == 'motr':
            return {'inst_loss': loss}, matched_indices
        else:
            raise NotImplementedError(f"mot_type {mot_type} not implemented")

@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1) # [31, 2]
        cost = -scores[:, gt_instances.labels] # [31, 17]
        return cost * self.weight # [31, 17]


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """
    def __init__(self, costs):
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)
        # cost_value = torch.where(
        #     gt_instances.query_masks.T, cost_value, self.inf) 
        # 如果额外传了 matched_dict 进来，需要只匹配 valid_gt_idx，
        # 并且保持之前已匹配 (track_idx, gt_idx) 不变
        if 'matched_dict' in kwargs and kwargs['matched_dict'] is not None:
            matched_dict = kwargs['matched_dict']
            valid_gt_idx = matched_dict['valid_gt_idx']  # 待匹配的一批 GT
            track_idx = matched_dict['track_idx']        # 已匹配好的 query
            gt_idx = matched_dict['gt_idx']              # 已匹配好的 GT
            current_obj_idxes = matched_dict['current_obj_idxes'] # 当前帧的 preds

            # # 移除当前帧没匹配上的pred
            # is_with_pred_mask = (current_obj_idxes != -1).nonzero(as_tuple=False).squeeze(1)
            
            # track_idx = track_idx[is_with_pred_mask]
            # gt_idx = gt_idx[is_with_pred_mask]

            # current_obj_idxes = current_obj_idxes[is_with_pred_mask]

            # ---------- 移除已经匹配过的 gt_idx 和 pred_idx -----------
            is_in_mask = torch.isin(valid_gt_idx, gt_idx)
            unmatched_gt_idx = valid_gt_idx[~is_in_mask]  # 只取未匹配过的
            
            pred_index = torch.arange(len(pred_instances.scores), device=labels.device)
            is_in_mask_pred = torch.isin(pred_index, current_obj_idxes)
            unmatched_pred_idx = pred_index[~is_in_mask_pred]
            # 筛选有效的gt_idx 和 pred_idx
            is_valid_track_gt = torch.isin(gt_idx, valid_gt_idx)
            is_valid_track_pred = torch.isin(current_obj_idxes, pred_index)
            valid_track = is_valid_track_gt & is_valid_track_pred
            # 如果已经没有可匹配的 GT 了，就直接返回原先匹配的对
            if unmatched_gt_idx.numel() == 0 or unmatched_pred_idx.numel() == 0:
                return current_obj_idxes[valid_track], gt_idx[valid_track]
            # cost_value_valid = cost_value[unmatched_pred_idx][:, unmatched_gt_idx]
            # v_query_ids, v_object_ids = linear_sum_assignment(cost_value_valid.cpu().numpy())
            # print(v_query_ids, v_object_ids)
            valid_mask = gt_instances.query_masks.T
            cost_value = torch.where(
                valid_mask, cost_value, self.inf) # [31, 27]
            # 只对 unmatched_gt_idx 做线性分配
            cost_value_valid = cost_value[unmatched_pred_idx][:, unmatched_gt_idx]
            v_query_ids, v_object_ids = linear_sum_assignment(cost_value_valid.cpu().numpy())
            # print('after', v_query_ids, v_object_ids)
            v_query_ids = torch.as_tensor(v_query_ids, dtype=torch.long, device=labels.device)
            v_object_ids = torch.as_tensor(v_object_ids, dtype=torch.long, device=labels.device)

            # 查看是否非法
            keep = cost_value_valid[v_query_ids, v_object_ids] != self.inf
            v_query_ids = v_query_ids[keep]
            v_object_ids = v_object_ids[keep]
            # print('after keep', v_query_ids, v_object_ids)

            # 将 v_object_ids v_query_ids 映射回原始 GT 索引空间
            v_object_ids = unmatched_gt_idx[v_object_ids]
            v_query_ids = unmatched_pred_idx[v_query_ids]
            
            # 合并旧匹配和新匹配
            final_pred_ids = torch.cat([current_obj_idxes[valid_track], v_query_ids], dim=0)
            final_gt_ids = torch.cat([gt_idx[valid_track], v_object_ids], dim=0)

            return final_pred_ids, final_gt_ids
        else:
            raise NotImplementedError("Please provide matched_dict to use this matcher.")
            query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())
            return labels.new_tensor(query_ids), labels.new_tensor(object_ids)
        # if 'matched_dict' in kwargs:
        #     matched_dict = kwargs['matched_dict']
        #     # 只匹配有效的gt
        #     valid_gt_idx = matched_dict['valid_gt_idx'] 
        #     # 已经匹配的track_idx和gt_idx保持不变
        #     track_idx = matched_dict['track_idx']
        #     gt_idx = matched_dict['gt_idx']
        
        # query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())
        # return labels.new_tensor(query_ids), labels.new_tensor(object_ids)


@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances)) # 类别损失 BCE(mask) DICE
        # of shape (n_queries, n_gts) 
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf) # [31, 27]
        if 'top_k' in kwargs:
            top_k = kwargs['top_k']
        else:
            top_k = self.topk
        n_queries = cost_value.size(0)
        k_star = min(top_k + 1, n_queries)
        values = torch.topk( # 最小的 topk+1 个值中的最大值
            cost_value, k_star, dim=0, sorted=True,        
            largest=False).values[-1:, :] # [1, 27]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]

@TASK_UTILS.register_module()
class One2Many_Matcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """
    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances)) # 类别损失 BCE(mask) DICE
        # of shape (n_queries, n_gts) 
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf) # [31, 27]
        n_queries = cost_value.size(0)
        k_star = min(self.topk + 1, n_queries)
        if k_star == 0:                                # 极端兜底
            return labels.new_empty((0,)), labels.new_empty((0,))
        values = torch.topk( # 最小的 topk+1 个值中的最大值
            cost_value, k_star, dim=0, sorted=True,        
            largest=False).values[-1:, :] # [1, 27]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]
