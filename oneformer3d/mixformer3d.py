import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter
import MinkowskiEngine as ME
import pointops
import pdb, time
from functools import partial
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmengine.structures import InstanceData
from .mask_matrix_nms import mask_matrix_nms
from .oneformer3d import ScanNetOneFormer3DMixin
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge, GTMerge, DQ_Track_OnlineMerge
import numpy as np
from .img_backbone import point_sample
import os
# Added
from mmdet3d.registry import TASK_UTILS
from mmengine.model import BaseModule
from oneformer3d.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from oneformer3d.util import box_ops, checkpoint
from oneformer3d.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .dq_utils import match_for_indices, QueryInteractionX, build_pairwise_mask, cluster_with_threshold, analyze_masked_stats, find_optimal_threshold, calc_mean_grouped, frozen_inference, MergeFusion, sigmoid_focal_loss, cluster_complete_link, evaluate_clustering_pairwise,cluster_with_per_class_threshold
from .dq_utils import FFN as DQ_FFN
from .dq_utils import replace_bn_with_ln as replace_bn
from .motr_utils import SelfAttention as MOTR_SelfAttention
from .motr_utils import FFN as MOTR_FFN
from scipy.optimize import linear_sum_assignment
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
from easydict import EasyDict
@MODELS.register_module()
class ScanNet200MixFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse()) # [N_segment, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [batch_size * 20000, 96]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask # [20000, 96]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz)

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)): # batch_size
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples) # batch_size * [N_segment, 96] batch_size * [200000, 99] [batch_size * 200000, 1] 
        ## GT-prepare
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        gt_point_instances = []
        for i in range(len(gt_instances)): # batch_size
            ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask # [20000]
            if torch.sum(ins == -1) != 0:
                ins[ins == -1] = torch.max(ins) + 1
                ins = F.one_hot(ins)[:, :-1]
            else:
                ins = F.one_hot(ins)
            ins = ins.bool().T
            gt_point = InstanceData()
            gt_point.p_masks = ins
            gt_point_instances.append(gt_point)
        queries, gt_instances = self._select_queries(x, gt_instances) # 随机选出 0.5 ~ 1数量的query
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w) # 每个点的segment ID以及归一化权重
        x = self.decoder(x, point_features, queries, super_points, use_one2many=self.use_one2many) # [N_segment, 96] [20000, 99] [(0.5 ~ 1) * N_segment, 96] ([20000, 1])
        loss = self.criterion(x, gt_instances, gt_point_instances, None, self.decoder.mask_pred_mode)
        if self.use_one2many:
            loss_one2many = self.criterion_one2many(x['one2many_outputs'], gt_instances, gt_point_instances, None, self.decoder.mask_pred_mode, use_one2many=self.use_one2many)
            for key, value in loss_one2many.items():
                loss_one2many[key] = value * self.one2many_loss_weight
            loss.update(loss_one2many)
        ## Loss
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)
        x = self.decoder(x, point_features, x, super_points)
        ## Post-processing
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

@MODELS.register_module()
class ScanNet200MixFormer3D_FF(ScanNet200MixFormer3D):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()
    
    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

                
        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # extract image features
        with torch.no_grad():
            img_features = self.img_backbone(batch_inputs_dict['img_path']) # batch_size * [C, H, W][960, 60, 80]
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
        
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']))
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz)

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x

@MODELS.register_module()
class ScanNet200MixFormer3D_Online(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 map_to_rec_pcd=True,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,

                 use_query_memory=False,
                 use_self_attn=False,
                 use_noise=False,
                 noise_p=0.05,
                 noise_k=10,
                 use_temporal_loss=False,
                 use_decouple=False,
                 use_mot=False,
                 mot_type='motr',
                 train_asso_only=False,
                 matcher=None,
                 use_aug=False,
                 asso_loss_weight=0.5,
                 use_refine=False,
                 asso_config=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 use_3d_refine=False,
                 reweight_dict=None,
                 use_relative_asso=False,
                 merge_sp_masks = False,
                 replace_bn_with_ln=False,
                 debug_mode=False
                 ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.map_to_rec_pcd = map_to_rec_pcd
        # self.use_query_memory = decoder['use_query_memory']
        self.use_query_memory = use_query_memory
        if self.use_query_memory:
            self.muti_scale_query = MultiScaleQuery()
            self.query_memory = None
            self.pos_memory = None
            self.query_memory_relu = nn.ReLU()
            self.muti_scale_query.init_weights()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.muti_scale_self_attn = MultiScaleQuery()
            self.self_attn_relu = nn.ReLU()
            self.muti_scale_self_attn.init_weights()
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_p = noise_p
            self.noise_k = noise_k
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.before_query_ids = None
        self.use_decouple = use_decouple
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_query_boxes = None
        self.use_mot = use_mot
        if self.use_mot:
            self.mot_type = mot_type
            if mot_type == 'dq_track':
                self.use_relative_asso = use_relative_asso
                if self.use_relative_asso:
                    self.embed_trans2 = nn.Linear(256, 1)
                    self.heatmap_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                self.use_refine = use_refine
                # self.asso_loss_weight = asso_loss_weight
                self.iou_calculator = AxisAlignedBboxOverlaps3D()
                self.matcher = TASK_UTILS.build(matcher)
                self.tracklet_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.detector_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.box_trans = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                # query_trans = {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                query_trans = asso_config['query_trans'] if asso_config is not None else {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                self.query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
                # self.update_type = asso_config['update_type'] if asso_config is not None else 'ema'
                self.asso_config = EasyDict(asso_config)
                
                self.rel_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                self.embed_trans = nn.Linear(256, 1)
                loss_asso = {'use_sigmoid': False, 'loss_weight': 1.0}
                from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
                self.loss_asso = CrossEntropyLoss(**loss_asso)
                self.ema_decay_rate = 0.5
                # self.train_asso_only = train_asso_only
            else:
                raise NotImplementedError(f"mot_type {mot_type} is not supported")

        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])
        self.reweight_dict = reweight_dict
        self.merge_sp_masks = merge_sp_masks
        if self.merge_sp_masks:
            self.acc_dict = {}
            self.merge_box_trans = nn.Sequential(
                nn.Linear(6, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_embed_trans = nn.Linear(256, 1)
            self.merge_heatmap_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.merge_iou_calculator = AxisAlignedBboxOverlaps3D()
            self.fuse_linear = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256)
            )
            self.merge_fusion = MergeFusion(256)
            query_trans = {'with_att': True, 'with_pos': False, 'min_channels': 256, 'drop_rate': 0.0}
            self.merge_query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
        self._prev_param_snapshot = None
        if replace_bn_with_ln:
            replace_bn(self)
        self.debug_mode = debug_mode
        self.init_weights()
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
            
    def reset_query_memory(self):
        """Reset the detector.
        """
        if self.use_query_memory:
            self.query_memory = None
            self.pos_memory = None

    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i, track_instances=None):
        """Extract features from sparse tensor.
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates # [20000, 3]

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features) 

        # forward of backbone and neck
        x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [20000, 99] 坐标和特征进行拼接
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone() # [20000] 每个点属于的segment ID
            if self.use_temporal_loss and self.inst_dict is not None:
                points_xyz = all_xyz[batch_idx]  # (N,3)
                point_ids  = sp_pts_mask # (N,)dd
                bboxes_6d  = self.inst_dict['bboxes_3d'][batch_idx]  # (M,6)

            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) # [20000] 每个点在所有点中的ID
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]

        if self.use_temporal_loss and self.inst_dict is not None:
            self.inst_dict['best_obj_ids_list'] = best_obj_ids_list
        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list # [N_segment, 96], [20000, 99], [20000, 1], [N_segment, 3]

    def merge_superpixels_extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i, overlap_threshold=0.99):
        """Merge superpixels and extract features once.

        This integrates feature extraction with superpixel merging to avoid
        recomputing backbone/neck. It first pools by current SPs to decide
        merges, updates masks, then pools again to produce final features.

        Returns: features, point_features, all_xyz_w, sp_xyz_list
        """
        # 1) image features (same as extract_feat) — support no-image cases
        use_images = ('img_paths' in batch_inputs_dict) and hasattr(self, 'img_backbone')
        if use_images:
            with torch.no_grad():
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]

        # 2) construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates

        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # 3) backbone + neck (once)
        if use_images:
            x = self.backbone(field.sparse(),
                              partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                              memory=self.memory if hasattr(self,'memory') else None)
        else:
            x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # 4) pool with current superpixels to compute merging decisions
        sp_pts_masks, n_super_points = [], []
        current_sp_pts_mask = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
        for sp_pts_mask in current_sp_pts_mask:
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x_pooled, all_xyz_w_orig = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        features_orig, sp_xyz_list_orig = [], []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features_orig.append(x_pooled[begin: end, :-3])
            sp_xyz_list_orig.append(x_pooled[begin: end, -3:])

        # predictor for per-SP detections (lightweight path)
        x_detach = [features_orig[i] for i in range(len(features_orig))]
        pred_bboxes, pred_cls_list, new_queries = [], [], []
        queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask))
        for i in range(len(queries)):
            norm_query = self.decoder.out_norm(queries[i])
            reg_final = self.decoder.out_reg(norm_query)
            reg_cls = self.decoder.out_cls(norm_query)
            reg_cls = reg_cls.softmax(1)
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            pred_cls_list.append(reg_cls)
            pred_bboxes.append(pred_bbox)
            new_queries.append(norm_query)
        x_detach = [new_queries[i] for i in range(len(new_queries))]

        valid_sps_list = []
        for batch_idx in range(len(pred_bboxes)):
            labels = pred_cls_list[batch_idx].argmax(dim=1)
            bg = pred_cls_list[batch_idx].shape[1] - 1
            labels_mask = ((labels[:, None] == labels) & (labels[:, None] != bg) & (labels != bg)[:, None])
            det_bboxes = pred_bboxes[batch_idx].clone()
            det_bboxes[:, :3] += sp_xyz_list_orig[batch_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx], x_detach[batch_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1)
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist)
            merge_appear_embedding = obj_embedding1[:, None] * obj_embedding2[None]
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            m = merge_det_mat.sigmoid()
            iou_map = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='iou')
            cluster_both = cluster_complete_link(iou_map * m * labels_mask.float(), 0.5)
            valid_sps_list.append(cluster_both)

        # generate merged masks and update in-place
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            merged_groups = []
            for valid_sps in valid_sps_list[batch_idx]:
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0]
            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            new_id = 0
            id_mapping = {}
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            final_mask = merged_mask.clone()
            for old_id, nid in id_mapping.items():
                final_mask[merged_mask == old_id] = nid
            batch_data_samples[batch_idx].gt_pts_seg.sp_pts_mask[frame_i] = final_mask

        # 5) pool with merged superpixels to produce final features
        sp_pts_masks_new, n_super_points_new = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks_new.append(sp_pts_mask + sum(n_super_points_new))
            n_super_points_new.append(sp_pts_mask.max() + 1)
        sp_idx_new = torch.cat(sp_pts_masks_new)
        x_pooled_new, all_xyz_w = self.pool(x, sp_idx_new, all_xyz, with_xyz=True)
        features_final, sp_xyz_list = [], []
        for i in range(len(n_super_points_new)):
            begin = sum(n_super_points_new[:i])
            end = sum(n_super_points_new[:i + 1])
            features_final.append(x_pooled_new[begin: end, :-3])
            sp_xyz_list.append(x_pooled_new[begin: end, -3:])
        return features_final, point_features, all_xyz_w, sp_xyz_list
    
    def _select_queries(self, x, gt_instances, sp_xyz, frame_i):
        """Select queries for train pass.
        """

        gt_instances_ = []
        for i in range(len(x)): # batch_size
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(x[i].device)
            temp.bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(x[i].device)])
            gt_instances_.append(temp)

        if self.use_temporal_loss:
            with torch.no_grad():
                before_query_memory = [x[i].clone() for i in range(len(x))]
                before_mask_memory = [gt_instances_[i].sp_masks.clone() for i in range(len(gt_instances_))]
                before_sp_xyz = [sp_xyz[i].clone() for i in range(len(sp_xyz))]
            if self.before_mask_memory is not None:
                for i in range(len(x)):
                    gt_instances_[i].sp_masks = torch.cat([self.before_mask_memory[i], gt_instances_[i].sp_masks], dim=1) # [20000, 2]
                    sp_xyz[i] = torch.cat([self.before_sp_xyz[i], sp_xyz[i]], dim=0) # [20000, 3]
                    x[i] = torch.cat([self.before_query_memory[i], x[i]], dim=0) # [20000, 96]
            with torch.no_grad():
                self.before_query_memory = before_query_memory
                self.before_mask_memory = before_mask_memory
                self.before_sp_xyz = before_sp_xyz
        
        queries = []
        for i in range(len(x)): # batch_size
            if self.query_thr < 1: # 0.5 
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).ceil().int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
                sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
      
        return queries, gt_instances_, sp_xyz
    def _select_queries_predict(self, device, gt_instances, frame_i):
        """Select queries for train pass.
        """



        gt_instances_ = [] # 
        for i in range(len(gt_instances)): # batch_size
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(device)
            bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(device)
            temp.bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(device)])
            gt_instances_.append(temp)

        for i in range(len(gt_instances)): # batch_size
            gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        return gt_instances_
    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def _process_frame_queries(self, track_instances, active_queries_num):
        """Process and update track_instances for the given frame."""
        # Handle query initialization and self-attention
        for i in range(len(track_instances.queries)):
            track_instances.queries[i] = self.input_proj(track_instances.queries[i])
            track_instances.queries[i] = self.self_attn(track_instances.queries[i],active_queries_num[i])
            track_instances.queries[i] = track_instances.queries[i].squeeze(0)
            track_instances.queries[i] = self.ffn(track_instances.queries[i])
            track_instances.queries[i] = self.output_proj(track_instances.queries[i])

        return track_instances

    def _init_query(self, queries, mot_type='motr'):

        track_instances = Instances((1, 1))
        device = next(self.backbone.parameters()).device

        fields = {
            'obj_idxes':        (None,    torch.long,    -1,    True),
            'matched_gt_idxes': (None,    torch.long,    -1,    True),
            'cls_preds':        (2,       torch.float32, -1,    False),
            'sem_preds':        (201,     torch.float32, -1,    False),
            'masks':            (20000,   torch.float32, -1,    False),
            'bboxes':           (6,       torch.float32, -1,    False),
            'fp_flag':          (None,    torch.bool,    False, True),
            'track_age':        (None,    torch.int,      0,    True),
        }

        for name, (feat_dim, dtype, fill_val, is_1d) in fields.items():
            data = [
                torch.full(
                    (q.shape[0],) if is_1d else (q.shape[0], feat_dim),
                    fill_val, dtype=dtype, device=device
                )
                for q in queries
            ]
            setattr(track_instances, name, data)

        if mot_type == 'dq_track':
            track_instances.queries = [
                torch.full((q.shape[0], 256), -1, dtype=torch.float32, device=device)
                for q in queries
            ]
        else:
            raise NotImplementedError(f"mot_type {mot_type} is not supported")

        return track_instances.to(device)
    def _init_query_test(self, queries, mot_type='motr', mode='train', fix_num=500, batch_size=1):
        track_instances = Instances((1, 1))
        device = next(self.backbone.parameters()).device
        fields = {
            # 数值型字段: 填 -1
            'obj_idxes':        (None,    torch.long,    -1,    True),
            'obj_labels':       (None,    torch.long,    -1,    True), 
            'global_track_id': (None,    torch.long, -1,   True),
            'category':        (None,    torch.long,    -1,    True),
            # 'matched_gt_idxes': (None,    torch.long,    -1,    True),
            # 'current_obj_idxes': (None,    torch.long,    -1,    True),
            'cls_preds':        (2,       torch.float32, -1,    False),
            'scores':           (None,    torch.float32, -1,    True),
            # 'sem_preds':        (201,     torch.float32, -1,    False),
            # 'masks':            (20000,   torch.float32, -1,    False),
            'bboxes':           (6,       torch.float32, -1,    False),
            # 布尔型或需要特殊初始值的
            'long_track':       (None,    torch.bool,    False, True),
            'valid_track':      (None,    torch.bool,    False, True),
            'active':           (None,    torch.bool,    False, True),
            # 浮点型零填充
            'track_age':        (None,    torch.int, 0,   True),
            'disappear_time':   (None,    torch.int, 0,   True),
            
        }


        for name, (feature_dim, dtype, fill_val, is_1d) in fields.items():
            data = [
                torch.full((fix_num,) if is_1d else (fix_num, feature_dim), fill_val, dtype=dtype, device=device)
                for batch_idx in range(batch_size)
            ]
            setattr(track_instances, name, data)

        # Store queries as a list
        if mot_type == 'dq_track':
            track_instances.queries = [torch.full((fix_num, 256), 0, dtype=torch.float32, device=device) for batch_idx in range(batch_size)]
        else:
            raise NotImplementedError(f"mot_type {mot_type} is not supported")

        return track_instances.to(device)
    def update_untracked_gt_instances(self, gt_instances, gt_point_instances, untracked_tgt_indexes, new_indexes):
        """Update and create untracked GT instances."""
        untracked_gt_instances, untracked_gt_points_instances = [], []

        untracked_tgt_indexes_gt = [
            torch.cat([untracked_tgt_indexes[i], new_indexes[i]], dim=0) 
            for i in range(len(untracked_tgt_indexes))
        ]
        for idx,gt in enumerate(gt_instances):
            new_gt = gt[untracked_tgt_indexes_gt[idx]]
            new_gt.labels_3d = gt.labels_3d[untracked_tgt_indexes_gt[idx]]
            new_gt.bboxes_3d = gt.bboxes_3d[untracked_tgt_indexes_gt[idx]]
            new_gt.sp_masks = gt.sp_masks[untracked_tgt_indexes_gt[idx]]
            new_gt.query_masks = gt.query_masks[untracked_tgt_indexes_gt[idx]]
            untracked_gt_instances.append(new_gt)

        for idx,gt_points in enumerate(gt_point_instances):
            new_gt_points = gt_points[untracked_tgt_indexes[idx]]
            new_gt_points.p_masks = gt_points.p_masks[untracked_tgt_indexes[idx]]
            untracked_gt_points_instances.append(new_gt_points)

        return untracked_gt_instances, untracked_gt_points_instances

    def _empty_single_track(self,active_track_instances):
        # track_instances = Instances((1,1))
        device = next(self.backbone.parameters()).device

        active_track_instances.obj_idxes.append(torch.empty(0, dtype=torch.long, device=device))
        active_track_instances.matched_gt_idxes.append(torch.empty(0, dtype=torch.long, device=device))
        # active_track_instances.scores.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.cls_preds.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.sem_preds.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.masks.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.bboxes.append(torch.empty(0, dtype=torch.float, device=device))
        active_track_instances.queries.append(torch.empty(0, dtype=torch.float, device=device))
        return active_track_instances.to(device)
    
    def _empty_query(self, track_instances_old):
        track_instances = Instances((1,1))
        device = next(self.backbone.parameters()).device
        for key_name in track_instances_old.get_fields().keys():
            setattr(track_instances, key_name, [])
        return track_instances.to(device)
    
    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data['track_instances']
        active_track_instances = self._empty_query(track_instances)
        for batch_idx in range(len(track_instances.matched_gt_idxes)):
            if self.training:
                active_idxes = (track_instances.matched_gt_idxes[batch_idx] >= 0)
                # active_track_instances = track_instances[active_idxes]
                for key_name in track_instances.get_fields().keys():
                    # track_instances[key_name] = track_instances[key_name][batch_idx][active_idxes]
                    active_value = getattr(track_instances, key_name)[batch_idx][active_idxes]
                    getattr(active_track_instances, key_name).append(active_value)


            else:
                active_track_instances = track_instances[track_instances.matched_gt_idxes[batch_idx] >= 0]

        return active_track_instances
    def update_track_instances(self, track_instances, gt_instances, indices, unmatched_track_idxes_list, untracked_tgt_indexes_list, gt_num_list, is_last, track_embedding_for_update_list=None, unmatched_track_embedding_for_update_list=None):
        track_aug=dict(
            # drop_prob=0,
            fp_ratio=0.5,
            # trans_noise=0.0,
            )
        for batch_idx in range(len(gt_instances)):             
            current_matched_pred_id = indices[batch_idx][0]
            current_matched_gt_id = indices[batch_idx][1] 
            before_tracked_num = (track_instances.matched_gt_idxes[batch_idx] >= 0).sum()
            new_matched_mask = ~torch.isin(current_matched_gt_id, track_instances.matched_gt_idxes[batch_idx][:before_tracked_num])
            new_matched_gt_index = new_matched_mask.nonzero(as_tuple=True)[0]
            use_aug = self.asso_config.get('use_aug', False)
            if len(new_matched_gt_index) == 0:
                if use_aug and track_aug['fp_ratio'] > 0:
                    fp_track = (track_instances.matched_gt_idxes[batch_idx] == 99999)
                    if fp_track.sum() > 0: 
                        track_instances.obj_idxes[batch_idx][fp_track] = -1
                        track_instances.matched_gt_idxes[batch_idx][fp_track] = -1
                continue
            
            new_matched_pred_index = current_matched_pred_id[new_matched_gt_index]
            
            track_instances.obj_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            track_instances.matched_gt_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            if track_embedding_for_update_list is not None:
                track_instances.queries[batch_idx][new_matched_pred_index + before_tracked_num] = track_embedding_for_update_list[batch_idx][new_matched_mask]
            # Data Augmentation
            if use_aug and track_aug['fp_ratio'] > 0:
                current_scores = F.softmax(track_instances.cls_preds[batch_idx], dim=-1)[:, 0]
                fp_track = (track_instances.matched_gt_idxes[batch_idx] == 99999)
                
                if fp_track.sum() > 0: 
                    track_instances.obj_idxes[batch_idx][fp_track] = -1
                    track_instances.matched_gt_idxes[batch_idx][fp_track] = -1
                # select background det embedding
                det_mask = torch.ones_like(current_scores).bool() #
                det_mask[:before_tracked_num] = False 
                det_mask[before_tracked_num + current_matched_pred_id] = False 
                # select fp embedding according to prob
                fp_mask = det_mask & (current_scores > 0.0) 
                current_scores[~det_mask] = 0 
                fp_num = int(fp_mask.sum() * track_aug['fp_ratio'])
                if fp_mask.sum() > fp_num: 
                    score_sort = torch.argsort(current_scores, descending=True)
                    assert score_sort[fp_num:].max() < len(fp_mask), f"score_sort[fp_num:]={score_sort[fp_num:]} len(fp_mask)={len(fp_mask)}"
                    fp_mask[score_sort[fp_num:]] = False 
                fp_indices = fp_mask.nonzero(as_tuple=True)[0] 
                if len(fp_indices) == 0:
                    continue
                track_instances.fp_flag[batch_idx][fp_indices] = True 
                
                track_instances.queries[batch_idx][fp_indices] = unmatched_track_embedding_for_update_list[batch_idx][fp_indices - before_tracked_num] #将假阳性目标的查询嵌入更新为 track_embedding_for_update_list 中对应的值
                track_instances.matched_gt_idxes[batch_idx][fp_indices] = 99999

        # select active tracks
        tmp = {} 
        tmp['track_instances'] = track_instances 
        if not is_last:
            out_track_instances = self._select_active_tracks(tmp) 
            # frame_res['track_instances'] = out_track_instances
        else:
            out_track_instances = None
            # frame_res['track_instances'] = None        
        # track_instances = frame_res['track_instances']
        return out_track_instances
    def update_track_instances_predict(self, track_instances, gt_instances, indices, is_last):

        for batch_idx in range(len(gt_instances)):             
            current_matched_pred_id = indices[batch_idx][0]
            current_matched_gt_id = indices[batch_idx][1] 
            before_tracked_num = (track_instances.matched_gt_idxes[batch_idx] >= 0).sum()
            new_matched_gt_index = (~torch.isin(current_matched_gt_id, track_instances.matched_gt_idxes[batch_idx][:before_tracked_num])).nonzero(as_tuple=True)[0]
            if len(new_matched_gt_index) == 0:
                continue
            new_matched_pred_index = current_matched_pred_id[new_matched_gt_index]
            track_instances.obj_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]
            track_instances.matched_gt_idxes[batch_idx][new_matched_pred_index + before_tracked_num] = current_matched_gt_id[new_matched_gt_index]

        # 
        # select active tracks
        tmp = {} 
        tmp['track_instances'] = track_instances 
        if not is_last:
            out_track_instances = self._select_active_tracks(tmp) 
            # frame_res['track_instances'] = out_track_instances
        else:
            out_track_instances = None

        return out_track_instances
    def _check_param_diffs(self):
        curr_params = {
            name: p.detach().cpu().clone()
            for name, p in self.named_parameters()
        }
        curr_buffers = {
            name: b.detach().cpu().clone()
            for name, b in self.named_buffers()
            if "running_mean" in name or "running_var" in name
        }

        if self._prev_param_snapshot is None:
            self._prev_param_snapshot = curr_params
            self._prev_buffer_snapshot = curr_buffers
            return

        for name, prev in self._prev_param_snapshot.items():
            now = curr_params[name]
            diff = (now - prev).abs().view(-1)
            max_diff = diff.max().item()
            if max_diff != 0:
                print(f"[Param Δ]  {name:40s} max|Δ| = {max_diff:.5e}")

        for name, prev in self._prev_buffer_snapshot.items():
            now = curr_buffers[name]
            diff = (now - prev).abs().view(-1)
            max_diff = diff.max().item()
            if max_diff != 0:
                print(f"[Buffer Δ] {name:40s} max|Δ| = {max_diff:.5e}")

        self._prev_param_snapshot  = curr_params
        self._prev_buffer_snapshot = curr_buffers

    def get_loss_track(self, track_instances, current_dict, mot_type):
        loss_track = 0
        return loss_track
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.
        """
        if self.debug_mode:
            self._check_param_diffs()
        losses, merge_feat_n_frames, ins_masks_query_n_frames = {}, [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        if self.use_query_memory:
            self.reset_query_memory()
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.inst_dict = None
        if self.decoder.use_query_memory2:
            self.decoder.reset_query_memory2()
        if self.use_decouple:
            self.decoder.reset_decouple()
        if self.use_mot:
            use_after_features = True
            self.merge_type = 'count'
        if self.merge_sp_masks:
            merge_loss = torch.tensor(0.0).to(batch_inputs_dict['points'][0].device)
        for frame_i in range(num_frames): 
            if self.merge_sp_masks:
                current_sp_pts_mask = [bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples]
                current_pt_instace_mask = [bds.gt_pts_seg.pts_instance_mask[frame_i] for bds in batch_data_samples]
                merge_loss += self.merge_superpixels_train(batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instace_mask)
                
                if not self.use_mot:
                    if frame_i != num_frames - 1:
                        continue
                    else:
                        loss = {'merge_mask_loss': merge_loss}
                        return loss
            else:
                merged_sp_pts_masks = None
                merged_sp_masks = None
            ## Backbone
            if self.use_mot:
                if self.mot_type == 'dq_track':
                    if self.asso_config.get('train_asso_only', True):
                        with frozen_inference(self.backbone), frozen_inference(self.memory):
                            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
                    else:
                        raise NotImplementedError(f"mot_type {self.mot_type} and train_asso_only {self.asso_config.get('train_asso_only', True)} is not supported")

            else:
                x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## GT-prepare
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]

            gt_point_instances, ins_masks_query_batch = [], []
            for i in range(len(gt_instances)): # batch_size
                ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i] # [20000]
                if torch.sum(ins == -1) != 0: 
                    # Use global instance number for each frame
                    ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    ins = F.one_hot(ins)[:, :-1]
                else:
                    ins = F.one_hot(ins)
                    max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    if ins.shape[1] < max_ids:
                        zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                        ins = torch.cat([ins, zero_pad], dim=-1)
                ins = ins.bool().T # [3, 20000]
                gt_point = InstanceData()
                gt_point.p_masks = ins
                gt_point_instances.append(gt_point)
            ## Query
            if self.use_query_memory:
                x = self.query_memory_aggregation(x, sp_xyz)
            if self.use_self_attn:
                query_self = self.muti_scale_self_attn(sp_xyz, x, x, x, sp_xyz)
                x = [x[i] + query_self[i] for i in range(len(x))]
                x = [self.self_attn_relu(x[i]) for i in range(len(x))]

            queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz, frame_i)
            if self.use_mot:
                device = x[0].device
                is_last = frame_i == num_frames - 1 
                if frame_i == 0:
                    track_instances = self._init_query(queries, mot_type=self.mot_type)
                    active_queries_num =  [q.shape[0] for q in track_instances.queries]
                else: 
                    init_track_instances: Instances = track_instances
                    if len(init_track_instances.queries) > 0:
                        active_queries_num = [len(q) for q in init_track_instances.queries]
                    else: #active_queries = 0
                        active_queries_num = [len(q) for q in init_track_instances.queries]  # Set to 0 if queries is empty
                    track_instances = Instances.cat([init_track_instances, self._init_query(queries, mot_type=self.mot_type)])

            ## Decoder
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) 
            if self.use_mot:
                if self.mot_type == 'dq_track' and self.asso_config.get('train_asso_only', True):
                    with frozen_inference(self.decoder):
                        x = self.decoder(x, point_features, queries, super_points, use_temporal_loss=self.use_temporal_loss, inst_dict=self.inst_dict if self.use_temporal_loss else None)
                else:
                    raise NotImplementedError(f"mot_type {self.mot_type} and train_asso_only {self.asso_config.get('train_asso_only', True)} is not supported")
            else:
                x = self.decoder(x, point_features, queries, super_points, use_temporal_loss=self.use_temporal_loss, inst_dict=self.inst_dict if self.use_temporal_loss else None, use_one2many=self.use_one2many) # ! 还是这里？
            if self.use_mot:          
                untracked_tgt_indexes_list = []
                # new_indexes_list = []
                unmatched_track_idxes_list = []
                gt_num_list = []
                gt_obj_list = []
                for batch_idx in range(len(queries)):
                    if frame_i == 0:
                        abs_boxes = x['bboxes'][batch_idx].clone()
                        abs_boxes[:, :3] += sp_xyz[batch_idx][:, :3]
                        track_instances.bboxes[batch_idx][:, :] = abs_boxes
                        # track_instances.bboxes[batch_idx][:, :] = x['bboxes'][batch_idx]
                        track_instances.masks[batch_idx][:, :] = x['masks'][batch_idx]
                        track_instances.sem_preds[batch_idx][:, :] = x['sem_preds'][batch_idx]
                        track_instances.cls_preds[batch_idx][:, :] = x['cls_preds'][batch_idx]
                        if self.mot_type == 'dq_track':
                            if not use_after_features:
                                track_instances.queries[batch_idx][:, :] = x['queries'][batch_idx]
                        else:
                            raise NotImplementedError(f"mot_type {self.mot_type} is not supported")
                    else:
                        abs_boxes = x['bboxes'][batch_idx].clone()
                        abs_boxes[:, :3] += sp_xyz[batch_idx][:, :3] 
                        track_instances.bboxes[batch_idx][active_queries_num[batch_idx]:, :] = abs_boxes
                        # track_instances.bboxes[batch_idx][active_queries_num[batch_idx]:, :] = x['bboxes'][batch_idx]
                        track_instances.masks[batch_idx][active_queries_num[batch_idx]:, :] = x['masks'][batch_idx]
                        track_instances.sem_preds[batch_idx][active_queries_num[batch_idx]:, :] = x['sem_preds'][batch_idx]
                        track_instances.cls_preds[batch_idx][active_queries_num[batch_idx]:, :] = x['cls_preds'][batch_idx]
                        if self.mot_type == 'dq_track':
                            if not use_after_features:
                                track_instances.queries[batch_idx][active_queries_num[batch_idx]:, :] = x['queries'][batch_idx]
                        else:
                            raise NotImplementedError(f"mot_type {self.mot_type} is not supported")

                if self.mot_type == 'dq_track':
                    track_embedding_for_update_list = []
                    if self.asso_config.get('use_aug', False):
                        unmatched_track_embedding_for_update_list = []
                    matched_list = []
                    for batch_idx in range(len(gt_instances)):
                        # valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                        matched_list.append({
                            'track_idx': torch.empty(0, dtype=torch.int64, device=device),
                            'current_obj_idxes': torch.empty(0, dtype=torch.int64, device=device),
                            'gt_idx': torch.empty(0, dtype=torch.int64, device=device),
                            'valid_gt_idx':(gt_instances[batch_idx].labels_3d[:-201] != -1).nonzero(as_tuple=True)[0],
                            'mot_type': self.mot_type,})
                    indices = match_for_indices(self.matcher, x, gt_instances, gt_point_instances, matched_list)

                    loss_asso = torch.tensor(0.0).to(device)
                    if self.use_relative_asso:
                        loss_rel_asso = torch.tensor(0.0).to(device)
                    
                    for batch_idx in range(len(indices)):
                        pred_indices = indices[batch_idx][0]
                        if self.asso_config.get('use_aug', False):
                            all_pred_indices = torch.arange(len(x['queries'][batch_idx])).to(device)
                            unmatched_pred_indices = all_pred_indices
                            unmatched_det_embedding  = x['queries'][batch_idx][unmatched_pred_indices].clone()
                            unmatched_track_embedding_for_update = self.tracklet_trans(unmatched_det_embedding)
                            unmatched_obj_embedding = self.detector_trans(unmatched_det_embedding)
                            unmatched_det_bboxes = x['bboxes'][batch_idx][unmatched_pred_indices]
                            unmatched_det_bboxes[:, :3] += sp_xyz[batch_idx][unmatched_pred_indices][:, :3] 
                            unmatched_pos_embedding = self.box_trans(unmatched_det_bboxes)
                            unmatched_track_embedding_for_update, _ = self.query_inter(unmatched_track_embedding_for_update, unmatched_obj_embedding, unmatched_pos_embedding, unmatched_det_bboxes[:, :3])
                            unmatched_track_embedding_for_update_list.append(unmatched_track_embedding_for_update)


                        gt_indices = indices[batch_idx][1]
                        det_embedding = x['queries'][batch_idx][pred_indices].clone()
                        track_embedding_for_update = self.tracklet_trans(det_embedding)
                        obj_embedding = self.detector_trans(det_embedding)
                        det_bboxes = x['bboxes'][batch_idx][pred_indices]
                        det_bboxes[:, :3] += sp_xyz[batch_idx][pred_indices][:, :3] 
                        pos_embedding = self.box_trans(det_bboxes)
                        track_embedding_for_update, obj_embedding = self.query_inter(track_embedding_for_update, obj_embedding, pos_embedding, det_bboxes[:, :3])
                        valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                        if self.use_relative_asso and valid_track.sum() > 0 and len(x['queries'][batch_idx]) > 0:
                            all_det_embedding = x['queries'][batch_idx].clone()
                            all_track_embedding_for_update = self.tracklet_trans(all_det_embedding)
                            all_obj_embedding = self.detector_trans(all_det_embedding)
                            all_det_bboxes = x['bboxes'][batch_idx].clone()
                            all_det_bboxes[:, :3] += sp_xyz[batch_idx][:, :3] 
                            all_pos_embedding = self.box_trans(all_det_bboxes)
                            all_track_embedding_for_update, all_obj_embedding = self.query_inter(all_track_embedding_for_update, all_obj_embedding, all_pos_embedding, all_det_bboxes[:, :3])
                            
                            valid_track = track_instances.matched_gt_idxes[batch_idx] >= 0
                            all_track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                            all_track_embedding = track_instances.queries[batch_idx][valid_track]
                            if self.use_bbox:
                                all_rel_dist = self.iou_calculator(bbox_pred_to_bbox(all_det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                                all_rel_dist = all_rel_dist.unsqueeze(-1)
                            else:
                                all_det_pos = all_det_bboxes[:, :3]
                                all_rel_dist = (all_det_pos[:,None] - all_track_pos[None])**2
                                all_rel_dist = all_rel_dist.sum(-1, keepdim=True).sqrt()
                            all_geometry_embedding = self.rel_dist_embed(all_rel_dist)
                            all_appear_embedding = all_obj_embedding[:,None] * all_track_embedding[None]
                            all_fused_embedding = all_appear_embedding + all_geometry_embedding
                            all_det2track_heatmap = self.embed_trans2(all_fused_embedding).sum(-1) # [N_det, N_track]
                            target_query_masks = gt_instances[batch_idx].query_masks[:-self.sem_len, :].clone().T.float() # [gt_num, N_det]
                            nonzero_per_row = [
                                torch.nonzero(target_query_masks[i], as_tuple=True)[0].tolist()
                                for i in range(target_query_masks.size(0))
                            ]
                            track_inds = track_instances.matched_gt_idxes[batch_idx][valid_track]
                            target_heatmap = torch.zeros_like(all_det2track_heatmap) # [N_det, N_track]
                            for i in range(target_query_masks.size(0)):
                                cols = torch.nonzero(target_query_masks[i], as_tuple=True)[0]  # e.g. Tensor([2,5,7], device=...)
                                mask_tracks = torch.isin(track_inds, cols)                   # Bool Tensor, shape [num_tracks]
                                track_index = torch.nonzero(mask_tracks, as_tuple=True)[0] 
                                target_heatmap[i, track_index] = 1.0
                            loss_rel_asso += self.heatmap_loss_fn(all_det2track_heatmap, target_heatmap)

                        
                        if valid_track.sum() > 0:
                            track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                            track_embedding = track_instances.queries[batch_idx][valid_track]
                            
                            if self.use_bbox:
                                rel_dist = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                                rel_dist = rel_dist.unsqueeze(-1) # [N_det, N_track, 1]
                            else:
                                det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移,xyz有没有加进来
                                rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                                rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                            geometry_embedding = self.rel_dist_embed(rel_dist) # [N_det, N_track, 1] -> [N_det, N_track, 256]
                            appear_embedding = obj_embedding[:,None] * track_embedding[None] # [N_det, N_track, 256]
                            fused_embedding = appear_embedding + geometry_embedding
                            det2track_mat = self.embed_trans(fused_embedding).sum(-1)

                            
                        else:
                            det2track_mat = None
                        track_embedding_for_update_list.append(track_embedding_for_update)
                        if det2track_mat is not None and valid_track.sum() > 0:
                            gt_per_frame = torch.full((len(det2track_mat),), -1).to(det2track_mat.device)
                            track_inds = track_instances.matched_gt_idxes[batch_idx][valid_track]
                            for _idx, obj_id in enumerate(gt_indices):
                                if obj_id not in track_inds: # 在一维张量 track_inds 中寻找等于 obj_id 的位置，并将该位置索引赋值给 gt_per_frame[_idx]。
                                    continue
                                gt_per_frame[_idx] = (track_inds==obj_id).nonzero(as_tuple=True)[0][0]
                            
                            # filter out new-born object
                            gt_mask = (gt_per_frame >= 0)
                            det2track_mat = det2track_mat[gt_mask] # [det_num, track_num]
                            gt_per_frame = gt_per_frame[gt_mask] # [det_num]
                            # assert gt_per_frame.max() < det2track_mat.shape[1]
                            if len(det2track_mat) > 0:
                                loss_asso += self.asso_config.get('asso_loss_weight', 0.5) * self.loss_asso(det2track_mat, gt_per_frame.long())
                            else:
                                loss_asso += 0

                            if use_after_features:
                                update_type = self.asso_config.get('update_type', 'ema')
                                update_rate = 1 - self.asso_config.get('no_update_rate', 0.0)
                                mask_upd = torch.rand(gt_per_frame.size(0), device=gt_per_frame.device) < update_rate
                                update_index = gt_per_frame[mask_upd]
                                if update_type == 'ema':
                                    track_instances.queries[batch_idx][update_index] = track_instances.queries[batch_idx][update_index] * self.ema_decay_rate + \
                                        (1 - self.ema_decay_rate) * track_embedding_for_update[gt_mask][mask_upd]
                                    track_instances.bboxes[batch_idx][update_index] = track_instances.bboxes[batch_idx][update_index] * self.ema_decay_rate + \
                                        (1 - self.ema_decay_rate) * det_bboxes[gt_mask][mask_upd]
                                elif update_type == 'count':
                                    track_instances.track_age[batch_idx][update_index] += 1
                                    track_age = track_instances.track_age[batch_idx][update_index].unsqueeze(1)
                                    track_instances.queries[batch_idx][update_index] = (track_instances.queries[batch_idx][update_index] * track_age + track_embedding_for_update[gt_mask][mask_upd]) / (track_age + 1)
                                    track_instances.bboxes[batch_idx][update_index] = (track_instances.bboxes[batch_idx][update_index] * track_age + det_bboxes[gt_mask][mask_upd]) / (track_age + 1)
                                else:
                                    raise NotImplementedError(f"Unknown update_type: {self.update_type}")
                        else:
                            pass
                else:
                    raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")

                # track_loss = self.get_loss_track(track_instances, current_dict, mot_type=self.mot_type)
                # update the track_instances
                track_instances = self.update_track_instances(track_instances, gt_instances, indices, unmatched_track_idxes_list, untracked_tgt_indexes_list, gt_num_list, is_last,
                                                               track_embedding_for_update_list if (self.mot_type == 'dq_track' and use_after_features) else None,
                                                               unmatched_track_embedding_for_update_list if self.asso_config.get('use_aug', False) else None)
                

            ## Query projector
            for i in range(len(gt_instances)):
                ins_masks_query = gt_instances[i].query_masks[:-self.sem_len, :]
                ins_masks_query = [ins_masks_query[i].nonzero().flatten()
                        for i in range(ins_masks_query.shape[0])]
                ins_masks_query_batch.append(ins_masks_query)
            if hasattr(self, 'merge_head'): # True
                merge_feat = self.merge_head(x['queries'])
                merge_feat_n_frames.append(merge_feat)
                ins_masks_query_n_frames.append(ins_masks_query_batch)
            
            ## Loss
            if self.use_temporal_loss:
                loss, inst_dict = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode, use_temporal_loss=self.use_temporal_loss)
                inst_dict['inst_pred_masks'] = []
                inst_dict['inst_points'] = []
                inst_dict['correspond_inst_gt'] = []
                inst_dict['bboxes_3d'] = []
                inst_dict['querys'] = []
                for i in range(len(gt_instances)):
                    # 选出保留的id(socre + matched)
                    pred_scores = x['cls_preds'][i].sigmoid()[:, :-1]
                    scores, topk_idx = pred_scores.flatten(0, 1).topk(min(self.test_cfg.topk_insts, pred_scores.shape[0]), sorted=False)
                    matched_idx = inst_dict['indices'][i][0]
                    mask = ~torch.isin(topk_idx, matched_idx)
                    filtered_idx = topk_idx[mask]
                    correspond_inst_gt = torch.cat([inst_dict['indices'][i][1], -1 * torch.ones(filtered_idx.shape[0], device=filtered_idx.device)], dim=0)
                    all_idx = torch.cat([matched_idx, filtered_idx], dim=0)
                    scores = torch.cat([torch.ones(matched_idx.shape[0], device=scores.device), scores[mask]], dim=0)
                    labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(len(all_idx), 1).flatten(0, 1)
                    

                    # nms 进行过滤
                    inst_pred_mask = x['masks'][i].sigmoid() > self.test_cfg.sp_score_thr
                    inst_pred_mask = inst_pred_mask[all_idx]
                    kernel = self.test_cfg.matrix_nms_kernel
                    scores, labels, inst_pred_mask, keep_inds = mask_matrix_nms(
                        inst_pred_mask, labels, scores, kernel=kernel)
                    all_idx = all_idx[keep_inds]
                    correspond_inst_gt = correspond_inst_gt[keep_inds]

                    # 利用点的数量进行过滤
                    mask_pointnum = inst_pred_mask.sum(1) > self.test_cfg.npoint_thr
                    # scores = scores[mask_pointnum]
                    # labels = labels[mask_pointnum]
                    inst_pred_mask = inst_pred_mask[mask_pointnum]
                    all_idx = all_idx[mask_pointnum]
                    correspond_inst_gt = correspond_inst_gt[mask_pointnum]

                    bboxes_3d = x['bboxes'][i][all_idx].detach()
                    bboxes_3d[:, :3] += x['centers'][i][all_idx].detach()
                    inst_dict['bboxes_3d'].append(bboxes_3d)
                    inst_dict['inst_pred_masks'].append(inst_pred_mask.detach())
                    inst_dict['correspond_inst_gt'].append(correspond_inst_gt.detach())
                    inst_dict['querys'].append(x['queries'][i][all_idx].detach()) # [N, 96]

                    # 每个物体对应的点

                    inst_dict['inst_points'].append([])
                    current_points = batch_inputs_dict['points'][i][frame_i]
                    for obj_id in range(inst_pred_mask.shape[0]):
                        inst_dict['inst_points'][i].append(current_points[inst_pred_mask[obj_id]])
                self.inst_dict = inst_dict
                

            else:
                if self.use_mot:
                    if self.mot_type == 'dq_track':
                        if self.asso_config.get('train_asso_only', True):
                            # loss = loss_asso
                            loss = {'loss_asso': loss_asso}
                            if self.use_relative_asso:
                                loss.update({'loss_rel_asso': loss_rel_asso})
                        else:
                            loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode) # + loss_asso
                            loss.update({'loss_asso': loss_asso})
                    else:
                        raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")
                else:
                    loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)
                    if self.use_one2many:
                        loss_one2many = self.criterion_one2many(x['one2many_outputs'], gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode, use_one2many=self.use_one2many)
                        for key, value in loss_one2many.items():
                            loss_one2many[key] = value * self.one2many_loss_weight
                        loss.update(loss_one2many)
                    if self.reweight_dict is not None:
                        for key in loss.keys():
                            if key in self.reweight_dict:
                                loss[key] = loss[key] * self.reweight_dict[key]
            # print(loss)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        ## Query contrast 计算同一个物体的对比损失
        if hasattr(self, 'merge_criterion'): # True
            merge_feat_n_frames = [[frame[i] for frame in merge_feat_n_frames]
                 for i in range(len(merge_feat_n_frames[0]))]
            ins_masks_query_n_frames = [[frame[i] for frame in ins_masks_query_n_frames]
                 for i in range(len(ins_masks_query_n_frames[0]))]
            loss = self.merge_criterion(merge_feat_n_frames, ins_masks_query_n_frames)
            losses.update(loss)
        return losses
    def query_memory_aggregation(self, x, sp_xyz):
        if self.query_memory is not None and self.pos_memory is not None:
            query_x = self.muti_scale_query(sp_xyz, x, self.query_memory, self.query_memory, self.pos_memory)
            x = [x[i] + query_x[i] for i in range(len(x))]
            x = [self.query_memory_relu(x[i]) for i in range(len(x))]
        detach_query = [x[i].clone() for i in range(len(x))]
        detach_pos = [sp_xyz[i].clone() for i in range(len(sp_xyz))]
        self.query_memory = detach_query
        self.pos_memory = detach_pos
        return x
    def query_memory_aggregation_predict(self, x, sp_xyz):
        if self.query_memory is not None and self.pos_memory is not None:
            query_x = self.muti_scale_query(sp_xyz, x, self.query_memory, self.query_memory, self.pos_memory)
            x = [x[i] + query_x[i] for i in range(len(x))]
            x = [self.query_memory_relu(x[i]) for i in range(len(x))]
        with torch.no_grad():
            detach_query = [x[i].clone().detach() for i in range(len(x))]
            detach_pos = [sp_xyz[i].clone().detach() for i in range(len(sp_xyz))]
            self.query_memory = detach_query
            self.pos_memory = detach_pos
        return x
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        if self.use_query_memory:
            self.reset_query_memory()
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
        if self.use_temporal_loss:
            self.inst_dict = None
        if self.use_mot:
            self.current_max_track_id = 0
        for frame_i in range(num_frames):
            ## Backbone + SP merge (optional)  -> features, point_features, all_xyz_w, sp_xyz
            if self.merge_sp_masks:
                # Integrated path: merge SPs and extract features once
                x, point_features, all_xyz_w, sp_xyz = self.merge_superpixels_extract_feat(
                    batch_inputs_dict, batch_data_samples, frame_i)
            else:
                x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## Query
            if self.use_query_memory:
                x = self.query_memory_aggregation_predict(x, sp_xyz)
            if self.use_self_attn:
                query_self = self.muti_scale_self_attn(sp_xyz, x, x, x, sp_xyz)
                x = [x[i] + query_self[i] for i in range(len(x))]
                x = [self.self_attn_relu(x[i]) for i in range(len(x))]
            if self.use_mot:
                device = x[0].device
                is_last = frame_i == num_frames - 1 

                if frame_i == 0:
                    track_instances = self._init_query_test(x, mot_type=self.mot_type, mode='test')
                    active_queries_num =  [q.shape[0] for q in track_instances.queries]
                else:
                    init_track_instances: Instances = track_instances
                    if len(init_track_instances.queries) > 0:
                        active_queries_num = [len(q) for q in init_track_instances.queries]
                    else: #active_queries = 0
                        active_queries_num = [len(q) for q in init_track_instances.queries]  # Set to 0 if queries is empty
            ## Decoder 
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) # ([20000], [20000, 1])
            x = self.decoder(x, point_features, x, super_points) # [N_segment, 96] [20000, 99] [N_segment, 96] ([20000], [20000, 1])
            ## Post-processing
            pred_pts_seg, mapping = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            results.append(pred_pts_seg[0])
            if self.use_mot:  
                batch_idx = 0
                if self.asso_config.get('debug', False):
                    gt_instances = [s.gt_instances_3d for s in batch_data_samples] 
                    gt_point_instances, matched_list = [], []
                    for i in range(len(gt_instances)): # batch_size
                        ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i] # [20000]
                        if torch.sum(ins == -1) != 0: 
                            # Use global instance number for each frame
                            ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                            ins = F.one_hot(ins)[:, :-1]
                        else:
                            ins = F.one_hot(ins)
                            max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                            if ins.shape[1] < max_ids:
                                zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                                ins = torch.cat([ins, zero_pad], dim=-1)
                        ins = ins.bool().T # [3, 20000]
                        gt_point = InstanceData()
                        gt_point.p_masks = ins
                        gt_point_instances.append(gt_point)
                    gt_instances = self._select_queries_predict(device, gt_instances, frame_i) 
                    matched_list.append({
                        'track_idx': torch.empty(0, dtype=torch.int64, device=device),
                        'current_obj_idxes': torch.empty(0, dtype=torch.int64, device=device),
                        'gt_idx': torch.empty(0, dtype=torch.int64, device=device),
                        'valid_gt_idx':(gt_instances[batch_idx].labels_3d[:-201] != -1).nonzero(as_tuple=True)[0],
                        'mot_type': self.mot_type,})
                    indices = match_for_indices(self.matcher, x, gt_instances, gt_point_instances, matched_list)  
                if self.mot_type == 'dq_track':

                    valid_track = track_instances.valid_track[0].clone()
                    valid_det = mapping[0]
                    
                    det_embedding = x['queries'][batch_idx][valid_det]
                    det_category = x['sem_preds'][batch_idx][valid_det].argmax(1)
                    track_embedding_for_update = self.tracklet_trans(det_embedding)
                    obj_embedding = self.detector_trans(det_embedding)
                    det_bboxes = x['bboxes'][batch_idx][valid_det]
                    det_bboxes[:, :3] += sp_xyz[batch_idx][valid_det]
                    pos_embedding = self.box_trans(det_bboxes)
                    track_embedding_for_update, obj_embedding = self.query_inter(track_embedding_for_update, obj_embedding, pos_embedding, det_bboxes[:, :3])

                    if valid_track.sum() > 0:
                        track_pos = track_instances.bboxes[batch_idx][valid_track][:, :3]
                        track_embedding = track_instances.queries[batch_idx][valid_track]
                        track_category = track_instances.category[batch_idx][valid_track]
                        # det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移
                        # rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                        # rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                        if self.use_bbox:
                            rel_dist = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(track_instances.bboxes[batch_idx][valid_track]))
                            rel_dist = rel_dist.unsqueeze(-1) # [N_det, N_track, 1]
                        else:
                            det_pos = det_bboxes[:, :3] # TODO 查看是否需要偏移
                            rel_dist = (det_pos[:,None] - track_pos[None])**2  # [N_det, N_track, 3]
                            rel_dist = rel_dist.sum(-1, keepdim=True).sqrt() # [N_det, N_track, 1]
                        geometry_embedding = self.rel_dist_embed(rel_dist) # [N_det, N_track, 1] -> [N_det, N_track, 256]
                        appear_embedding = obj_embedding[:,None] * track_embedding[None] # [N_det, N_track, 256]
                        fused_embedding = appear_embedding + geometry_embedding
                        det2track_mat = self.embed_trans(fused_embedding).sum(-1)
                        det2track_mat = det2track_mat.softmax(1)
                        if self.use_relative_asso:
                            det2track_heatmap = self.embed_trans2(fused_embedding).sum(-1).sigmoid() # [N_det, N_track]
                            det2track_mat = det2track_mat * det2track_heatmap

                        # Scheme A: compute det2buffer_mat with buffer snapshots (align with det2track path)
                        det2buffer_mat = None
                        if 'online_merger' in locals() and getattr(online_merger, 'use_buffer', False):
                            buf_ids, buf_queries, buf_bboxes, buf_cats = online_merger.get_buffer_snapshots(device=det_embedding.device)
                            if buf_queries is not None and buf_queries.shape[0] > 0:
                                if self.use_bbox:
                                    rel_dist_buf = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(buf_bboxes))
                                    rel_dist_buf = rel_dist_buf.unsqueeze(-1)
                                else:
                                    det_pos = det_bboxes[:, :3]
                                    buf_pos = buf_bboxes[:, :3]
                                    rel_dist_buf = (det_pos[:, None] - buf_pos[None]) ** 2
                                    rel_dist_buf = rel_dist_buf.sum(-1, keepdim=True).sqrt()
                                geometry_embedding_buf = self.rel_dist_embed(rel_dist_buf)
                                appear_embedding_buf = obj_embedding[:, None] * buf_queries[None]
                                fused_embedding_buf = appear_embedding_buf + geometry_embedding_buf
                                det2buffer_mat = self.embed_trans(fused_embedding_buf).sum(-1)
                                det2buffer_mat = det2buffer_mat.softmax(1)
                                if self.use_relative_asso:
                                    det2buffer_heatmap = self.embed_trans2(fused_embedding_buf).sum(-1).sigmoid()
                                    det2buffer_mat = det2buffer_mat * det2buffer_heatmap

                    else:
                        assert len(valid_det) > 0
                        abs_boxes = x['bboxes'][batch_idx][valid_det]
                        abs_boxes[:, :3] += sp_xyz[batch_idx][valid_det][:, :3] # 这里是将sp_xyz的偏移量加到abs_boxes上
                        track_instances.bboxes[batch_idx][:valid_det.shape[0], :] = abs_boxes
                        track_instances.valid_track[batch_idx][:valid_det.shape[0]] = True
                        track_instances.long_track[batch_idx][:valid_det.shape[0]] = False
                        track_instances.active[batch_idx][:valid_det.shape[0]] = True
                        track_instances.disappear_time[batch_idx][:valid_det.shape[0]] = 0
                        track_instances.obj_idxes[batch_idx][:valid_det.shape[0]] = torch.arange(valid_det.shape[0], device=valid_det.device)
                        track_instances.track_age[batch_idx][:valid_det.shape[0]] = 0
                        track_instances.queries[batch_idx][:valid_det.shape[0], :] = track_embedding_for_update
                        track_instances.obj_labels[batch_idx][:valid_det.shape[0]] = 0
                        track_instances.scores[batch_idx][:valid_det.shape[0]] =  F.softmax(x['cls_preds'][batch_idx][valid_det], dim=-1)[:, :-1].flatten(0, 1)
                        track_instances.global_track_id[batch_idx][:valid_det.shape[0]] = torch.arange(valid_det.shape[0], device=valid_det.device)
                        track_instances.category[batch_idx][:valid_det.shape[0]] = det_category
                        self.current_max_track_id = valid_det.shape[0]
                        det2track_mat = None
                        # Scheme A: compute det2buffer_mat when there are buffered tracks
                        det2buffer_mat = None
                        if 'online_merger' in locals() and getattr(online_merger, 'use_buffer', False):
                            buf_ids, buf_queries, buf_bboxes, buf_cats = online_merger.get_buffer_snapshots(device=det_embedding.device)
                            if buf_queries is not None and buf_queries.shape[0] > 0:
                                if self.use_bbox:
                                    rel_dist_buf = self.iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(buf_bboxes))
                                    rel_dist_buf = rel_dist_buf.unsqueeze(-1)
                                else:
                                    det_pos = det_bboxes[:, :3]
                                    buf_pos = buf_bboxes[:, :3]
                                    rel_dist_buf = (det_pos[:, None] - buf_pos[None]) ** 2
                                    rel_dist_buf = rel_dist_buf.sum(-1, keepdim=True).sqrt()
                                geometry_embedding_buf = self.rel_dist_embed(rel_dist_buf)
                                appear_embedding_buf = obj_embedding[:, None] * buf_queries[None]
                                fused_embedding_buf = appear_embedding_buf + geometry_embedding_buf
                                det2buffer_mat = self.embed_trans(fused_embedding_buf).sum(-1)
                                det2buffer_mat = det2buffer_mat.softmax(1)
                                if self.use_relative_asso:
                                    det2buffer_heatmap = self.embed_trans2(fused_embedding_buf).sum(-1).sigmoid()
                                    det2buffer_mat = det2buffer_mat * det2buffer_heatmap
                        det2buffer_mat = None

                else:
                    raise NotImplementedError(f"Unknown mot_type: {self.mot_type}")

            ## Query projector, semantic and geometric information
            if hasattr(self, 'merge_head'): # True
                query_feats = self.merge_head(x['queries'][0])
                query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]]) # 取出对应的query
                sem_preds = x['cls_preds'][0]
                sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]]) # 前景背景
                sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
                if self.use_bbox:
                    bbox_preds = x['bboxes'][0] # [N, 6]
                    bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])
            ## Online merging
            if self.test_cfg.merge_type == 'learnable_online': # True

                if frame_i == 0:
                    if self.use_mot and self.mot_type == 'dq_track':
                        online_merger = DQ_Track_OnlineMerge(
                            self.test_cfg.inscat_topk_insts,
                            self.use_bbox,
                            self.asso_config.get('update_type', 'count'),
                            self.asso_config.get('use_buffer', False))
                    else:
                        online_merger = OnlineMerge(self.test_cfg.inscat_topk_insts, self.use_bbox)
                if self.use_mot and self.mot_type == 'dq_track':
                    mv_mask, mv_labels, mv_scores, self.current_max_track_id= online_merger.merge( # , mv_queries
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None,
                        det2track_mat, det2buffer_mat, track_instances, track_embedding_for_update, self.current_max_track_id, det_category)
                else:
                    mv_mask, mv_labels, mv_scores, mv_bboxes = online_merger.merge( # , mv_queries
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None,)
                if self.use_mot and self.mot_type == 'dq_track':
                    pass
                    # track_instances = self.update_track_instances_predict(track_instances, mapping, indices, is_last)
                # Empty cache. Only offline merging requires the whole list.
                torch.cuda.empty_cache()
                if frame_i == num_frames - 1:
                    online_merger.clean() # Ignore panoptic segmentation
        
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge_mat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                [res[1] for res in query_feats_list],
                [res[1] for res in sem_preds_list],
                [res[1] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_cat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline panoptic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        # if self.use_bbox and not self.use_mot:
        #     batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
        # Not mapping to reconstructed point clouds, return directly for visualization
        if not self.map_to_rec_pcd: # False
            merged_result = PointData(
                pts_semantic_mask=[mv_sem.cpu().numpy()],
                pts_instance_mask=[mv_mask.cpu().numpy()],
                instance_labels=mv_labels.cpu().numpy(),
                instance_scores=mv_scores.cpu().numpy())
            batch_data_samples[0].pred_pts_seg = merged_result
            return batch_data_samples
        
        ## Mapping to reconstructed point clouds for evaluation
        mv_xyz = batch_inputs_dict['points'][0][:, :, :3].reshape(-1, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])[:, :3]
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float() # [239388, 3]
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float() # [680000, 3]
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()

        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())

        # Ensemble the predictions with mesh segments (eval_ann_info['segment_ids']) 
        if 'segment_ids' in batch_data_samples[0].eval_ann_info: # True
            merged_result = self.segment_smooth(merged_result, mv_xyz.device,
                batch_data_samples[0].eval_ann_info['segment_ids'])
        batch_data_samples[0].pred_pts_seg = merged_result

        return batch_data_samples
    def merge_superpixels(self, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.8):

        merge_masks = []

        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]

            merged_mask = sp_pts_mask.clone()

            merged_groups = []

            for inst_id in pts_instance_mask.unique():
                if inst_id == -1: 
                    continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() #
                valid_sps = []
                for sp_id in sp_ids_in_instance:
                    sp_mask = (sp_pts_mask == sp_id) 
                
                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 
            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            
            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            
            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            
            merge_masks.append(final_mask)
            # merged_sp_masks.append(merged_sp_mask)
        
        return merge_masks #, merged_sp_masks


    def merge_superpixels_predict(self, batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.99):
        merge_masks = []
        merged_sp_masks = []
        merged_groups_list = []
        class_names = [
            'cabinet', 'bed', 'chair', 'sofa', 'table',
            'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
            'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
            'bathtub', 'otherfurniture'
        ]
        id2class = dict(enumerate(class_names))
        for batch_idx in range(len(current_sp_pts_mask)):
            labels_3d = batch_data_samples[batch_idx].gt_instances_3d.labels_3d[frame_i]
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]
            #sp_masks = current_sp_masks[batch_idx]  # [gt_num+201, num_queries]
            
            merged_mask = sp_pts_mask.clone()
            
            merged_groups = [] 
            
            for inst_id in pts_instance_mask.unique():
                
                if inst_id == -1: # -1号是背景
                    continue
                category = labels_3d[inst_id]
                assert category != -1, f"Invalid category for instance {inst_id}."
                # if category == 6 or category == 15: # 6: table, 7: chair
                #     continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() 
                valid_sps = []
                for sp_id in sp_ids_in_instance:
                    # if sp_id == -1: 
                    #     continue
                    sp_mask = (sp_pts_mask == sp_id) 
                
                    # if (sp_mask & inst_mask).sum() == sp_mask.sum():
                    #     valid_sps.append(sp_id.item())
                    
                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    # print(id2class[category.item()],': valid_sps:', valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0]

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]

            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1
            
            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id
            if len(merged_groups) > 0:
                merged_sp_masks.append(build_pairwise_mask(merged_groups, compact=False, max_value=sp_pts_mask.max()))
                merge_masks.append(final_mask)
                merged_groups_list.append(merged_groups)
            else:
                merged_sp_masks.append(([], []))
                merge_masks.append(final_mask)
                merged_groups_list.append([])
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with torch.no_grad():
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates # [20000, 3]

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features) 

        # forward of backbone and neck 
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            x = self.backbone(field.sparse(),
                            partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                            memory=self.memory if hasattr(self,'memory') else None)
        else:
            x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)] # [20000, 99] 
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone() # [20000] 
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) # [20000] 
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        # super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w) # ([20000], [20000, 1])
        # x_final = self.decoder(features, point_features, features, super_points)
        x_detach = [features[i] for i in range(len(features))]
        pred_bboxes = []
        pred_cls_list = []
        new_queries = []
        queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask)) 
        for i in range(len(queries)):
            norm_query = self.decoder.out_norm(queries[i])
            reg_final = self.decoder.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
            reg_cls = self.decoder.out_cls(norm_query) # [N_segments, 256] -> [N_segments, 1]
            reg_cls = reg_cls.softmax(1)
            
            reg_distance = torch.exp(reg_final[:, 3:6])
            pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)

            pred_cls_list.append(reg_cls)
            pred_bboxes.append(pred_bbox)
            new_queries.append(norm_query)
        x_detach = [new_queries[i] for i in range(len(new_queries))]
        valid_sps_list = []
        for batch_idx in range(len(pred_bboxes)):
            # invalid_index = pred_cls_list[batch_idx][:, 0] < 0.5
            labels = pred_cls_list[batch_idx].argmax(dim=1) 
            bg = pred_cls_list[batch_idx].shape[1] - 1
            labels_mask = ((labels[:, None] == labels)  & (labels[:, None] != bg) & (labels != bg)[:, None]  ) # 
            det_bboxes = pred_bboxes[batch_idx].clone() # [N_det, 6]
            det_bboxes[:, :3] += sp_xyz_list[batch_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx], x_detach[batch_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1) # [N_det, N_det, 1]
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist) # [N_det, N_det, 1] -> [N_det, N_det, 256]
            merge_appear_embedding = obj_embedding1[:,None] * obj_embedding2[None] # [N_det, N_det, 256]
            # V1
            # merge_fused_embedding = merge_appear_embedding + merge_geometry_embedding
            # V2
            # fused = torch.cat([merge_appear_embedding, merge_geometry_embedding], dim=-1)
            # merge_fused_embedding = self.fuse_linear(fused)  # nn.Linear(2*D, D)
            # V3
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            m = merge_det_mat.sigmoid() # [N_det, N_det]

            # m = merge_det_mat.softmax(dim=1)    
            m_sym = (m + m.t()) / 2             
            if len(merged_sp_masks[batch_idx][0]) > 0:
                merge_det_heatmap = merged_sp_masks[batch_idx][0].to(det_bboxes.device).float()
            else:
                merge_det_heatmap = torch.zeros_like(m_sym).to(det_bboxes.device).float()
            # cluster = cluster_with_threshold(m_sym, 0.7)
            # cluster = cluster_with_threshold(self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='giou'), 0.7)
            # print('===================================================================')
            tmp_heatmap = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            error = torch.abs(tmp_heatmap - merge_det_heatmap).sum() / len(det_bboxes) 
            tmp_heatmap_mask = (tmp_heatmap > 0.5).float()
            
            error_mask = torch.abs(tmp_heatmap_mask - merge_det_heatmap).sum() / len(det_bboxes) 
            # print(f"Error: {error.item()}, Error mask{error_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            best_threshold, best_error_mask = find_optimal_threshold(tmp_heatmap, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold: {best_threshold}, Best error mask: {best_error_mask.item()}")
            if len(det_bboxes) in self.acc_dict:
                self.acc_dict[len(det_bboxes)].append(best_threshold)
            else:
                self.acc_dict[len(det_bboxes)] = [best_threshold]
            
            # error2 = torch.abs(m_sym - merge_det_heatmap).sum() / len(det_bboxes)
            # tmp_heatmap_mask2 = (m_sym > 0.5).float()
            # error2_mask = torch.abs(tmp_heatmap_mask2 - merge_det_heatmap).sum() / len(det_bboxes)
            # print(f"Error2: {error2.item()}, Error mask{error2_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            # best_threshold2, best_error_mask2 = find_optimal_threshold(m_sym, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold2: {best_threshold2}, Best error mask2: {best_error_mask2.item()}")

            # iou_mask = tmp_heatmap < 0.2
            # m_sym[iou_mask] = 0
            # error3 = torch.abs(m_sym - merge_det_heatmap).sum() / len(det_bboxes)
            # tmp_heatmap_mask3 = (m_sym > 0.5).float()
            # error3_mask = torch.abs(tmp_heatmap_mask3 - merge_det_heatmap).sum() / len(det_bboxes)
            # print(f"Error3: {error3.item()}, Error mask{error3_mask.item()}, Num Det Bboxes: {len(det_bboxes)}")
            # best_threshold3, best_error_mask3 = find_optimal_threshold(m_sym, merge_det_heatmap, det_bboxes)
            # print(f"Best threshold3: {best_threshold3}, Best error mask3: {best_error_mask3.item()}")
            # cluster = cluster_with_threshold(tmp_heatmap, best_threshold)
            # print('===================================================================')
            # if len(det_bboxes) > 100:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.65)
            # elif len(det_bboxes) > 50:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.65)
            # else:
            #     cluster = cluster_with_threshold(tmp_heatmap, 0.63)
            # cluster = cluster_with_threshold(tmp_heatmap, 0.63)
            iou_map = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes), mode='iou')
            # iou_map[invalid_index] = 0
            # iou_map[:, invalid_index] = 0
            # cluster_ori = cluster_with_threshold(m_sym, 0.95)
            # m_sym = torch.where(merge_rel_dist[:, :, 0] > 0.2, m_sym, 0)
            cluster_gt = cluster_complete_link(merge_det_heatmap, 0.96)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_ori))
            # cluster = cluster_complete_link(m_sym*labels_mask.float(), 0.9)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster))

            cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.3)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.4)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # cluster_iou = cluster_complete_link(iou_map*labels_mask.float(), 0.5)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_iou))
            # thresh_dict = {0: 0.4, 1: 0.5, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.4, 8: 0.5, 9: 0.5, 
            #                10: 0.3, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.3, 15: 0.4, 16: 0.5, 17: 0.3, 18: 0.5,}

            # clusters_per_cls = cluster_with_per_class_threshold(
            #     iou_map, labels_mask, labels,
            #     thresh_per_class=thresh_dict,
            #     base_thresh=1.0
            # )
            # print(evaluate_clustering_pairwise(cluster_gt, clusters_per_cls))
            cluster_both = cluster_complete_link(iou_map*m_sym*labels_mask.float(), 0.5)
            # print(evaluate_clustering_pairwise(cluster_gt, cluster_both))
            # print('==============================')
            
            valid_sps_list.append(cluster_both)
            # valid_sps_list.append(cluster_iou)
            # valid_sps_list.append(clusters_per_cls)
            # valid_sps_list.append(clusters_per_cls)
            

        new_merge_masks = []
        new_merged_sp_masks = []
        
        # 对每个batch进行处理
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            
            merged_groups = []
            for valid_sps in valid_sps_list[batch_idx]:
           
                if len(valid_sps) > 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]
            
            new_id = 0
            id_mapping = {}

            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1
                
            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1

            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            new_merged_sp_masks.append(merged_groups)
            new_merge_masks.append(final_mask)
        
        return new_merge_masks

    def merge_superpixels_train(self, batch_inputs_dict, batch_data_samples, frame_i, current_sp_pts_mask, current_pt_instance_mask, overlap_threshold=0.7):
        merge_masks = []
        merged_sp_masks = []
        
        for batch_idx in range(len(current_sp_pts_mask)):
            sp_pts_mask = current_sp_pts_mask[batch_idx]
            pts_instance_mask = current_pt_instance_mask[batch_idx]
            merged_mask = sp_pts_mask.clone()
            merged_groups = []
            for inst_id in pts_instance_mask.unique():
                if inst_id == -1:
                    continue
                inst_mask = (pts_instance_mask == inst_id)
                if not inst_mask.any():
                    continue
                sp_ids_in_instance = sp_pts_mask[inst_mask].unique() 
                valid_sps = []
                for sp_id in sp_ids_in_instance:

                    sp_mask = (sp_pts_mask == sp_id)

                    overlap_ratio = (sp_mask & inst_mask).sum().float() / sp_mask.sum().float()
                    if overlap_ratio > overlap_threshold:
                        valid_sps.append(sp_id.item())
                        
                if len(valid_sps) >= 1:
                    merged_groups.append(valid_sps)
                    for sp_id in valid_sps:
                        merged_mask[sp_pts_mask == sp_id] = valid_sps[0] 

            all_original_ids = sp_pts_mask.unique().tolist()
            merged_ids = set()
            for group in merged_groups:
                merged_ids.update(group)
            unmerged_ids = [id for id in all_original_ids if id not in merged_ids]

            new_id = 0
            id_mapping = {}
            
            for group in merged_groups:
                for old_id in group:
                    id_mapping[old_id] = new_id
                new_id += 1

            for old_id in unmerged_ids:
                id_mapping[old_id] = new_id
                new_id += 1

            final_mask = merged_mask.clone()
            for old_id, new_id in id_mapping.items():
                final_mask[merged_mask == old_id] = new_id

            merged_sp_masks.append(build_pairwise_mask(merged_groups, compact=True))
            merge_masks.append(final_mask)
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with frozen_inference(self.img_backbone):
                img_features = []
                for img_paths in batch_inputs_dict['img_paths']:
                    img_features.append(self.img_backbone(img_paths[frame_i])[0])
            img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
            for img_meta in img_metas:
                img_meta['depth2img'] = img_meta['depth2img'][frame_i]
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict: # False
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates # [20000, 3]

        coordinates, features = ME.utils.batch_sparse_collate( # [20000, 4] [20000, 3]
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck 
        if isinstance(self, ScanNet200MixFormer3D_FF_Online):
            with frozen_inference(self.backbone), frozen_inference(self.memory):
                x = self.backbone(field.sparse(),
                                partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                                memory=self.memory if hasattr(self,'memory') else None)
        else:
            with frozen_inference(self.backbone), frozen_inference(self.memory):
                x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None) # [13141, 96]
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field) # [20000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features # [20000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        if self.use_temporal_loss and self.inst_dict is not None:
            best_obj_ids_list = []
        for batch_idx, (data_sample, tmp_xyz) in enumerate(zip(batch_data_samples, all_xyz)):
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i].clone()
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points)) 
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks) # [20000]
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True) # [N_segment, 96], [20000, 1]
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])

        
        x_detach = [features[i].detach() for i in range(len(features))]
        pred_bboxes = []
        new_queries = []
        with frozen_inference(self.decoder):
            queries = self.decoder._get_queries(x_detach, len(current_sp_pts_mask)) 
            for i in range(len(queries)):
                norm_query = self.decoder.out_norm(queries[i])
                reg_final = self.decoder.out_reg(norm_query) # [N_segments, 256] -> [N_segments, 6]
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
                pred_bboxes.append(pred_bbox)
                new_queries.append(norm_query)
        x_detach = [new_queries[i].detach() for i in range(len(new_queries))]
        pred_bboxes_detach = [pred_bboxes[i].detach() for i in range(len(pred_bboxes))]
        sp_xyz_list_detach = [sp_xyz_list[i].detach() for i in range(len(sp_xyz_list))]
        # loss = torch.tensor(0.0).to(x_detach[0].device)
        loss = self.get_mask_heatmap_loss(x_detach, pred_bboxes_detach, sp_xyz_list_detach, merged_sp_masks)
                    
        return loss
    def get_mask_heatmap_loss(self, x_detach, pred_bboxes_detach, sp_xyz_list,  merged_sp_masks):
        loss = 0
        for batch_idx in range(len(pred_bboxes_detach)):

            valid_det_idx = merged_sp_masks[batch_idx][1]
            if len(valid_det_idx) == 0:
                continue
            det_bboxes = pred_bboxes_detach[batch_idx][valid_det_idx].clone() # [N_det, 6]
            det_bboxes[:, :3] += sp_xyz_list[batch_idx][valid_det_idx][:, :3]
            pos_embedding = self.merge_box_trans(det_bboxes)
            obj_embedding1, obj_embedding2 = self.merge_query_inter(x_detach[batch_idx][valid_det_idx], x_detach[batch_idx][valid_det_idx], pos_embedding)
            merge_rel_dist = self.merge_iou_calculator(bbox_pred_to_bbox(det_bboxes), bbox_pred_to_bbox(det_bboxes))
            merge_rel_dist = merge_rel_dist.unsqueeze(-1) # [N_det, N_det, 1]
            merge_geometry_embedding = self.merge_dist_embed(merge_rel_dist) # [N_det, N_det, 1] -> [N_det, N_det, 256]
            merge_appear_embedding = obj_embedding1[:,None] * obj_embedding2[None] # [N_det, N_det, 256]
            # V1
            # merge_fused_embedding = merge_appear_embedding + merge_geometry_embedding
            # V2
            # fused = torch.cat([merge_appear_embedding, merge_geometry_embedding], dim=-1)
            # merge_fused_embedding = self.fuse_linear(fused)  # nn.Linear(2*D, D)
            # V3
            merge_fused_embedding = self.merge_fusion(merge_appear_embedding, merge_geometry_embedding)
            merge_det_mat = self.merge_embed_trans(merge_fused_embedding).sum(-1)
            merge_det_mat = merge_det_mat.sigmoid() # [N_det, N_det]
            merge_det_heatmap = merged_sp_masks[batch_idx][0].to(det_bboxes.device).float()
            gamma = 2.0  
            alpha = 0.25  
            # loss += F.binary_cross_entropy(merge_det_mat, merge_det_heatmap, reduction='mean')
            bce_loss = F.binary_cross_entropy(merge_det_mat, merge_det_heatmap, reduction='none')
            # pt = torch.exp(-bce_loss)  

            # focal_loss = alpha * (1 - pt) ** gamma * bce_loss

            # 最终损失为加权后的Focal Loss
            loss += bce_loss.sum() / max(merge_det_heatmap.eq(1).float().sum().item(), 1)
            # loss += sigmoid_focal_loss(
            #     pred=merge_det_mat.view(-1),
            #     tgt=merge_det_heatmap.view(-1),
            #     alpha=0.25,
            #     gamma=2.0,
            #     reduction='mean'
            # )
        return loss
    def segment_smooth(self, results, device, segment_ids):
        unique_ids = np.unique(segment_ids)
        new_segment_ids = np.zeros_like(segment_ids)
        for i, ids in enumerate(unique_ids):
            new_segment_ids[segment_ids == ids] = i
        segment_ids = new_segment_ids
        segment_ids = torch.from_numpy(segment_ids).to(device)
        sem_mask = torch.from_numpy(results.pts_semantic_mask[0]).to(device)
        ins_mask = torch.from_numpy(results.pts_instance_mask[0]).to(device)
        sem_mask = scatter_mean(F.one_hot(sem_mask).float(), segment_ids, dim=0)
        sem_mask = sem_mask.argmax(dim=1)[segment_ids]
        ins_mask = scatter_mean(ins_mask.float(), segment_ids, dim=1)
        ins_mask = (ins_mask > 0.5)[:, segment_ids]
        results.pts_semantic_mask[0] = sem_mask.cpu().numpy()
        results.pts_instance_mask[0] = ins_mask.cpu().numpy()
        return results
    
    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr) # dict [20000] 0.3
        sem_res = self.predict_by_feat_semantic(out, superpoints) # [20000]

        sem_map2 = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes) # [20000]
        inst_res2 = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)

        pts_semantic_mask = [sem_res, sem_map2]
        pts_instance_mask = [inst_res[0].bool(), inst_res2[0].bool()]
        instance_labels = [inst_res[1], inst_res2[1]]
        instance_scores = [inst_res[2], inst_res2[2]]
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                instance_queries=instance_queries)], mapping
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        mapping = torch.arange(len(out['cls_preds'][0])).to(superpoints.device)
        cls_preds = out['cls_preds'][0] # [N_seg, 2]
        pred_masks = out['masks'][0] # [N_seg, 20000]
        queries = out['queries'][0] # [N_seg, 256]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1] # [N_pred, 1] 
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1) # N_pred
        topk_num = min(self.test_cfg.topk_insts, scores.shape[0] * scores.shape[1]) # 取最大前20
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False) 
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        queries = queries[topk_idx] # [topk_num, 256]
        mapping = mapping[topk_idx] # [topk_num]

        if self.test_cfg.get('obj_normalization', None): 
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6) # [topk_num]
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel # 'linear'
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel) 
            mapping = mapping[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold # [n_preds] 
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]

        return mask_pred, labels, scores, queries, mapping
    
    def predict_by_feat_panoptic(self, sem_map, mask_pred, labels, scores):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask

        return sem_map, inst_map


@MODELS.register_module()
class ScanNet200MixFormer3D_FF_Online(ScanNet200MixFormer3D_Online):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None,

                 use_query_memory=False,
                 use_self_attn=False,
                 use_noise=False,
                 noise_p=0.05,
                 noise_k=10,
                 use_temporal_loss=False,
                 use_decouple=False,
                 use_mot=False,
                 mot_type='motr',
                 train_asso_only=False,
                 matcher=None,
                 use_aug=False,
                 asso_loss_weight=0.5,
                 use_refine=False,
                 asso_config=None,
                 use_one2many=False,
                 criterion_one2many=None,
                 use_3d_refine=False,
                 reweight_dict=None,
                 use_relative_asso=False,
                 merge_sp_masks = False,
                 replace_bn_with_ln=False,
                 debug_mode=False
                 ):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = MODELS.build(img_backbone)
        self.backbone = MODELS.build(backbone)
        if memory is not None:
            self.memory = MODELS.build(memory)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.pool = MODELS.build(pool)
        self.decoder = MODELS.build(decoder)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(criterion)
        self.decoder_online = decoder['temporal_attn']
        self.use_bbox = decoder['bbox_flag']
        self.sem_len = decoder['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.map_to_rec_pcd=True
        self.use_query_memory = use_query_memory
        if self.use_query_memory:
            self.muti_scale_query = MultiScaleQuery()
            self.query_memory = None
            self.pos_memory = None
            self.query_memory_relu = nn.ReLU()
            self.muti_scale_query.init_weights()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.muti_scale_self_attn = MultiScaleQuery()
            self.self_attn_relu = nn.ReLU()
            self.muti_scale_self_attn.init_weights()
        self.use_noise = use_noise
        if self.use_noise:
            self.noise_p = noise_p
            self.noise_k = noise_k
        self.use_temporal_loss = use_temporal_loss
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_mask_memory = None
            self.before_sp_xyz = None
            self.before_query_ids = None
        self.use_decouple = use_decouple
        if self.use_temporal_loss:
            self.before_query_memory = None
            self.before_query_boxes = None
        self.use_mot = use_mot
        if self.use_mot:
            self.mot_type = mot_type
            if mot_type == 'dq_track':
                self.use_relative_asso = use_relative_asso
                if self.use_relative_asso:
                    self.embed_trans2 = nn.Linear(256, 1)
                    self.heatmap_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                self.use_refine = use_refine
                # self.asso_loss_weight = asso_loss_weight
                self.iou_calculator = AxisAlignedBboxOverlaps3D()
                self.matcher = TASK_UTILS.build(matcher)
                self.tracklet_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.detector_trans = DQ_FFN(d_model=256, d_ffn=256, dropout=0)
                self.box_trans = nn.Sequential(
                    nn.Linear(6, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                # query_trans = {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                query_trans = asso_config['query_trans'] if asso_config is not None else {'with_att': True, 'with_pos': True, 'min_channels': 256, 'drop_rate': 0.0}
                self.query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
                # self.update_type = asso_config['update_type'] if asso_config is not None else 'ema'
                self.asso_config = EasyDict(asso_config)
                
                self.rel_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
                self.embed_trans = nn.Linear(256, 1)
                loss_asso = {'use_sigmoid': False, 'loss_weight': 1.0}
                from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
                self.loss_asso = CrossEntropyLoss(**loss_asso)
                self.ema_decay_rate = 0.5
                # self.train_asso_only = train_asso_only
            else:
                raise NotImplementedError(f"mot_type {mot_type} is not supported")

        self.use_one2many = use_one2many
        if self.use_one2many:
            self.one2many_loss_weight = 0.5
            self.criterion_one2many = MODELS.build(criterion_one2many[0])
        self.reweight_dict = reweight_dict
        self.merge_sp_masks = merge_sp_masks
        if self.merge_sp_masks:
            self.acc_dict = {}
            self.merge_box_trans = nn.Sequential(
                nn.Linear(6, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_dist_embed = nn.Sequential(
                    nn.Linear(1, 256),
                    DQ_FFN(d_model=256, d_ffn=256, dropout=0))
            self.merge_embed_trans = nn.Linear(256, 1)
            self.merge_heatmap_loss = nn.BCEWithLogitsLoss(reduction='mean')
            self.merge_iou_calculator = AxisAlignedBboxOverlaps3D()
            self.fuse_linear = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256)
            )
            self.merge_fusion = MergeFusion(256)
            query_trans = {'with_att': True, 'with_pos': False, 'min_channels': 256, 'drop_rate': 0.0}
            self.merge_query_inter = QueryInteractionX(in_channels=256, mid_channels=256, **query_trans)
        self._prev_param_snapshot = None
        if replace_bn_with_ln:
            replace_bn(self)
        self.debug_mode = debug_mode
        self.init_weights()
        
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(960, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))
    
    def init_weights(self):
        if hasattr(self, 'memory'):
            self.memory.init_weights()
        if hasattr(self, 'img_backbone'):
            self.img_backbone.init_weights()
    def reset_query_memory(self):
        """Reset the detector.
        """
        if self.use_query_memory:
            self.query_memory = None
            self.pos_memory = None
    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
        """Extract features from sparse tensor.
        """
        # extract image features
        with torch.no_grad():
            img_features = []
            for img_paths in batch_inputs_dict['img_paths']:
                img_features.append(self.img_backbone(img_paths[frame_i])[0])
        
        # TODO check
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
        for img_meta in img_metas:
            img_meta['depth2img'] = img_meta['depth2img'][frame_i]
    
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            # pdb.set_trace()
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_features, img_metas=img_metas, img_shape=img_metas[0]['img_shape']),
                          memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck: # False
            x = self.neck(x)
        x = x.slice(field) # [45611, 96] -> [80000, 96]
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]  # [B, N, 3+D]
        x = x.features # [80000, 96]

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=True)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list # [N_segment, 96], [20000, 99], [20000, 1], [N_segment, 3]

    def _f(self, x, img_features, img_metas, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0) # [N, 960]
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        projected_features = self.conv(projected_features)
        return projected_features + x



class MultiScaleQuery(nn.Module):
    """Scale-adaptive Self Attention with variable-length tokens per batch."""
    def __init__(self, embed_dims=96, num_heads=8, dropout=0.1, init_cfg=None):

        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

        self.embed_dims = embed_dims
        self.num_heads = num_heads

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self, pos1_xyz, query_feat, key_feat, val_feat, pos2_xyz=None):
        """
        pos1_xyz: list of [num_queries_i, 3]
        query_feat: list of [num_queries_i, embed_dims]
        key_feat: list of [num_keys_i, embed_dims]
        val_feat: list of [num_values_i, embed_dims]
        pos2_xyz: list of [num_keys_i, 3] or None
        """
        batch_size = len(query_feat)
        
        Q_max = max([q.shape[0] for q in query_feat])
        K_max = max([k.shape[0] for k in key_feat])
        V_max = max([v.shape[0] for v in val_feat])

        padded_query = torch.zeros(batch_size, Q_max, self.embed_dims, device=query_feat[0].device)
        padded_key = torch.zeros(batch_size, K_max, self.embed_dims, device=key_feat[0].device)
        padded_val = torch.zeros(batch_size, V_max, self.embed_dims, device=val_feat[0].device)
        
        query_mask = torch.zeros(batch_size, Q_max, dtype=torch.bool, device=query_feat[0].device)
        key_mask = torch.zeros(batch_size, K_max, dtype=torch.bool, device=key_feat[0].device)
        
        for i in range(batch_size):
            q_len = query_feat[i].shape[0]
            k_len = key_feat[i].shape[0]
            v_len = val_feat[i].shape[0]
            
            padded_query[i, :q_len, :] = query_feat[i]
            padded_key[i, :k_len, :] = key_feat[i]
            padded_val[i, :v_len, :] = val_feat[i]
            
            query_mask[i, :q_len] = 1
            key_mask[i, :k_len] = 1

        padded_pos1_xyz = torch.zeros(batch_size, Q_max, 3, device=pos1_xyz[0].device)
        pos1_mask_spatial = torch.zeros(batch_size, Q_max, dtype=torch.bool, device=pos1_xyz[0].device)
        for i in range(batch_size):
            q_len = pos1_xyz[i].shape[0]
            padded_pos1_xyz[i, :q_len, :] = pos1_xyz[i]
            pos1_mask_spatial[i, :q_len] = 1

        if pos2_xyz is not None:
            padded_pos2_xyz = torch.zeros(batch_size, K_max, 3, device=pos2_xyz[0].device)
            pos2_mask_spatial = torch.zeros(batch_size, K_max, dtype=torch.bool, device=pos2_xyz[0].device)
            for i in range(batch_size):
                k_len = pos2_xyz[i].shape[0]
                padded_pos2_xyz[i, :k_len, :] = pos2_xyz[i]
                pos2_mask_spatial[i, :k_len] = 1
        else:
            padded_pos2_xyz = None
            pos2_mask_spatial = None

        if pos2_xyz is None:
            dist = self.calc_bbox_dists(padded_pos1_xyz, pos1_mask_spatial)  # [B, Q_max, Q_max]
        else:
            dist = self.calc_bbox_dists2(padded_pos1_xyz, pos1_mask_spatial, 
                                        padded_pos2_xyz, pos2_mask_spatial)  # [B, Q_max, K_max]
        
        if torch.isnan(dist).any():
            raise ValueError("NaN detected in dist")

        # 8. 生成 tau
        tau = self.gen_tau(padded_query)  # [B, Q_max, num_heads]
        tau = tau.clamp(max=10.0)  # Clamp to prevent extremely large values
        tau = tau.permute(0, 2, 1)  # [B, num_heads, Q_max]
        
        if torch.isnan(tau).any():
            raise ValueError("NaN detected in tau")

        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, num_heads, Q_max, K_max]

        if pos2_xyz is None:
            key_padding_mask = ~pos1_mask_spatial  # [B, Q_max], True 表示要屏蔽
        else:
            key_padding_mask = ~pos2_mask_spatial  # [B, K_max], True 表示要屏蔽

        attn_mask_combined = attn_mask.reshape(batch_size * self.num_heads, Q_max, -1)  # [B*num_heads, Q_max, K_max]

        assert attn_mask_combined.shape == (batch_size * self.num_heads, Q_max, attn_mask_combined.shape[-1]), "Attention mask shape mismatch"

        attn_output, attn_weights = self.attention(
            query=padded_query, 
            key=padded_key, 
            value=padded_val, 
            attn_mask=attn_mask_combined,
            key_padding_mask=key_padding_mask
        )
        
        if torch.isnan(attn_output).any():
            raise ValueError("NaN detected in attn_output")

        attn_output = attn_output * pos1_mask_spatial.unsqueeze(-1)  # [B, Q_max, D]

        queries = []
        for i in range(batch_size):
            queries.append(attn_output[i, :pos1_mask_spatial[i].sum(), :])
        
        return queries

    @torch.no_grad()
    def calc_bbox_dists(self, pos1_xyz, pos1_mask):
        pos1_xyz_exp1 = pos1_xyz.unsqueeze(2)  # [B, Q_max, 1, 3]
        pos1_xyz_exp2 = pos1_xyz.unsqueeze(1)  # [B, 1, Q_max, 3]

        dist = torch.norm(pos1_xyz_exp1 - pos1_xyz_exp2, dim=-1)  # [B, Q_max, Q_max]
        dist = -dist  # [B, Q_max, Q_max]

        return dist

    @torch.no_grad()
    def calc_bbox_dists2(self, pos1_xyz, pos1_mask, pos2_xyz, pos2_mask):
        pos1_xyz_exp = pos1_xyz.unsqueeze(2)  # [B, Q_max, 1, 3]
        pos2_xyz_exp = pos2_xyz.unsqueeze(1)  # [B, 1, K_max, 3]

        dist = torch.norm(pos1_xyz_exp - pos2_xyz_exp, dim=-1)  # [B, Q_max, K_max]

        dist = -dist  # [B, Q_max, K_max]

        return dist

def bbox_pred_to_bbox(bbox_pred):
    """Transform predicted bbox parameters to bbox.
    """
    if bbox_pred.shape[0] == 0:
        return bbox_pred
    bbox = bbox_pred

    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)
