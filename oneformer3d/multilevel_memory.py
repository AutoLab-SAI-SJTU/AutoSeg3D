# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, constant_init
import pdb


@MODELS.register_module()
class MultilevelMemory(BaseModule):
    def __init__(self, in_channels=[64, 128, 256, 512], scale=2.5, queue=-1, vmp_layer=(0,1,2,3)):
        super().__init__()
        self.scale = scale
        self.queue = queue
        self.vmp_layer = list(vmp_layer)
        self.conv_d1 = nn.ModuleList()
        self.conv_d3 = nn.ModuleList()
        self.conv_convert = nn.ModuleList()
        for i, C in enumerate(in_channels):
            if i in self.vmp_layer:
                self.conv_d1.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_d3.append(nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=C,
                        out_channels=C,
                        kernel_size=3,
                        stride=1,
                        dilation=3,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C),
                    ME.MinkowskiReLU()))
                self.conv_convert.append(nn.Sequential(
                    ME.MinkowskiConvolutionTranspose(
                        in_channels=3*C,
                        out_channels=C,
                        kernel_size=1,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=3),
                    ME.MinkowskiBatchNorm(C)))
            else:
                self.conv_d1.append(nn.Identity())
                self.conv_d3.append(nn.Identity())
                self.conv_convert.append(nn.Identity())
        self.relu = ME.MinkowskiReLU()
        self.accumulated_feats = None
        if self.queue > 0:
            self.pruning = ME.MinkowskiPruning()
            self.time_st = 0
            self.accumulated_ts = None
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiConvolutionTranspose):
                constant_init(m.kernel, 0)

            if isinstance(m, ME.MinkowskiBatchNorm):
                constant_init(m.bn.weight, 1)
                constant_init(m.bn.bias, 0)
    
    def reset(self):
        self.accumulated_feats = None
        if self.queue > 0:
            self.time_st = 0
            self.accumulated_ts = None
    
    def global_avg_pool_and_cat(self, feat1, feat2, feat3):
        coords1 = feat1.decomposed_coordinates
        feats1 = feat1.decomposed_features
        coords2 = feat2.decomposed_coordinates
        feats2 = feat2.decomposed_features
        coords3 = feat3.decomposed_coordinates
        feats3 = feat3.decomposed_features
        for i in range(len(coords3)): # batch size
            # shape 1 N
            global_avg_feats3 = torch.mean(feats3[i], dim=0).unsqueeze(0).repeat(coords3[i].shape[0],1)
            feats1[i] = torch.cat([feats1[i], feats2[i]], dim=1)     
            feats1[i] = torch.cat([feats1[i], global_avg_feats3], dim=1)      
        coords_sp, feats_sp = ME.utils.sparse_collate(coords1, feats1)
        feat_new = ME.SparseTensor(
            coordinates=coords_sp,
            features=feats_sp,
            tensor_stride=feat1.tensor_stride,
            coordinate_manager=feat1.coordinate_manager
        )
        return feat_new
    
    def accumulate(self, accumulated_feat, accumulated_t, current_feat, index):
        """Accumulate features for a single stage.

        Args:
            accumulated_feat (ME.SparseTensor)
            current_feat (ME.SparseTensor)

        Returns:
            ME.SparseTensor: refined accumulated features
            ME.SparseTensor: current features after accumulation
        """
        if index in self.vmp_layer: # [0,1,2,3]
            # VMP
            tensor_stride = current_feat.tensor_stride
            accumulated_feat = ME.TensorField(
                features=torch.cat([current_feat.features, accumulated_feat.features], dim=0),
                coordinates=torch.cat([current_feat.coordinates, accumulated_feat.coordinates], dim=0),
                quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
            ).sparse() # 对于相同坐标的特征执行MAX_POOL
            accumulated_feat = ME.SparseTensor(
                coordinates=accumulated_feat.coordinates,
                features=accumulated_feat.features,
                tensor_stride=tensor_stride,
                coordinate_manager=accumulated_feat.coordinate_manager
            )

            # queued cache
            if self.queue > 0: # -1
                current_t = ME.SparseTensor(
                    coordinates=current_feat.coordinates,
                    features=(torch.ones(current_feat.coordinates.shape[0],1)*self.time_st).to(current_feat.device),
                    tensor_stride=tensor_stride,
                    coordinate_manager=current_feat.coordinate_manager
                )
                accumulated_t = ME.TensorField(
                    features=torch.cat([current_t.features, accumulated_t.features], dim=0),
                    coordinates=torch.cat([current_t.coordinates, accumulated_t.coordinates], dim=0),
                    quantization_mode=ME.SparseTensorQuantizationMode.MAX_POOL
                ).sparse()
                accumulated_t = ME.SparseTensor(
                    coordinates=accumulated_feat.coordinates,
                    features=accumulated_t.features_at_coordinates(accumulated_feat.coordinates.float()),
                    tensor_stride=tensor_stride,
                    coordinate_manager=accumulated_feat.coordinate_manager
                )
                if (accumulated_t.features.max() - accumulated_t.features.min()) >= (self.queue - 1):
                    mask = (accumulated_t.features != accumulated_t.features.min()).squeeze(1)
                    accumulated_t = self.pruning(accumulated_t, mask)
                    accumulated_feat = self.pruning(accumulated_feat, mask)

            # Select neighbor region for current frame
            accumulated_coords = accumulated_feat.decomposed_coordinates
            current_coords = current_feat.decomposed_coordinates
            accumulated_coords_select_list=[]
            zero_batch_feature_list=[]
            for i in range(len(current_coords)): # scale size
                accumulated_coords_batch = accumulated_coords[i]
                current_coords_batch = current_coords[i]
                current_coords_batch_max, _ = torch.max(current_coords_batch,dim=0)
                current_coords_batch_min, _ = torch.min(current_coords_batch,dim=0)
                current_box_size = current_coords_batch_max - current_coords_batch_min # 计算当前帧特征坐标的最大值和最小值，用于定义当前帧的包围盒
                current_box_add = ((self.scale-1)/2) * current_box_size # 根据缩放因子 self.scale 计算需要添加的边界，以扩大当前包围盒的范围 放大到当前的2.5倍
                margin_positive = accumulated_coords_batch-current_coords_batch_max-current_box_add
                margin_negative = accumulated_coords_batch-current_coords_batch_min+current_box_add
                in_criterion = torch.mul(margin_positive,margin_negative) # 通过逐元素相乘，确定哪些累积坐标在当前包围盒内。in_criterion 的值为负表示坐标在包围盒内（因为累积坐标在包围盒范围内时，margin_positive 和 margin_negative 的乘积小于等于零）。
                zero = torch.zeros_like(in_criterion)
                one = torch.ones_like(in_criterion)
                in_criterion = torch.where(in_criterion<=0,one,zero) # 将满足条件的坐标标记为 1，否则为 0。
                mask = in_criterion[:,0]*in_criterion[:,1]*in_criterion[:,2] # 生成一个布尔掩码，表示哪些累积坐标位于当前包围盒内
                mask = mask.type(torch.bool)
                mask = mask.reshape(mask.shape[0],1)
                accumulated_coords_batch_select = torch.masked_select(accumulated_coords_batch,mask) # 根据掩码选择符合条件的累积坐标 [21258]
                accumulated_coords_batch_select = accumulated_coords_batch_select.reshape(-1,3) # [7086, 3]
                zero_batch_feature = torch.zeros_like(accumulated_coords_batch_select) # 创建一个与选中坐标相同形状的零特征张量，用于后续的稀疏张量拼接
                accumulated_coords_select_list.append(accumulated_coords_batch_select)
                zero_batch_feature_list.append(zero_batch_feature)
            accumulated_coords_select_coords, _ = ME.utils.sparse_collate(accumulated_coords_select_list, zero_batch_feature_list) # 将所有选择的坐标和对应的零特征张量拼接成一个统一的坐标张量
            current_feat_new = ME.SparseTensor( # 创建一个新的稀疏张量，包含选中的累积坐标和对应的特征。这些特征是从 accumulated_feat 中提取的，与选择的坐标相对应
                coordinates=accumulated_coords_select_coords,
                features=accumulated_feat.features_at_coordinates(accumulated_coords_select_coords.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager # new shorcut
            )
            # ? 对选择的累积特征进行不同尺度的卷积处理 但是使用的是一样的卷积层？？
            branch1 = self.conv_d1[index](current_feat_new)
            branch3 = self.conv_d3[index](current_feat_new)
            branch  = self.global_avg_pool_and_cat(branch1, branch3, current_feat_new)
            branch = self.conv_convert[index](branch) # 维度转换
            current_feat_new = branch + current_feat # new shorcut 对于每一个共有坐标，branch 和 current_feat 中对应的特征向量会进行逐元素相加 对于每一个独有坐标，结果张量中只包含来自对应张量的特征向量
            current_feat_new = self.relu(current_feat_new)
            current_feat = ME.SparseTensor(
                coordinates=current_feat.coordinates,
                features=current_feat_new.features_at_coordinates(current_feat.coordinates.float()),
                tensor_stride=tensor_stride,
                coordinate_manager=current_feat.coordinate_manager
            )
        return accumulated_feat, accumulated_t, current_feat
    
    def forward(self, x):
        if self.accumulated_feats is None:
            accumulated_feats = x # ? 这里是想保存修改前还是修改后
            for i in range(len(x)): # batch size # ? 这里的处理很奇怪，为什么没有多尺度之类的 
                if i in self.vmp_layer: # 分别处理下采样1 2 3 4 次的特征
                    branch1 = self.conv_d1[i](x[i])
                    branch3 = self.conv_d3[i](x[i])
                    branch  = self.global_avg_pool_and_cat(branch1, branch3, x[i]) # 对于不同感受野的特征进行聚合
                    branch = self.conv_convert[i](branch)
                    x[i] = branch + x[i]
                    x[i] = self.relu(x[i])
            self.accumulated_feats = accumulated_feats
            if self.queue > 0: # -1
                self.accumulated_ts = [ME.SparseTensor(
                    coordinates=x[i].coordinates,
                    features=(torch.ones(x[i].coordinates.shape[0],1)*self.time_st).to(x[i].device),
                    tensor_stride=x[i].tensor_stride,
                    coordinate_manager=x[i].coordinate_manager
                ) for i in range(len(x))]
                self.time_st += 1
            return x
        else:
            if self.queue > 0: # -1
                tuple_feats = [self.accumulate(self.accumulated_feats[i], self.accumulated_ts[i], x[i], i)
                     for i in range(len(x))]
                self.accumulated_ts = [tuple_feats[i][1] for i in range(len(x))]
                self.time_st += 1
            else:
                tuple_feats = [self.accumulate(self.accumulated_feats[i], None, x[i], i) for i in range(len(x))] # batch size [accumulated_feat, accumulated_t, current_feat]
            self.accumulated_feats = [tuple_feats[i][0] for i in range(len(x))]
            return [tuple_feats[i][2] for i in range(len(x))]
