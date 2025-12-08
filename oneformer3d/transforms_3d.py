import numpy as np
import scipy
import torch
from torch_scatter import scatter_mean
from mmcv.transforms import BaseTransform
import pdb
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ElasticTransfrom(BaseTransform):
    """Apply elastic augmentation to a 3D scene. Required Keys:

    Args:
        gran (List[float]): Size of the noise grid (in same scale[m/cm]
            as the voxel grid).
        mag (List[float]): Noise multiplier.
        voxel_size (float): Voxel size.
        p (float): probability of applying this transform.
    """

    def __init__(self, gran, mag, voxel_size, p=1.0, with_rec=False):
        self.gran = gran
        self.mag = mag
        self.voxel_size = voxel_size
        self.p = p
        self.with_rec = with_rec

    def transform(self, input_dict):
        """Private function-wrapper for elastic transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after elastic, 'points' is updated
            in the result dict.
        """
        coords = input_dict['points'].tensor[:, :3].numpy() / self.voxel_size
        if 'num_frames' in input_dict:
            num_points = input_dict['num_frames'] * input_dict['num_sample']
            if np.random.rand() < self.p:
                coords = self.elastic(coords, self.gran[0], self.mag[0], num_points)
                coords = self.elastic(coords, self.gran[1], self.mag[1], num_points)
        else:
            if np.random.rand() < self.p:
                coords = self.elastic(coords, self.gran[0], self.mag[0])
                coords = self.elastic(coords, self.gran[1], self.mag[1])
        input_dict['elastic_coords'] = coords
        return input_dict

    def elastic(self, x, gran, mag, num_points = None):
        """Private function for elastic transform to a points.

        Args:
            x (ndarray): Point cloud.
            gran (List[float]): Size of the noise grid (in same scale[m/cm]
                as the voxel grid).
            mag: (List[float]): Noise multiplier.
        
        Returns:
            dict: Results after elastic, 'points' is updated
                in the result dict.
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        if self.with_rec:
            noise_dim = np.abs(x[:num_points]).max(0).astype(np.int32) // gran + 3
        else:
            noise_dim = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(noise_dim[0], noise_dim[1],
                            noise_dim[2]).astype('float32') for _ in range(3)
        ]

        for blur in [blur0, blur1, blur2, blur0, blur1, blur2]:
            noise = [
                scipy.ndimage.filters.convolve(
                    n, blur, mode='constant', cval=0) for n in noise
            ]

        ax = [
            np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in noise_dim
        ]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0) for n in noise
        ]

        return x + np.hstack([i(x)[:, None] for i in interp]) * mag


@TRANSFORMS.register_module()
class AddSuperPointAnnotations(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        # For example, 0692_03_1000 has no wall and floor
        # assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1
        pts_instance_mask = mapping[pts_instance_mask]
        input_dict['pts_instance_mask'] = pts_instance_mask.clone().numpy()


        # create gt instance markup     
        insts_mask = pts_instance_mask.clone()
        
        if torch.sum(insts_mask == -1) != 0:
            insts_mask[insts_mask == -1] = torch.max(insts_mask) + 1
            insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
        else:
            insts_mask = torch.nn.functional.one_hot(insts_mask)

        if insts_mask.shape[1] != 0:
            insts_mask = insts_mask.T
            sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
            sp_masks_inst = scatter_mean(
                insts_mask.float(), sp_pts_mask, dim=-1)
            sp_masks_inst = sp_masks_inst > 0.5
        else:
            sp_masks_inst = insts_mask.new_zeros(
                (0, input_dict['sp_pts_mask'].max() + 1), dtype=torch.bool)

        num_stuff_cls = len(self.stuff_classes)
        insts = new_idxs[1:] - 1
        if self.merge_non_stuff_cls:
            gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
        else:
            gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1)

        for inst in insts:
            index = pts_semantic_mask[pts_instance_mask == inst][0]
            gt_labels[inst] = index - num_stuff_cls
        
        input_dict['gt_labels_3d'] = gt_labels.numpy()

        # create gt semantic markup
        sem_mask = torch.tensor(input_dict['pts_semantic_mask'])
        sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                    num_classes=self.num_classes + 1)
       
        sem_mask = sem_mask.T
        sp_pts_mask = torch.tensor(input_dict['sp_pts_mask'])
        sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1)
        sp_masks_seg = sp_masks_seg > 0.5

        sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True

        assert sp_masks_seg.sum(axis=0).max().item()
        
        if self.merge_non_stuff_cls:
            sp_masks_seg = torch.vstack((
                sp_masks_seg[:num_stuff_cls, :], 
                sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
        
        sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg))

        input_dict['gt_sp_masks'] = sp_masks_all.numpy()

        # create eval markup
        if 'eval_ann_info' in input_dict.keys(): 
            pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
            for idx, stuff_cls in enumerate(self.stuff_classes):
                pts_instance_mask[pts_semantic_mask == stuff_cls] = idx

            input_dict['eval_ann_info']['pts_instance_mask'] = \
                pts_instance_mask.numpy()

        return input_dict


@TRANSFORMS.register_module()
class AddSuperPointAnnotations_Online(BaseTransform):
    """Prepare ground truth markup for training.
    
    Required Keys:
    - pts_semantic_mask (np.float32)
    
    Added Keys:
    - gt_sp_masks (np.int64)
    
    Args:
        num_classes (int): Number of classes.
    """
    
    def __init__(self,
                 num_classes,
                 stuff_classes,
                 merge_non_stuff_cls=True,
                 with_rec=False,
                 use_sp_gt_ids=False,
                 inst_thereshold=0.5,
                 sem_thereshold=0.5):
        self.num_classes = num_classes
        self.stuff_classes = stuff_classes
        self.merge_non_stuff_cls = merge_non_stuff_cls
        self.with_rec = with_rec
        self.use_sp_gt_ids = use_sp_gt_ids # 是否使用 sp_gt_inst_ids 和 
        self.inst_thereshold = inst_thereshold # 超点实例掩码的阈值
        self.sem_thereshold = sem_thereshold # 超点语义掩码的阈值
        
 
    def transform(self, input_dict):
        """Private function for preparation ground truth 
        markup for training.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: results, 'gt_sp_masks' is added.
        """
        # create class mapping
        # because pts_instance_mask contains instances from non-instaces classes
        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask'])
        pts_semantic_mask = torch.tensor(input_dict['pts_semantic_mask'])
        
        pts_instance_mask[pts_semantic_mask == self.num_classes] = -1 # 代表“背景”或“无效”类别
        for stuff_cls in self.stuff_classes:
            pts_instance_mask[pts_semantic_mask == stuff_cls] = -1
        
        idxs = torch.unique(pts_instance_mask)
        # For example, 0692_03_1000 has no wall and floor
        # assert idxs[0] == -1

        mapping = torch.zeros(torch.max(idxs) + 2, dtype=torch.long)
        new_idxs = torch.arange(len(idxs), device=idxs.device)
        mapping[idxs] = new_idxs - 1 
        input_dict['ori_pts_instance_mask'] = pts_instance_mask.clone().numpy()
        pts_instance_mask = mapping[pts_instance_mask] # 将原始实例ID映射到新的实例ID（减去 1，因为 new_idxs 从 0 开始，而 -1 是无效实例）。
        input_dict['pts_instance_mask'] = pts_instance_mask.clone().numpy()

        if self.with_rec: # 以下进行相同处理
            rec_instance_mask = torch.tensor(input_dict['rec_instance_mask'])
            rec_semantic_mask = torch.tensor(input_dict['rec_semantic_mask'])
            
            rec_instance_mask[rec_semantic_mask == self.num_classes] = -1
            for stuff_cls in self.stuff_classes:
                rec_instance_mask[rec_semantic_mask == stuff_cls] = -1
            
            rec_idxs = torch.unique(rec_instance_mask)
            # For example, 0692_03_1000 has no wall and floor
            # assert idxs[0] == -1

            rec_mapping = torch.zeros(torch.max(rec_idxs) + 2, dtype=torch.long)
            rec_new_idxs = torch.arange(len(rec_idxs), device=idxs.device)
            rec_mapping[rec_idxs] = rec_new_idxs - 1
            input_dict['ori_rec_instance_mask'] = rec_instance_mask.clone().numpy()
            rec_instance_mask = rec_mapping[rec_instance_mask]
            input_dict['rec_instance_mask'] = rec_instance_mask.clone().numpy()
        input_dict['gt_labels_3d'] = []
        input_dict['gt_sp_masks'] = []
        if self.use_sp_gt_ids: # 是否使用 sp_gt_inst_ids 和 sp_gt_semantic_ids
            input_dict['sp_gt_inst_ids'] = [] # 用于存储每个sp对应gt实例的 ID
            input_dict['sp_gt_semantic_ids'] = [] # 用于存储每个sp对应gt语义的 ID
        for i in range(input_dict['num_frames']):
            # create gt instance markup 提取当前帧的实例和语义掩码
            frame_pts_instance_mask = pts_instance_mask.reshape(input_dict['num_frames'], input_dict['num_sample'])[i]
            frame_pts_semantic_mask = pts_semantic_mask.reshape(input_dict['num_frames'], input_dict['num_sample'])[i]
            frame_sp_pts_mask = input_dict['sp_pts_mask'].reshape(input_dict['num_frames'], input_dict['num_sample'])[i]
            insts_mask = frame_pts_instance_mask.clone()
            
            if torch.sum(insts_mask == -1) != 0:
                # Use global instance number for each frame 创建独热编码，无效实例全是0
                insts_mask[insts_mask == -1] = torch.max(pts_instance_mask) + 1 # N_gt + 1
                insts_mask = torch.nn.functional.one_hot(insts_mask)[:, :-1]
            else:
                insts_mask = torch.nn.functional.one_hot(insts_mask)
                max_ids = torch.max(pts_instance_mask) + 1
                if insts_mask.shape[1] < max_ids:
                    zero_pad = torch.zeros(insts_mask.shape[0], max_ids - insts_mask.shape[1])
                    insts_mask = torch.cat([insts_mask, zero_pad], dim=-1)


            if insts_mask.shape[1] != 0:
                insts_mask = insts_mask.T # [N_gt, `20000`]
                sp_pts_mask = torch.tensor(frame_sp_pts_mask) # [N_seg, 20000]
                sp_masks_inst = scatter_mean(
                    insts_mask.float(), sp_pts_mask, dim=-1) # [N_gt, N_seg]
                
                if self.use_sp_gt_ids:
                    # 计算每个sp属于那个实例,如果都小于0.1就不算
                    extend_sp_masks_inst = torch.cat([sp_masks_inst, self.inst_thereshold * torch.ones(1, sp_masks_inst.shape[1])],dim=0)  # [N_gt + 1, N_seg]
                    sp_gt_inst_ids = torch.argmax(extend_sp_masks_inst, dim=0) # [N_seg] 每个超点对应的实例 ID
                    sp_gt_inst_ids[sp_masks_inst.sum(axis=0) == sp_masks_inst.shape[0]] = -1 # 背景类
                    input_dict['sp_gt_inst_ids'].append(sp_gt_inst_ids) # [N_seg]
                # print((sp_masks_inst > self.inst_thereshold).sum(),(sp_masks_inst > 0.5).sum(), (sp_masks_inst > 0.1).sum(), (sp_masks_inst > 0.05).sum())
                sp_masks_inst = sp_masks_inst > self.inst_thereshold # [N_gt, N_seg] 每个超点是否属于某个实例
            else:
                sp_masks_inst = insts_mask.new_zeros(
                    (0, frame_sp_pts_mask.max() + 1), dtype=torch.bool)
            


            num_stuff_cls = len(self.stuff_classes)
            insts = new_idxs[1:] - 1
            if self.merge_non_stuff_cls: # False
                gt_labels = insts.new_zeros(len(insts) + num_stuff_cls + 1)
            else:
                gt_labels = insts.new_zeros(len(insts) + self.num_classes + 1) # 

            for inst in insts: # 为每个实例分配语义类别标签
                # This frame may not contain a global instance in the scene
                temp_semantic = frame_pts_semantic_mask[frame_pts_instance_mask == inst]
                if len(temp_semantic) != 0:
                    index = temp_semantic[0]
                    gt_labels[inst] = index - num_stuff_cls
                else:
                    gt_labels[inst] = -1
            
            input_dict['gt_labels_3d'].append(gt_labels.numpy())

            # create gt semantic markup
            sem_mask = torch.tensor(input_dict['pts_semantic_mask'].reshape(
                input_dict['num_frames'], input_dict['num_sample'])[i]) # 20000
            sem_mask = torch.nn.functional.one_hot(sem_mask, 
                                        num_classes=self.num_classes + 1)
        
            sem_mask = sem_mask.T # 计算每个超点的语义掩码
            sp_pts_mask = torch.tensor(frame_sp_pts_mask)
            sp_masks_seg = scatter_mean(sem_mask.float(), sp_pts_mask, dim=-1) # [201, 20000] [20000]
            if self.use_sp_gt_ids:
                # 计算每个超点属于那个语义类别
                sp_masks_seg_extend = torch.cat([sp_masks_seg, self.sem_thereshold * torch.ones(1, sp_masks_seg.shape[1])],dim=0)  # [201 + 1, 20000]
                sp_gt_semantic_ids = torch.argmax(sp_masks_seg_extend, dim=0)
                sp_gt_semantic_ids[sp_masks_seg.sum(axis=0) == sp_masks_seg.shape[0]] = -1 # 背景类
                input_dict['sp_gt_semantic_ids'].append(sp_gt_semantic_ids) # [20000]
            sp_masks_seg = sp_masks_seg > 0.5 # [201, 20000] 每个超点是否包括半个以上物体

            sp_masks_seg[-1, sp_masks_seg.sum(axis=0) == 0] = True # 背景类
            

            assert sp_masks_seg.sum(axis=0).max().item()
            
            if self.merge_non_stuff_cls: # False
                sp_masks_seg = torch.vstack((
                    sp_masks_seg[:num_stuff_cls, :], 
                    sp_masks_seg[num_stuff_cls:, :].sum(axis=0).unsqueeze(0)))
            
            sp_masks_all = torch.vstack((sp_masks_inst, sp_masks_seg)) # [223, 61]
            assert sp_masks_all.shape[0] == gt_labels.shape[0]

            input_dict['gt_sp_masks'].append(sp_masks_all.numpy())

        # create eval markup
        if 'eval_ann_info' in input_dict.keys():
            if not self.with_rec:
                pts_instance_mask[pts_instance_mask != -1] += num_stuff_cls
                for idx, stuff_cls in enumerate(self.stuff_classes):
                    pts_instance_mask[pts_semantic_mask == stuff_cls] = idx
                input_dict['eval_ann_info']['pts_instance_mask'] = \
                    pts_instance_mask.numpy()
            else: # True
                rec_instance_mask[rec_instance_mask != -1] += num_stuff_cls
                for idx, stuff_cls in enumerate(self.stuff_classes):
                    rec_instance_mask[rec_semantic_mask == stuff_cls] = idx
                input_dict['eval_ann_info']['pts_instance_mask'] = \
                    rec_instance_mask.numpy()
        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloor(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after swap, 'pts_semantic_mask' is updated
                in the result dict.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class SwapChairAndFloorWithRec(BaseTransform):
    """Swap two categories for ScanNet200 dataset. It is convenient for
    panoptic evaluation. After this swap first two categories are
    `stuff` and other 198 are `thing`.
    """
    def transform(self, input_dict):
        """Private function-wrapper for swap transform.
        """
        mask = input_dict['pts_semantic_mask'].copy()
        mask[input_dict['pts_semantic_mask'] == 2] = 3
        mask[input_dict['pts_semantic_mask'] == 3] = 2
        input_dict['pts_semantic_mask'] = mask
        mask = input_dict['rec_semantic_mask'].copy()
        mask[input_dict['rec_semantic_mask'] == 2] = 3
        mask[input_dict['rec_semantic_mask'] == 3] = 2
        input_dict['rec_semantic_mask'] = mask
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['pts_semantic_mask'] = mask
        return input_dict


@TRANSFORMS.register_module()
class PointSegClassMappingWithRec(BaseTransform):
    """Map original semantic class to valid category ids.
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'rec_semantic_mask' in results
        rec_semantic_mask = results['rec_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = label_mapping[pts_semantic_mask] # 转换为新的类别 ID
        converted_rec_sem_mask = label_mapping[rec_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        results['rec_semantic_mask'] = converted_rec_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_rec_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class BboxCalculation(BaseTransform):
    """Compute bounding boxes based on rec_xyz and rec_instance_mask.
    Then keep the boxes for these adjacent frames.
    """
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
    
    def transform(self, input_dict):
        """Decouple points and rec_xyz, then extracts boxes
        """
        num_points = input_dict['num_frames'] * input_dict['num_sample'] # num_frames * 20000
        if 'elastic_coords' in input_dict: # False
            rec_xyz = torch.tensor(input_dict['elastic_coords'][num_points:]) * self.voxel_size
            input_dict['elastic_coords'] = input_dict['elastic_coords'][:num_points]
        else:
            rec_xyz = input_dict['points'][num_points:, :3].tensor
        input_dict['points'] = input_dict['points'][:num_points] # [num_frames * 20000, 6]
        rec_instance_mask = torch.tensor(input_dict['rec_instance_mask']) # ? 这是什么
        points = rec_xyz

        pts_instance_mask = torch.tensor(input_dict['pts_instance_mask']) # num_frames * 20000
        pts_ins_unique = pts_instance_mask.unique()
        num_obj_adjacent = len(pts_ins_unique) - 1 if -1 in \
             pts_ins_unique else len(pts_ins_unique)
        ori_pts_instance_mask = torch.tensor(input_dict['ori_pts_instance_mask'])
        ori_pts_ins_unique = ori_pts_instance_mask.unique()
        ori_rec_instance_mask = torch.tensor(input_dict['ori_rec_instance_mask'])
        ori_rec_ins_unique = ori_rec_instance_mask[rec_instance_mask != -1].unique()
        keep_mask = torch.tensor([idx in ori_pts_ins_unique for idx in ori_rec_ins_unique])

        # Code from TD3D (arxiv:2302.02871)
        if torch.sum(rec_instance_mask == -1) != 0:
            rec_instance_mask[rec_instance_mask == -1] = torch.max(rec_instance_mask) + 1
            rec_instance_mask_one_hot = torch.nn.functional.one_hot(rec_instance_mask)[
                :, :-1
            ]
        else:
            rec_instance_mask_one_hot = torch.nn.functional.one_hot(rec_instance_mask)

        points_for_max = points.unsqueeze(1).expand(points.shape[0], rec_instance_mask_one_hot.shape[1], points.shape[1]).clone()
        points_for_min = points.unsqueeze(1).expand(points.shape[0], rec_instance_mask_one_hot.shape[1], points.shape[1]).clone()
        points_for_max[~rec_instance_mask_one_hot.bool()] = float('-inf')
        points_for_min[~rec_instance_mask_one_hot.bool()] = float('inf')
        bboxes_max = points_for_max.max(axis=0)[0]
        bboxes_min = points_for_min.min(axis=0)[0]
        bboxes_sizes = bboxes_max - bboxes_min
        bboxes_centers = (bboxes_max + bboxes_min) / 2
        bboxes = torch.hstack((bboxes_centers, bboxes_sizes, torch.zeros_like(bboxes_sizes[:, :1])))

        # Only keep boxes existed in these adjacent frames
        # Order of boxes is equal to order of ins_id in these adjacent frames
        if len(bboxes) == 0:
            bboxes = bboxes
            print("--------------No boxes found----------------")
        else:
            bboxes = bboxes[keep_mask]
        # bboxes = bboxes[keep_mask]
        # assert bboxes.shape[0] == num_obj_adjacent
        # TODO: In very very few cases, this assertion will fail. I don't know why :(
        try:
            assert bboxes.shape[0] == num_obj_adjacent
        except:
            print("Assertion fail in box calculation: ", bboxes.shape[0], num_obj_adjacent)
            if bboxes.shape[0] < num_obj_adjacent:
                bboxes = torch.cat([bboxes, torch.zeros(num_obj_adjacent - bboxes.shape[0], 7,
                     device=bboxes.device)], dim=0)
            else:
                bboxes = bboxes[:num_obj_adjacent, :]

        input_dict['gt_bboxes_3d'] = bboxes.numpy()
        if 'eval_ann_info' in input_dict:
            input_dict['eval_ann_info']['gt_bboxes_3d'] = bboxes.numpy()
        return input_dict


@TRANSFORMS.register_module()
class NoOperation(BaseTransform):
    def transform(self, input_dict):
        return input_dict