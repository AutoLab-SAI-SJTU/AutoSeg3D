## Train and Evaluation

### Class-agnostic 3D instance segmentation on ScanNet200:

Train and evaluate AutoSeg3D on ScanNet200-SV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet200/AutoSeg3D_sv_scannet200.py --work-dir work_dirs/AutoSeg3D_sv_scannet200/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet200/AutoSeg3D_sv_scannet200.py work_dirs/AutoSeg3D_sv_scannet200/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_sv_scannet200/
```

Train and evaluate AutoSeg3D on ScanNet200-MV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet200/AutoSeg3D_scannet200_stage1.py --work-dir work_dirs/AutoSeg3D_scannet200_stage1/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet200/AutoSeg3D_scannet200_stage1.py work_dirs/AutoSeg3D_scannet200_stage1/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_scannet200_stage1/

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet200/AutoSeg3D_scannet200_stage2.py --work-dir work_dirs/AutoSeg3D_scannet200_stage2/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet200/AutoSeg3D_scannet200_stage2.py work_dirs/AutoSeg3D_scannet200_stage1/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_scannet200_stage2/
```

### Class-aware 3D instance segmentation on ScanNet:


Train and evaluate AutoSeg3D on ScanNet-SV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet/AutoSeg3D_sv_scannet.py --work-dir work_dirs/AutoSeg3D_sv_scannet/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet/AutoSeg3D_sv_scannet.py work_dirs/AutoSeg3D_sv_scannet/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_sv_scannet/
```

Train and evaluate AutoSeg3D on ScanNet-MV (Class Agnostic)：

```
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet/AutoSeg3D_scannet200_stage1.py --work-dir work_dirs/AutoSeg3D_scannet_stage1/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet/AutoSeg3D_scannet_stage1.py work_dirs/AutoSeg3D_scannet_stage1/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_scannet_stage1/

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/scannet/AutoSeg3D_scannet_stage2.py --work-dir work_dirs/AutoSeg3D_scannet_stage2/
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/scannet/AutoSeg3D_scannet_stage2.py work_dirs/AutoSeg3D_scannet_stage1/epoch_xx.pth --work-dir work_dirs/AutoSeg3D_scannet_stage2/
```