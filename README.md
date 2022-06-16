# Codes-for-PVKD
Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation (CVPR 2022)

Our model achieves state-of-the-art performance on three benchmarks, i.e., ranks **1st** in [Waymo 3D Semantic Segmentation Challenge](https://waymo.com/open/challenges/2022/3d-semantic-segmentation/) (the "Cylinder3D" and "Offboard_SemSeg" entities), ranks **1st** in [SemanticKITTI LiDAR Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/20331#results) (single-scan, the "Point-Voxel-KD" entity), ranks **3rd** in [SemanticKITTI LiDAR Semantic Segmentation Challenge](https://competitions.codalab.org/competitions/20331#results) (multi-scan, the "PVKD" entity). Our trained model has been used in one NeurIPS 2022 submission! Please do not hesitate to use our trained models!

## Installation

### Requirements
- PyTorch >= 1.2 
- yaml
- tqdm
- numba
- Cython
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==1.2.1 and cuda==10.2)

## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

### Waymo
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──first_return
		├──second_return

```

## Test

First, download the [pre-trained models]() and put them in ./model_load_dir.

Then, generate predictions on the SemanticKITTI test set.

```
CUDA_VISIBLE_DEVICES=0 python -u test_cyl_sem_tta.py
```

We perform test-time augmentation to boost the performance. The model predictions will be saved in ./out_cyl/test by default.


Convert label number back to the original dataset format before submitting:
```
python remap_semantic_labels.py -p out_cyl/test -s test --inverse
cd out_cyl/test
zip -r out_cyl.zip sequences/
```

Finally, upload out_cyl.zip to the [SemanticKITTI online server](https://competitions.codalab.org/competitions/20331#participate).

## Train

```
CUDA_VISIBLE_DEVICES=0 python -u train_cyl_sem.py
```

Currently, we only support vanilla training.


## Performance

1. SemanticKITTI test set (single-scan):

|Model|Reported|Reproduced|Gain|
|:---:|:---:|:---:|:---:|
|SPVNAS|66.4%|--|--|
|Cylinder3D|68.9%|71.8%|**2.9%**|
|Cylinder3D_0.5x|71.2%|71.4%|0.2%|

2. SemanticKITTI test set (multi-scan):


3. Waymo test set:


4. nuScenes val set:


## Citation
If you use the codes, please cite the following publications:
```
@InProceedings{Hou_2022_CVPR,
    author    = {Hou, Yuenan and Zhu, Xinge and Ma, Yuexin and Loy, Chen Change and Li, Yikang},
    title     = {Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year      = {2022},
    pages     = {8479-8488}
}

@inproceedings{zhu2021cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9939--9948},
  year={2021}
}

@article{zhu2021cylindrical-tpami,
  title={Cylindrical and Asymmetrical 3D {C}onvolution {N}etworks for LiDAR-based Perception},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Li, Wei and Ma, Yuexin and Li, Hongsheng and Yang, Ruigang and Lin, Dahua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgments
This repo is built upon the awesome [Cylinder3D](https://github.com/xinge008/Cylinder3D).