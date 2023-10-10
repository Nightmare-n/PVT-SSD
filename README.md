[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.06621)
[![GitHub Stars](https://img.shields.io/github/stars/Nightmare-n/PVT-SSD?style=social)](https://github.com/Nightmare-n/PVT-SSD)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Nightmare-n/PVT-SSD)

# PVT-SSD: Single-Stage 3D Object Detector with Point-Voxel Transformer


## Installation
We test this project on NVIDIA A100 GPUs and Ubuntu 18.04.
```
conda create -n pvt-ssd python=3.7
conda activate pvt-ssd
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 einops==0.6.0 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
git clone https://github.com/Nightmare-n/PVT-SSD
cd PVT-SSD && python setup.py develop --user
```

## Data Preparation

Please follow the [instruction](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) of OpenPCDet to prepare the dataset. For the Waymo dataset, we use the [evaluation toolkits](https://drive.google.com/drive/folders/1aa1kI9hhzBoZkIBcr8RBO3Zhg_RkOAag?usp=sharing) to evaluate detection results.
```
data
│── waymo
│   │── ImageSets/
│   │── raw_data
│   │   │── segment-xxxxxxxx.tfrecord
│   │   │── ...
│   │── waymo_processed_data
│   │   │── segment-xxxxxxxx/
│   │   │── ...
│   │── waymo_processed_data_gt_database_train_sampled_1/
│   │── waymo_processed_data_waymo_dbinfos_train_sampled_1.pkl
│   │── waymo_processed_data_infos_test.pkl
│   │── waymo_processed_data_infos_train.pkl
│   │── waymo_processed_data_infos_val.pkl
│   │── compute_detection_metrics_main
│   │── gt.bin
│── kitti
│   │── ImageSets/
│   │── training
│   │   │── label_2/
│   │   │── velodyne/
│   │   │── ...
│   │── testing
│   │   │── velodyne/
│   │   │── ...
│   │── gt_database/
│   │── kitti_dbinfos_train.pkl
│   │── kitti_infos_test.pkl
│   │── kitti_infos_train.pkl
│   │── kitti_infos_val.pkl
│   │── kitti_infos_trainval.pkl
│── once
│   │── ImageSets/
│   │── data
│   │   │── 000000/
│   │   │── ...
│   │── gt_database/
│   │── once_dbinfos_train.pkl
│   │── once_infos_raw_large.pkl
│   │── once_infos_raw_medium.pkl
│   │── once_infos_raw_small.pkl
│   │── once_infos_train.pkl
│   │── once_infos_val.pkl
│── kitti-360
│   │── data_3d_raw
│   │   │── xxxxxxxx_sync/
│   │   │── ...
│── ckpts
│   │── pvt_ssd.pth
│   │── ...
```

## Training & Testing
```
# train
bash scripts/dist_train.sh

# test
bash scripts/dist_test.sh
```

## Results

### Waymo
|                                             | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Model |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[PVT-SSD](tools/cfgs/waymo_models/pvt_ssd.yaml)| 79.2/78.7|70.2/69.8|79.9/74.0|72.6/67.0|77.1/76.0|74.0/73.0| [log](https://drive.google.com/file/d/1ZBq_4xLlaMTxxX57T2GFeTM87Ey9cwGu/view?usp=sharing) |
|[PVT-SSD_3f](tools/cfgs/waymo_models/pvt_ssd_3f.yaml)| 80.6/80.2|71.9/71.5|83.9/80.6|75.1/72.1|77.9/77.0|74.8/74.0| [log](https://drive.google.com/file/d/1Zx6OYludbb_WR6agAPdfWFL9HYVtalix/view?usp=sharing) |

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/).

## Citation 
If you find this project useful in your research, please consider citing:
```
@inproceedings{yang2023pvtssd,
    author    = {Yang, Honghui and Wang, Wenxiao and Chen, Minghao and Lin, Binbin and He, Tong and Chen, Hua and He, Xiaofei and Ouyang, Wanli},
    title     = {PVT-SSD: Single-Stage 3D Object Detector With Point-Voxel Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13476-13487}
}
```

## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!

* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
