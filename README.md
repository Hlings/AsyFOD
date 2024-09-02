# (CVPR2023) AsyFOD

### Data Preparation
Please follow the instructions shown in [Dataset-Preparation](https://github.com/Hlings/AsyFOD/blob/main/Dataset-Preparation.md).

### Requirements
This repo is based on [YOLOv5 repo](https://github.com/ultralytics/yolov5). Please follow that repo for installation and preparation.
The version I built for this project is YOLO v5 3.0. The proposed methods can easily be migrated into advanced YOLO versions.

### Training

Note: All the files in the "backup" folder are unrelated to the training process, but may be helpful for ablation studies and dataset construction.

1. Modify the config of the data in the data subfolders. Please refer to the instructions in the yaml file.

2. The command below can reproduce the corresponding results mentioned in the paper.

```bash
python train.py --img 640 --batch 12 --epochs 300 --data ./data/city_and_foggy8_3.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/mm1.yaml --weights '' --name "test"
```

The codes have been released but need further construction. If you are intersted in more details of the ablation studies, you can refer to the folder "train_files_for_abl". I have listed nearly every train.py in this folder. I hope you find them helpful.

I will try my best to update :(. You can also check our previous work AcroFOD "https://github.com/Hlings/AcroFOD".

- If you find this paper/repository useful, please consider citing our paper:

```
@inproceedings{gao2023asyfod,
  title={AsyFOD: An Asymmetric Adaptation Paradigm for Few-Shot Domain Adaptive Object Detection},
  author={Gao, Yipeng and Lin, Kun-Yu and Yan, Junkai and Wang, Yaowei and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3261--3271},
  year={2023}
}
```
