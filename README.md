# (CVPR2023) AsyFOD

### Data Preparation
Download the corresponding dataset for training. The data type has been prepared.

**Source domain:**

[Sim10K](https://pan.baidu.com/s/1fd1hwyGkwn-cjBL5YPCAbg?pwd=juf6) Key: juf6 (The synthetic dataset includes only car class.)

[KITTI](https://pan.baidu.com/s/1edDtirk4IX9yFnsCGrzjDg?pwd=8brv) Key: 8brv (The KITTI dataset includes only car class.)

[Cityscapes_8cls](https://pan.baidu.com/s/1lPjaHOgoh5YCJcnP1hTzDw?pwd=rg4z) Key: rg4z (The Cityscapes dataset includes 8 classes.)

[Viped](https://pan.baidu.com/s/1a1SHZ4eb2q5mSyqWY2ZQmQ?pwd=a9y7) Key: a9y7 (The synthetic dataset includes)

**Target domain:**

[Cityscapes_car](https://pan.baidu.com/s/1pU7NleGc-yG_JRLFjIKcxA?pwd=4ym4) Key: 4ym4 (The cityscapes dataset includes only car class.)

[Cityscapes_car_8_1](https://pan.baidu.com/s/1VjJn4aN5w9FdXzgIosr79Q?pwd=p69u) Key: p69u (The randomly selected 8 images from cityscapes_car.)

[Cityscapes_car_8_2](https://pan.baidu.com/s/1rndTqOBVq7tKaw7giN9hTQ?pwd=qe7a). Key: qe7a (The randomly selected 8 images from cityscapes_car.)

[Cityscapes_car_8_3](https://pan.baidu.com/s/178Vu0QpQAE8FGNy1xv0fpA?pwd=x3ei). Key: x3ei (The randomly selected 8 images from cityscapes_car.)

[Cityscapes_8cls_foggy](https://pan.baidu.com/s/1q560FQw-WSFq2_NucNNfVA?pwd=bjgr) Key: bjgr (The Foggy Cityscapes dataset includes 8 classes.)

[coco_person](https://pan.baidu.com/s/1nwr7qVAFnXM3mK2b5Ywc9g?pwd=je89) Key: je89 (The COCO dataset includes only person class.)

[coco_person_60](https://pan.baidu.com/s/1VqpxNbjGjAMZvOF3HBttqw?pwd=vg1m) Key: vg1m (The randomly selected 60 images from coco_person.)


You can also process the raw data to Yolo format via the tools shown [here](https://github.com/Hlings/AsyFOD/tree/main/utils/gaoyp-utils-yolov5-useless-for-model-training).

### Requirements
This repo is based on [YOLOv5 repo](https://github.com/ultralytics/yolov5). Please follow that repo for installation and preparation.
The version I built for this project is YOLO v5 3.0. The proposed methods can easily be migrated into advanced YOLO versions.

### Training
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
