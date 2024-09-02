# (CVPR2023) AsyFOD: An asymmetric adaptation paradigm for few-shot domain adaptive object detection

### Data Preparation
Please follow the instructions shown in [Dataset-Preparation](https://github.com/Hlings/AsyFOD/blob/main/Dataset-Preparation.md).

### Requirements
This repo is based on [YOLOv5 repo](https://github.com/ultralytics/yolov5). Please follow that repo for installation and preparation.
The version I built for this project is YOLO v5 3.0. The proposed methods can easily be migrated into advanced YOLO versions.

### Training

Note: All the files in the "backup" folder are unrelated to the training process but may be helpful for ablation studies and dataset construction.

1. Modify the config of the data in the data subfolders. Please refer to the instructions in the yaml file (shown in the examples from [here](https://github.com/Hlings/AsyFOD/tree/main/data)).

2. The command below can reproduce the corresponding results mentioned in the paper.

**For cityscapes to foggy cityscapes,** please use the following command:

```bash
python train.py --img 640 --batch 12 --epochs 300 --data ./data/city2foggy.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/mm1.yaml --weights '' --name "city2foggy_exp"
```

**For sim10k tocityscapes,** please use the following command:

```bash
python train.py --img 640 --batch 32 --epochs 300 --data ./data/sim10k2citycar.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/mm3.yaml --weights '' --name "sim10k2citycar_exp"
```

The codes have been released but may need further construction. If you are interested in more details of the ablation studies, you can refer to the folder "train_files_for_abl". I have listed nearly every train.py in this folder. I hope you find them helpful.

- Please don't hesitate to reach out to me via [yipengga@usc.edu](yipengga@usc.edu) or [gaoyp23@mail2.sysu.edu.cn](gaoyp23@mail2.sysu.edu.cn).

- If you find this paper/repository useful, please consider citing our papers:

```
@inproceedings{gao2023asyfod,
  title={AsyFOD: An Asymmetric Adaptation Paradigm for Few-Shot Domain Adaptive Object Detection},
  author={Gao, Yipeng and Lin, Kun-Yu and Yan, Junkai and Wang, Yaowei and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3261--3271},
  year={2023}
}
```

```
@inproceedings{gao2022acrofod,
  title={Acrofod: An adaptive method for cross-domain few-shot object detection},
  author={Gao, Yipeng and Yang, Lingxiao and Huang, Yunmu and Xie, Song and Li, Shiyong and Zheng, Wei-Shi},
  booktitle={European Conference on Computer Vision},
  pages={673--690},
  year={2022},
  organization={Springer}
}
```
