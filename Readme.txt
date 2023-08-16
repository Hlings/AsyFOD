###  配置环境  ###
基本配置文件在requirements.txt文件中

pip install -r requirements.txt  # install

一般的能运行yolov5的环境都能够正常运行

###  整理数据  ###
数据存储的格式需要为标准的YOLO格式
./folder
---/image
-----/train
-----/val
---/label
-----/train
-----/val

###  改数据配置文件 ###
AcroFOD的配置文件
样例文件在./AcroFOD/data/eg文件夹下的co_ps_01.yaml文件当中

需要修改的有五部分：
train_source： 训练过程的源域大量样本集合
train_target:  训练过程的目标域少量样本集合 这个集合的样本可以少到8张，甚至4张图片
val：          验证过程的目标域大量样本测试集，评估测试指标

nc改为数据对应的类别数目 class names也按照0、1这样标注的顺序修改为对应的名字

YOLOv5的baseline 配置文件
样例文件在./AcroFOD/data/eg文件夹下的city_and_foggy8_3.yaml文件当中
需要修改train 和 val路径 nc和 class names

###   运行命令 ###

(1)执行 AcroFOD算法示例
cd /userhome/AcroFOD; python train_MMD.py --img 640 --batch 4 --epochs 3 --data ./data/eg/city_and_foggy8_3.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/m3.yaml --weights '' --name "test"

(2)执行 yolov5 baseline实例
cd /userhome/AcroFOD; python train.py --img 640 --batch 4 --epochs 3 --data ./data/eg/co_ps_01.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/m1.yaml --weights '' --name "test"


###  测试命令 ###
(1)执行 AcroFOD算法示例
cd /userhome/AcroFOD; python test_MMD.py --weights './weights/best.pt' --data ./data/sim10k_to_city.yaml --conf-thres 0.0 --name 'val_cityscapes' --task 'val'

(2)执行 yolov5 baseline实例
cd /userhome/AcroFOD; python test.py --weights './weights/best.pt' --data ./data/sim10k_to_city.yaml --conf-thres 0.0 --name 'val_cityscapes' --task 'val'

备注：
1)hyp为对应的超参数配置文件，在AcroFOD算法中推荐使用mm1 mm2 mm3 mm4 不同配置会有2-3个点的浮动 
2)在yolov5 baseline推荐使用m1 m2 m3 m4 一般会带来较好的效果
3)一般一张V100对应最大的bs=16 显存是够的 yolov5在单节点上自动占满全部卡
4)bs推荐从32开始尝试，对应两张V100，如果数据原始尺寸较大且没有收敛，推荐降低bs数目到16甚至是12.
5)./utils 文件夹中放了一些当时处理数据的代码，不过没有整理。
6)通过train_MMD.py训练文件得到的模型需要用test_MMD.py文件进行测试，同样train对应test。