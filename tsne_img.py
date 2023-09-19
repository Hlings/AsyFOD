# -*- coding: utf-8 -*-
import argparse 
import json     
import os
from pathlib import Path
from threading import Thread #python中处理多线程的库

import numpy as np    #矩阵计算基础库
import torch          #pytorch 深度学习库
import yaml           #yaml是一种表达高级结构的语言 易读 便于指定模型架构及运行配置
from tqdm import tqdm #用于直观显示进度条的一个库 看起来很舒服

from models.experimental import attempt_load #调用models文件夹中的experimental.py文件中的attempt_load函数 目的是加载模型
#以下调用均为utils文件夹中各种已写好的函数
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

# for t-sne visualization
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
import random

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

# for polting eg:
"""
feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
label_test1 = [0 for index in range(40)]
label_test2 = [1 for index in range(40)]
label_test3 = [2 for index in range(48)]

label_test = np.array(label_test1 + label_test2 + label_test3)
print(label_test)
print(label_test.shape)

fig = plt.figure(figsize=(10, 10))

plotlabels(visual(feat), label_test, '(a)')

"""
def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    #print(S_data)
    print(S_data.shape)  # [num, 3]
    
    maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    colors = ['#e38c7a', 'blue', 'olivedrab', '#656667', '#99a4bc', 'cyan', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
    Label_Com = ['a', 'b', 'c', 'd']
    font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }
    for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)
    plt.savefig("/userhome/1_xin/yipeng/acrofodv2-branch/example.pdf")

#测试函数 输入为测试过程中需要的各种参数
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.0, # old is 0.001 but 0 is reasonalbe for NMS
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=True,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         mark_result='No name',
         plots=True,
         log_imgs=0):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path_source = data['train_source']
        path_known_target = data['train_known_target']
        path_unknown_target = data['train_unknown_target']
        
        dataloader_source = create_dataloader(path_source, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]
        dataloader_known_target = create_dataloader(path_known_target, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]
        dataloader_unknown_target = create_dataloader(path_unknown_target, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    batch_max_num = 5
    cur_batch = 0
    
    #-------  extract source feature representation ----- #
    feature_source = torch.tensor([]).to(device)
    batch_index=random.sample(range(0, 1244),10)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_source, desc=s)): # 对每一个batch推断
        if batch_i not in batch_index:
            continue
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t = time_synchronized()
            feature = model(img, augment=augment)[1].mean(3).mean(2)  # inference and training outputs
            feature_source = torch.cat((feature_source, feature))
        
        cur_batch +=1
        if cur_batch > batch_max_num:
            break
    feature_source = feature_source.cpu()
    # regard source as cls 0
    label_source = [0 for index in range(feature_source.shape[0])]
    #---------------------source end -------------------- #
    
    #----  extract known target feature representation -- #
    feature_known_target = torch.tensor([]).to(device)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_known_target, desc=s)): # 对每一个batch推断
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t = time_synchronized()
            feature = model(img, augment=augment)[1].mean(3).mean(2)  # inference and training outputs
            feature_known_target = torch.cat((feature_known_target, feature))
        
    feature_known_target = feature_known_target.cpu()
    # regard known_target as cls 0
    label_known_target = [1 for index in range(feature_known_target.shape[0])]
    #----------------known target end -------------------- #
    
    
    #----  extract unknown target feature representation -- #
    batch_max_num = 5
    cur_batch = 0
    feature_unknown_target = torch.tensor([]).to(device)
    batch_index=random.sample(range(0, 350),10)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_unknown_target, desc=s)): # 对每一个batch推断
        if batch_i not in batch_index:
            continue
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t = time_synchronized()
            feature = model(img, augment=augment)[1].mean(3).mean(2)  # inference and training outputs
            feature_unknown_target = torch.cat((feature_unknown_target, feature))
        cur_batch +=1
        if cur_batch > batch_max_num:
            break
        
    feature_unknown_target = feature_unknown_target.cpu()
    # regard known_target as cls 0
    label_unknown_target = [2 for index in range(feature_unknown_target.shape[0])]
    #----------------unknown target end -------------------- #
    
    
    #------------------visualization start--------------- #
    feature = torch.cat((feature_source, feature_known_target, feature_unknown_target))
    fig = plt.figure(figsize=(10, 10))
    
    label_np = np.array(label_source + label_known_target + label_unknown_target)
    plotlabels(visual(feature), label_np, '(a)')
    #---------------------------------------------------- #
    return batch_max_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--mark_result', type=str, default='no name', help='mark output result') # 后添加的
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             mark_result=opt.mark_result,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(f, x)  # plot
