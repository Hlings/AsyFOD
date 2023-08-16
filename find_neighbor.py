# -*- coding: utf-8 -*-
import argparse #解析命令行参数的库
import json     #实现字典列表和JSON字符串之间的相互解析
import os       #与操作系统进行交互的文件库 包含文件路径操作与解析
from pathlib import Path  #Path能够更加方便得对字符串路径进行处理
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
# for instance-level feature_extraction
from utils.MMD import get_ins_feature
import matplotlib.image as image

def crop_img(xyxy,img):
    x1 = int(xyxy[0])
    x2 = int(xyxy[2])
    y1 = int(xyxy[1])
    y2 = int(xyxy[3])
    cropped_img = img[x1:x2,y1:y2]
    return cropped_img

def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return 

#测试函数 输入为测试过程中需要的各种参数
def test(data,
         weights=None,
         batch_size=32,
         imgsz=1280,
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
         cur_number = 1,
         path_base = '/userhome/1_xin/yipeng/acrofodv2-branch/t-sne/example_a1.pdf',
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
        imgsz = 1280
        print("image size is,", imgsz)
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        print("image size is,", imgsz)
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
    batch_max_num = 30
    cur_batch = 0
    
    #----  extract known target feature representation -- #
    feature_known_target = torch.tensor([]).to(device)
    targets_t_ins = torch.tensor([]).to(device)
    imgs_t = torch.tensor([]).to(device)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_known_target, desc=s)): # 对每一个batch推断
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        targets_t_ins = torch.cat((targets_t_ins, targets))
        imgs_t = torch.cat((imgs_t, img))
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            t = time_synchronized()
            feature = model(img, augment=augment)[1] # inference and training outputs
            # return all of the feature representation
            feature = get_ins_feature(feature, targets, device, number=1000).to(device)
            feature_known_target = torch.cat((feature_known_target, feature))
    
    # feature_known_target: [N, D] targets_t_ins [N, 5]
    
    # random select target prototypes
    k = 50
    target_index = torch.tensor(random.sample(range(feature_known_target.shape[0]), k))
    print(feature_known_target.shape, targets_t_ins.shape,)
    print(target_index)
    target_ins_prototype_feature = feature_known_target[target_index]
    target_ins_prototype_label = targets_t_ins[target_index]
    print(target_ins_prototype_feature.shape, target_ins_prototype_label.shape)
    print("the shape of img is: ", imgs_t.shape)
    
    print("-------------debug end--------------")
    
    # for crop the instance
    for i in range(target_ins_prototype_label.shape[0]):
    # label: [img_idx, cls, x, y, w, h]
        print(target_ins_prototype_label[i])
        img_idx, _, x, y, w, h = target_ins_prototype_label[i]
        print("the xywh is ", x, y, w, h)
        # img: torch.Size([8, 3, 352, 672]) h 672 w 352
        print("the img_idx is,", img_idx)
        img = imgs_t[int(img_idx)]
        # testing if the problem is w/h
        feat_w, feat_h = img.shape[2], img.shape[1]
        x1, x2 = max(int(feat_w*(x-0.5*w)), 0), max(int(feat_w*(x+0.5*w)), int(feat_w*(x-0.5*w))+1)
        y1, y2 = max(int(feat_h*(y-0.5*h)), 0), max(int(feat_h*(y+0.5*h)), int(feat_h*(y-0.5*h))+1)
        print("the shape of img is", img.shape)
        print("the xxyy is ", x1, x2, y1, y2)
        crop = img[:, x1:x2, y1:y2].transpose(2,0).cpu()
        print(crop.shape)
        # visulization
        crop_array = np.array(crop)
        print("the array of crop:", crop_array.shape)
        out_path = "./ins_visual/target_debug_"+str(i)+"_.jpg"
        image.imsave(out_path, crop_array)
        if i > 5:
            break
    #----------------known target end -------------------- #
    
    #-------  extract source feature representation ----- #
    feature_source = torch.tensor([]).to(device)
    batch_index=random.sample(range(0, 1244),50)
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_source, desc=s)): # 对每一个batch推断
        if batch_i not in batch_index:
            continue
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        # get_ins_feature(feature_img_t, targets_t_, device, number=50).to(device)
        with torch.no_grad():
            t = time_synchronized()
            feature = model(img, augment=augment)[1]  # inference and training outputs
            feature = get_ins_feature(feature, targets, device, number=20).to(device)
            feature_source = torch.cat((feature_source, feature))
        
        cur_batch +=1
        if cur_batch > batch_max_num:
            break
    feature_source = feature_source.cpu()
    # regard source as cls 0
    label_source = [0 for index in range(feature_source.shape[0])]
    #---------------------source end -------------------- #
    
    return batch_max_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--path_base', default='/userhome/1_xin/yipeng/acrofodv2-branch/t-sne/example_a1.pdf', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='hhh')
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
    # path_base = '/userhome/1_xin/yipeng/acrofodv2-branch/t-sne/over_adaptation/over_adapt_a1.pdf'
    if opt.task in ['val', 'test']:  # run normally
        for cur_number in range(1,30):
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
                 cur_number=cur_number,
                 path_base=opt.path_base,
                 )
