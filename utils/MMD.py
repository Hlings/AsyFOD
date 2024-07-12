import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device, intersect_dicts
from models.yolo_feature import Model_feature
import random
import torch.nn as nn

# this feature extraction method exists bug !!!
"""
def get_ins_feature(feature, targets, device, number=100):
    feat_w, feat_h = feature.shape[2], feature.shape[3]
    ins_set = torch.tensor([]).to(device)
    targets_ins = random_sample_ins(targets, number)
    for ins in targets_ins:
        img_idx = int(ins[0])
        img_cls = ins[1] # for multi class api
        x,y,w,h = ins[2], ins[3], ins[4], ins[5]
        # x1 < x2; y1 < y2
        # now exsting bugs !!!
        x1, x2 = max(int(feat_w*(x-w)/2), 0), max(int(feat_w*(x+w)/2), int(feat_w*(x-w)/2)+1)
        y1, y2 = max(int(feat_h*(y-h)/2), 0), max(int(feat_h*(y+h)/2), int(feat_h*(y-h)/2)+1)
        #print(img_idx)
        #print(x1, x2, y1, y2)
        #print(feature.shape)
        o_ins = feature[img_idx, :, x1:x2, y1:y2].mean(2).mean(1).unsqueeze(0)
        #print(o_ins.shape)
        ins_set=torch.cat((ins_set, o_ins))
    return ins_set
"""
def get_ins_feature(feature, targets, device, number=100):
    feat_w, feat_h = feature.shape[2], feature.shape[3]
    ins_set = torch.tensor([]).to(device)
    targets_ins = random_sample_ins(targets, number)
    for ins in targets_ins:
        img_idx = int(ins[0])
        img_cls = ins[1] # for multi class api
        x,y,w,h = ins[2], ins[3], ins[4], ins[5]
        # x1 < x2; y1 < y2
        # now exsting bugs !!!
        x1, x2 = max(int(feat_w*(x-0.5*w)), 0), max(int(feat_w*(x+0.5*w)/2), int(feat_w*(x-0.5*w))+1)
        y1, y2 = max(int(feat_h*(y-0.5*h)), 0), max(int(feat_h*(y+0.5*h)/2), int(feat_h*(y-0.5*h))+1)
        #print(img_idx)
        #print(x1, x2, y1, y2)
        #print(feature.shape)
        o_ins = feature[img_idx, :, x1:x2, y1:y2].mean(2).mean(1).unsqueeze(0)
        #print(o_ins.shape)
        ins_set=torch.cat((ins_set, o_ins))
    return ins_set

def random_sample_ins(targets, number):
    n = targets.shape[0]
    k = number
    # print("number of targets is", n)
    # print("considered number is", k)
    if n <= k:
        #print("continue")
        return targets
    else:
        # print("seleted")
        indices = torch.tensor(random.sample(range(n), k))
        targets_ = targets[indices]
        return targets_

def get_feature(img, model): 
    model.eval()
    img_feature = model(img)[1] # torch.tensor [B 1280 H/32 W/32]
    img_feature = img_feature.mean(3).mean(2)
    return img_feature  # torch.tensor [B, 1280] feature

def get_feature_train(img, model): 
    img_feature = model(img)[1] # torch.tensor [B 1280 H/32 W/32]
    img_feature = img_feature.mean(3).mean(2)
    return img_feature  # torch.tensor [B, 1280] feature

# To get the weight for source and target samples in task-oriented supervised trainng
def MMD_weight(feature_s, feature_t, k): # S: [B1 1280] T: [B2 1280] 
    feature_s = feature_s - feature_t.mean(0) # boardcast
    feature_s = feature_s.mul(feature_s) # multiply by every position

    feature_s = feature_s.sum(dim=1)
    
    if feature_s.shape[0] > k:
        topk_index = feature_s.topk(k=k, largest = False)[1]
    else: 
        topk_index = torch.arange(0, feature_s.shape[0])
    
    batch_size = feature_s.shape[0]
    weight_ini = torch.zeros(batch_size)
    weight_ini[topk_index] = 1.0
    weight = weight_ini.clone()
    
    return weight # return sample's weight defined by MMD distance

def MMD_distance(f_S, f_T, k): # S: [B1 1280] T: [B2 1280] 这里的f_T其实代表源域特征 返回源域当中的topk个
    f_T = f_T - f_S.mean(0) # 广播 每行都会减小
    f_T = f_T.mul(f_T) #对位相乘

    f_T = f_T.sum(dim=1)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] # return topk_idx

def MMD_distance_v2(f_t, f_s, k): # S: [B1 1280] T: [B2 1280] 这里的f_T其实代表源域特征 返回源域当中的topk个
    time = 0
    for i in range(f_t.shape[0]):
    # print(i)
        f_i = f_s - f_t[i]
        f_i = f_i.mul(f_i).sum(dim=1)

        if f_i.shape[0] > k:
            i_topk = f_i.topk(k=k, largest = False)[1]
        else: 
            return torch.arange(0, f_i.shape[0])
        
        if time == 0:
            f_topk = i_topk 
        else:
            f_topk = torch.cat((f_topk, i_topk), 0)
        time += 1
    idx, counts = f_topk.unique(sorted=False, return_counts=True)
        
    return idx[counts.topk(k=k, largest = True)[1]] # return topk_idx

def cosine_distance(f_S, f_T, k): # f_S: [B1 1280] 所有target图片的特征 T: [B2 1280]
    dist_tensor = []
    for i in f_T:
        i = i.unsqueeze(0)
        dist = (1-torch.cosine_similarity(f_S,i,dim=1)).sum()
        dist_tensor.append(dist)
        
    f_T = torch.tensor(dist_tensor)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] # T中和 

def choice_topk(imgs, targets, paths, topk_index):
    paths_refine = []
    labels_refine = torch.tensor([])

    for i in list(topk_index):  #  这里可能遇到一种情况  就是要补充上的样本数目小于 筛选后的样本
        paths_refine.append(paths[i])
        
    for index in topk_index:
        t = targets[targets[:, 0] == index, :]
        if t.shape[0] > 0:
            number = torch.nonzero((topk_index == t[0][0]))[0][0]
            t[:, 0] = number
        labels_refine = torch.cat((labels_refine, t), dim = 0)
    
    imgs_refine = imgs[topk_index, :, :, :]
    return imgs_refine, labels_refine, paths_refine

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起
