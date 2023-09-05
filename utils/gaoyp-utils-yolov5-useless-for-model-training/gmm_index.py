# -*- coding: utf-8 -*-
import torch
import random
import utils.gmm as gmm
import torch.nn as nn

def get_ins_feature(feature, targets, device, number=100):
    feat_w, feat_h = feature.shape[2], feature.shape[3]
    ins_set = torch.tensor([]).to(device)
    targets_ins = random_sample_ins(targets, number)
    for ins in targets_ins:
        img_idx = int(ins[0])
        img_cls = ins[1] # for multi class api
        x,y,w,h = ins[2], ins[3], ins[4], ins[5]
        # x1 < x2; y1 < y2
        x1, x2 = max(int(feat_w*(x-w)/2), 0), max(int(feat_w*(x+w)/2), int(feat_w*(x-w)/2)+1)
        y1, y2 = max(int(feat_h*(y-h)/2), 0), max(int(feat_h*(y+h)/2), int(feat_h*(y-h)/2)+1)
        #print(img_idx)
        #print(x1, x2, y1, y2)
        #print(feature.shape)
        o_ins = feature[img_idx, :, x1:x2, y1:y2].mean(2).mean(1).unsqueeze(0)
        #print(o_ins.shape)
        ins_set=torch.cat((ins_set, o_ins))
    return ins_set

# eg: source_ins_index = get_index_via_gmm(source_ins, target_ins)
def get_index_via_gmm(source_ins, target_ins):
    AdaPool = nn.AdaptiveAvgPool1d(6)

    source_ins_gmm = AdaPool(source_ins.detach().unsqueeze(0)).squeeze(0)
    target_ins_gmm = AdaPool(target_ins.detach().unsqueeze(0)).squeeze(0)
    # print(target_ins_gmm.shape)
    # 迭代把source的实例加入到target当中 一次选
    gmm_iter = 2
    n_components = 2 # now the maximum is 5
    #print(target_ins_gmm.shape)
    full_index = [i for i in range(source_ins_gmm.shape[0])]
    source_ins_gmm_iter = source_ins_gmm.clone()
    target_ins_gmm_iter = target_ins_gmm.clone()
    for i in range(gmm_iter):
        # .. used to predict the data points as they where shifted
        # d = target_ins_gmm.shape[1]
        d = 6
        model_gmm = gmm.GaussianMixture(n_components, d).cuda()
        # print("target_ins_gmm_iter's shape is ")
        # print(target_ins_gmm_iter.shape)
        model_gmm.fit(target_ins_gmm_iter)
        source_p = model_gmm.predict(source_ins_gmm_iter, probs=True, ood=True).mean(1)
        # 选择top-k高的概率
        top_k_index = list(source_p.topk(k=10, largest = True)[1])
        # print(top_k_index) # start by 1
        #print("full_index", full_index)
        source_ins_gmm_iter, target_ins_gmm_iter, full_index = update_by_index(source_ins_gmm, target_ins_gmm_iter, top_k_index, full_index)
        #print(source_ins_gmm_iter.shape)
        #print(target_ins_gmm_iter.shape)

    # 可以利用这个full-index 直接从source feature 和 target feature里面调取
    # 后面的index是所有考虑的样本
    selected_index = [i for i in range(source_ins_gmm.shape[0]) if i not in full_index] 
    return full_index, selected_index

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


def update_by_index(source_ins_gmm, target_ins_gmm_iter, index, full_index):
    template = full_index.copy()
    # print(index)
    # full index 代表除了index之外剩下的样本
    for i in index:
        # print(template[i])
        full_index.remove(template[i])
        # print(full_index)
    # 这里选择到的样本应该是概率更大的样本 和目标域更近
    selected_ins = source_ins_gmm[index, :] 
    source_ins_gmm_iter = source_ins_gmm[full_index, :]
    target_ins_gmm_iter = torch.cat((target_ins_gmm_iter, selected_ins), dim=0)
    # 直接利用full_index从原始中取元素就可以了
    return source_ins_gmm_iter, target_ins_gmm_iter, full_index
