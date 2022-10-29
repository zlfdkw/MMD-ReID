from __future__ import print_function, absolute_import
import imp
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from utils import *
from data.data_loader import RGBIRData

np.set_printoptions(threshold=np.inf)
# p_pids = [15, 15, 15, 15, 15]
# g_pids = [11, 12, 13, 14, 15]
# g_pids = np.array(g_pids)
# p_pids = np.array(p_pids)
# # print(pids)
# distmat = [[ 6, 4, 8, 3, 2],
#            [ 5, 6, 7, 8, 9],
#            [10, 11, 12, 13, 14],
#            [15, 16, 17, 18, 19]]
# # print(distmat)
# indices = np.argsort(distmat, axis=1)
# print(indices)
# # print(pids[indices])
# pre_label = g_pids[indices]
# matches = (g_pids[indices] == p_pids[:, np.newaxis])
# print(matches)

# feat = np.arange(120).reshape(4, 30)
# feat = torch.from_numpy(feat)
# ptr = 0
# batch_num = feat.size(0)
# gall_feat_pool = np.zeros((4, 30))
# print(feat)
# gall_feat_pool[ptr:ptr+batch_num,: ] = feat.detach()
# print(gall_feat_pool)
# metrics = {'Rank-1':[], 'mAP': [], 'mINP': [], 'Rank-5':[], 'Rank-10':[], 'Rank-20':[]}
# print(type(metrics))
# m = nn.BatchNorm1d(10)
# input = torch.arange(50).reshape(5, 10).float()
# output = m(input)
# print(input)
# print(output)
# x = torch.randn(1, 2, 3, 4)
# print(x)
# b, c, h, w = x.shape
# x = x.view(b, c, -1)
# p = 3.0
# x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
# print(x)
# Avg_pool = nn.AdaptiveAvgPool2d((1, 1))
# x_pool = Avg_pool(x)
# x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
# print(x_pool)
data_path = 'C:\data\dataset\RGBNT\\rgbir'
# img = Image.open(data_path)
# # img.show()
# trans = transforms.RandomHorizontalFlip()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# img_trans = normalize(trans(img))
# img_trans.show()
trainset = RGBIRData(data_path)
# generate the idx of each person identity
color_pos, thermal_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)
print(trainset.train_rgb_label)
print(color_pos)
