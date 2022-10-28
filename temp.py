import argparse
from ast import parse
import imp
from importlib import import_module
import os
from xml.dom.expatbuilder import ParseEscape
import numpy as np
import random
import torch
from torch.nn import init
import torch.nn as nn
from torchvision import transforms
from data.data_loader import RGBIRData
from data.data_manager import *
from utils import *
import time
# ir_cameras = ['cam3','cam6']
# data_path = "data_set\SYSU-MM01"
# print(os.path.exists(data_path))
# random.seed(0)


# file_path = os.path.join(data_path,'exp/test_id.txt')
# files_ir = []
# files_ir_all = []

# with open(file_path, 'r') as file:
#     ids = file.read().splitlines()
#     ids = [int(y) for y in ids[0].split(',')]
#     ids = ["%04d" % x for x in ids]

# for id in sorted(ids):
#     for cam in ir_cameras:
#         img_dir = os.path.join(data_path,cam,id)
#         if os.path.isdir(img_dir):
#             new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
#             files_ir.append(random.choice(new_files))
#             files_ir_all.extend(new_files)
# query_img = []
# query_id = []
# query_cam = []
# for img_path in files_ir:
#     camid, pid = int(img_path[-15]), int(img_path[-13:-9])
#     query_img.append(img_path)
#     query_id.append(pid)
#     query_cam.append(camid)

# pid_container = set()
# for img_path in files_ir_all:
#     pid = int(img_path[-13:-9])
#     pid_container.add(pid)
# pid2label = {pid:label for label, pid in enumerate(pid_container)}

# print(pid2label)
# print(query_img)
# print(query_id)
# print(query_cam)

# print(np.array(query_id))
# print(np.array(query_cam))
# print(os.path.exists(file_path))
# print(ids)
# print(sorted(ids))
# print(files_ir)
# print(files_ir_all)

# x = torch.tensor([[1., 2.],
#                   [4., 3.]])
# y = torch.tensor([[4., 7.],
#                   [1., 5.]])
# z = torch.cat((x, y), 1)
# print(z)

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         init.zeros_(m.bias.data)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal_(m.weight.data, 1.0, 0.01)
#         init.zeros_(m.bias.data)

# class Mymodel(nn.Module):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         self.seq = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.seq(x)
#         return x

# model = Mymodel()
# for name in model.state_dict():
#     print(name)
# weights_init_kaiming(model)
# for i in range(1, 4):
#     print(i)
# data_path = "C:\data\dataset\RGBNT\\rgbir\\bounding_box_train"
# img_paths = os.listdir(data_path)
# ids = set()
# print(len(img_paths))
# for img in img_paths:
#     id = int(img[:4])
#     # print(id)
#     ids.add(id)

# print(ids)

# imgs_rgb = np.load(data_path + "\\train_rgb_img.npy")
# lable_rgb = np.load(data_path + "\\train_rgb_label.npy")

# img = transforms.ToPILImage()(imgs_rgb[199])
# print(lable_rgb[199])
# img.show()

# share_net = 2
# if share_net > 1:
#     for i in range(1, 5): 
#         print(i)

# parser = argparse.ArgumentParser(description="test")
# parser.add_argument('--dataset', default='sysu', help='dataset name:rgbdb or sysu')
# parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--model_path', default='save_model/', type=str,
#                     help='model save path')
# parser.add_argument('--log_path', default='log/', type=str,
#                     help='log save path')
# parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
#                     help='log save path')
# args = parser.parse_args()

# # dataset = args.dataset
# # print(dataset)

# # lr = args.lr
# # print(lr * 100)
# if not os.path.isdir(args.log_path + 'sysu_log/'):
#     os.makedirs(args.log_path + 'sysu_log/')
# if not os.path.isdir(args.model_path):
#     os.makedirs(args.model_path)
# if not os.path.isdir(args.vis_log_path):
#     os.makedirs(args.vis_log_path)
data_path = 'C:/data/dataset/RGBNT/rgbir'
print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    # transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])
end = time.time()
# training set
trainset = RGBIRData(data_path, transform=transform_train)
# generate the idx of each person identity
color_pos, thermal_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)

# testing set
query_img, query_label, query_cam = process_query_rgbir(data_path)
gall_img, gall_label, gall_cam = process_gallery_rgbir(data_path)

n_class = len(np.unique(trainset.train_rgb_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format('rgbir'))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_rgb_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_ir_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))