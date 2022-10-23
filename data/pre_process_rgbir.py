import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from RGBIR import DATASET_RGBIR

data_path = 'C:\data\dataset\RGBNT\\rgbir'

if not os.path.exists(data_path):
    raise IOError("{} does not exist".format(data_path))
    
dataset_rgbir = DATASET_RGBIR(data_path, False) 

train_data = dataset_rgbir.train_data

train_img_rgb = []
train_img_ir = []
train_label_rgb = []
train_label_ir =[]

for data in train_data:
    img_path, pid, cam_id = data
    img = Image.open(img_path)
    img_rgb = img.crop((0, 0, 256, 128))
    img_ir = img.crop((256, 0, 512, 128))
    # img_rgb.show()
    # img_ir.show()
    pix_array_rgb = np.array(img_rgb)
    pix_array_ir = np.array(img_ir)

    train_img_rgb.append(pix_array_rgb)
    train_img_ir.append(pix_array_ir)
    train_label_rgb.append(pid)
    train_label_ir.append(pid)

train_img_rgb = np.array(train_img_rgb)
train_img_ir = np.array(train_img_ir)
train_label_rgb = np.array(train_label_rgb)
train_label_ir = np.array(train_label_ir)

# rgb imges
np.save(data_path + '\\train_rgb_img.npy', train_img_rgb)
np.save(data_path + '\\train_rgb_label.npy', train_label_rgb)

# ir imges
np.save(data_path + '\\train_ir_img.npy', train_img_ir)
np.save(data_path + '\\train_ir_label.npy', train_label_ir)


