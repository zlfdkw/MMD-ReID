from cgi import print_arguments
from unittest import loader
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RGBIRData(Dataset):
    def __init__(self, data_dir, transform=None, colorIndex = None, thermalIndex = None):
        
        # data_dir = "C:\data\dataset\RGBNT\\rgbir"

        self.train_rgb_img = np.load(data_dir + "\\train_rgb_img.npy")
        self.train_rgb_label = np.load(data_dir + "\\train_rgb_label.npy")
        
        self.train_ir_img = np.load(data_dir + "\\train_ir_img.npy")
        self.train_ir_label = np.load(data_dir + "\\train_ir_label.npy")

        self.transform = transform
        self.cInex = colorIndex
        self.thermalIndex = thermalIndex

    def __getitem__(self, index):
        
        img1,  target1 = self.train_rgb_img[self.cIndex[index]],  self.train_rgb_label[self.cIndex[index]]
        img2,  target2 = self.train_ir_img[self.tIndex[index]], self.train_ir_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2
    
    def __len__(self):
        return len(self.train_rgb_img)

if __name__ == "__main__":
    loader = RGBIRData(data_dir="C:\data\dataset\RGBNT\\rgbir")
    train_rgb_img = loader.train_rgb_img
    train_rgb_label = loader.train_rgb_label
    train_ir_img = loader.train_ir_img
    train_ir_label = loader.train_ir_label

    img1 = transforms.ToPILImage()(train_rgb_img[-1])
    img1.show()
    print(train_rgb_label[-1])
    
    
    

    # img2 = transforms.ToPILImage()(train_ir_img[-1])
    # print(len(train_ir_img))

    
