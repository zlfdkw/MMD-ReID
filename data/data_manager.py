from cgi import print_arguments
import os
from re import L
from PIL import Image
import numpy as np
import random

def process_query_rgbir(data_path, relabel=False):
    file_path = os.path.join(data_path, "query2")
    
    if not os.path.exists(file_path):
        raise IOError("{} does not exist!".format(file_path))
    
    img_paths = os.listdir(file_path)

    query_img = []
    query_id = []
    query_cam = []

    for img_path in img_paths:
        camid, pid = int(img_path[9]), int(img_path[:4])
        img_path = os.path.join(file_path, img_path)
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_rgbir(data_path, relabel=False):
    file_path = os.path.join(data_path, "bounding_box_test2")

    if not os.path.exists(file_path):
        raise IOError("{} does not exist!".format(file_path))

    img_paths = os.listdir(file_path)

    gallery_img = []
    gallery_id = []
    gallery_cam = []

    for img_path in img_paths:
        camid, pid = int(img_path[9]), int(img_path[:4])
        img_path = os.path.join(file_path, img_path)
        gallery_img.append(img_path)
        gallery_id.append(pid)
        gallery_cam.append(camid)
    
    return gallery_img, np.array(gallery_id), np.array(gallery_cam)



if __name__ == "__main__":
    data_path = "C:/data/dataset/RGBNT/rgbir"
    query_img, query_id, query_cam = process_gallery_rgbir(data_path)
    img = Image.open(query_img[0])
    img.show()
    
    