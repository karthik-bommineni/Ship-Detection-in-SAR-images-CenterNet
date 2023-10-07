#calculate the mean and std for dataset
#The mean and std will be used in src/lib/datasets/dataset/oxfordhand.py line17-20
#The size of images in dataset must be the same, if it is not same, we can use reshape_images.py to change the size

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
import imageio

# -*- coding: utf-8 -*-
from PIL import Image
import os


def image_resize(image_path):  # 统一图片尺寸
    #print('============>>修改图片尺寸')
    for img_name in os.listdir(image_path):

        img_path = image_path + "/" + img_name  # 获取该图片全称
        image = Image.open(img_path)  # 打开特定一张图片
        image = image.resize((512, 512))  # 设置需要转换的图片大小
        # process the 1 channel image
        # image.save(new_path + '/' + img_name)
    # print("end the processing!")

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
filepath = "/home/wulele/RotateSparseRCNN/projects/SparseRCNN/datasets/UCAS_AOD/JPEGImages/" # 数据集目录
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    image = Image.open(os.path.join(filepath, filename))  # 打开特定一张图片
    img = image.resize((512, 512))  # 设置需要转换的图片大小
    img = np.array(img)
    #img = imageio.imread(os.path.join(filepath, filename))
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(pathDir) * 512 * 512  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    print('当前图像为：', filename)
    img = imageio.imread(os.path.join(filepath, filename))
    R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
