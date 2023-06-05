# generate augmented images for training 生成用于训练的增强图像（扩展数据集）
#    - input/: images with original palette
#    - output/: images with new palette
#    - old_palette/: pickled files of original palette 
#    - new_palette/: pickled files of new palette 
import collections
import pathlib
import random
import pickle
from typing import Dict, Tuple, Sequence

import cv2
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
import cv2
import matplotlib
import os
# def augment_image(img, title, hue_shift):
#     #plt.imshow(img)
#     #plt.title(f"Original {title} (in RGB)")
#     #plt.show()

#     # RGB -> HSV -> hue-shift 转换图像颜色的编码
#     #H（色调）S（饱和度）V（明度）RGB面向硬件，HSV面向用户，这里应该是进行了一些转换又变为RGB
#     img_HSV = matplotlib.colors.rgb_to_hsv(img)
#     a_2d_index = np.array([[1,0,0] for _ in range(img_HSV.shape[1])]).astype('bool')
#     img_HSV[:, a_2d_index] = (img_HSV[:, a_2d_index] + hue_shift) % 1

#     new_img = matplotlib.colors.hsv_to_rgb(img_HSV).astype(int)
#     #plt.imshow(new_img)
#     #plt.title(f"New {title} (in RGB)")
#     #plt.show()

#     # fixed original luminance固定原始亮度
#     img = img.astype(np.float) / 255.0 #astype函数用于转换数据的数据类型
#     new_img = new_img.astype(np.float) / 255.0
#     ori_img_LAB = rgb2lab(img)#from skimage.color import rgb2lab, lab2rgb
#     new_img_LAB = rgb2lab(new_img)
#     new_img_LAB[:, :, 0] = ori_img_LAB[:, :, 0]
#     new_img_augmented = (lab2rgb(new_img_LAB)*255.0).astype(int)
#     plt.imshow(new_img_augmented)
#     plt.title(f"New {title} (in RGB) with Fixed Luminance")
#     plt.show()
#     plt.close()

#     return new_img_augmented

# import os
# import pathlib
# for i in range(0,22):
#     pathlist=[]
#     pathlist = pathlib.Path("data/unlabeled_data/origin_palette/"+str(i)).glob("*.pkl")
#     count=0
#     for j, path in enumerate(pathlist):
#         count=count+1
#     print(count) #dirpath子目录路径，file_count子目录中的文件个数

# print("target")
# for i in range(0,22):
#     pathlist=[]
#     pathlist = pathlib.Path("data/unlabeled_data/target_palette/"+str(i)).glob("*.pkl")
#     count=0
#     for j, path in enumerate(pathlist):
#         count=count+1
#     print(count) #dirpath子目录路径，file_count子目录中的文件个数
    
# #pathlist = pathlib.Path("/datasets/dribbble_half/dribbble_half/data").glob("*.png")
# for i in range(0,22):
#     pathlist =[]
#     pathlist = pathlib.Path("data/unlabeled_data/train/"+str(i)).glob("*.jpg")
#     isExists = os.path.exists("data/unlabeled_data/presudo_target/"+str(i))
#     if not isExists:
#         os.makedirs("data/unlabeled_data/presudo_target/"+str(i))
#     for j, path in enumerate(pathlist):
#         print(path)
#         img = cv2.imread(str(path))
#         #im = Image.open(path) # Replace with your image name here
#         #indexed = np.array(im) # Convert to NumPy array to easier access
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#         hue_shift = random.random()#这里应该是随机生成一个色调偏移的值
#         augmented_image = augment_image(img, "Image", hue_shift)

#         #cv2.imwrite(f'train/input/{str(i)}/{path.name}', img)#将原图像输入input

#         cv2.imwrite(f'data/unlabeled_data/presudo_target/{str(i)}/{path.name}', augmented_image)#将color augmentation后的图像输出到output



# for i, path in enumerate(pathlist):#enumerate是列举的意思，i是序号，path是列举出的一个一个元素
#     print(i)
#     print(path)

#     img = cv2.imread(str(path))
#     #im = Image.open(path) # Replace with your image name here
#     #indexed = np.array(im) # Convert to NumPy array to easier access
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#     hue_shift = random.random()#这里应该是随机生成一个色调偏移的值
#     augmented_image = augment_image(img, "Image", hue_shift)

#     cv2.imwrite(f'train/input/{path.name}', img)#将原图像输入input

#     cv2.imwrite(f'train/output/{path.name}', augmented_image)#将color augmentation后的图像输出到output
for i in range(0,22):
    pathlist =[]
    isExists = os.path.exists("data/unlabeled_data/val/"+str(i))
    if not isExists:
        os.makedirs("data/unlabeled_data/val/"+str(i))