# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:02:33 2023

@author: up202111331
"""

import os
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import torch
import numpy as np

import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path(r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images\Negative\00026.jpg'))
torch.manual_seed(0)
data_path = r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images'
diz_class = {'Positive':'Crack','Negative':'No crack'}

np.asarray(orig_img).shape


# Introduction to Surface Crack dataset


def load_images_from_folder(folder,class_name):
    
    images = []
    class_path = os.path.join(folder, class_name)
    for filename in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path,filename))
        images.append(img)
        if len(images)>3:
           break
    
    plt.figure(figsize=(12,12))    
    for img,x in zip(images,range(1,7)):
        plt.subplot(1,4,x)
        plt.title(f'{diz_class[class_name]}')
        plt.axis("off")
        plt.imshow(img)
        
        
load_images_from_folder(data_path,"Positive")
load_images_from_folder(data_path,"Negative")



def plot(imgs, with_orig=True, col_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if col_title is not None:
        for col_idx in range(num_cols-1):
            axs[0, col_idx+1].set(title=col_title[col_idx])
            axs[0, col_idx+1].title.set_size(8)

    plt.tight_layout()


# Pytorch Transformations

# Since the images have very high height and width, there is the need to reduce the dimension before passing it to a neural network.

# For example, we can resize the 227x227 image into 32x32 and 128x128 images.


resized_imgs = [T.Resize(size=size)(orig_img) for size in [32,128]]
plot(resized_imgs,col_title=["32x32","128x128"])


# 2.2 GrayScale

gray_img = T.Grayscale()(orig_img)
plot([gray_img], cmap='gray', col_title=["Gray"])

# 2.3 Normalize

normalized_img = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(T.ToTensor()(orig_img)) 
normalized_img = [T.ToPILImage()(normalized_img)]
plot(normalized_img, col_title=["Standard normalize"])


# 2.4 Random Rotation

rotated_imgs = [T.RandomRotation(degrees=d)(orig_img) for d in range(50,151,50)]
plot(rotated_imgs, col_title=["Rotation 50","Rotation 100","Rotation 150"])

#  2.5 Center Crop
# We crop the central portion of the image using T.CenterCrop method, where the crop size needs to be specified.

center_crops = [T.CenterCrop(size=size)(orig_img) for size in (128,64, 32)]
plot(center_crops,col_title=['128x128','64x64','32x32'])



# 2.6 Random Crop
# Instead of cropping the central part of the image, we crop randomly a portion of the image through T.RandomCrop method, which takes in the output size of the crop as parameter.


random_crops = [T.RandomCrop(size=size)(orig_img) for size in (128,64, 32)]
plot(random_crops,col_title=['128x128','64x64','32x32'])


# GaussianBlur

blurred_imgs = [T.GaussianBlur(kernel_size=(51, 91), sigma=sigma)(orig_img) for sigma in (3,7)]
plot(blurred_imgs,col_title=["sigma=3","sigma=7"])

# 3.1 Gaussian Noise

def add_noise(inputs,noise_factor=0.3):
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy
    
noise_imgs = [add_noise(T.ToTensor()(orig_img),noise_factor) for noise_factor in (0.3,0.6,0.9)]
noise_imgs = [T.ToPILImage()(noise_img) for noise_img in noise_imgs]



# 3.2 Random Blocks

# def add_central_region(img,size=32):
#     h,w = size,size
#     img = np.asarray(img)
#     img_size = img.shape[1] 
#     img[int(img_size/2-h):int(img_size/2+h),int(img_size/2-w):int(img_size/2+w)] = 0
#     img = Image.fromarray(img.astype('uint8'), 'RGB')
#     return img
  
# central_imgs = [add_central_region(orig_img,size=s) for s in (32,64)]
# plot(central_imgs,col_title=["32","64"])


# ColorJitter
# The ColorJitter transform randomly changes the brightness, saturation, and other properties of an image.

jitter = T.ColorJitter(brightness=.5, hue=.3)
jitted_imgs = [jitter(orig_img) for i in range(4)]
plot(jitted_imgs)