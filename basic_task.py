# -*- coding : utf-8 -*-
# @FileName  : basic_task.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Apr 03, 2022
# @Github    : https://github.com/songrise
# @Description: script for basic task (task 1-5)

# %% import modules
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import os
# import tensorflow as tf
import cv2
import sklearn as sk
from sklearn.model_selection import train_test_split
#%% Constants Definition
BASE_DIR = "./dataset/"
EXAMPLE_ID = 42
RND_SEED = 42
VAL_SIZE = 0.2

#%% Task 1 of the basic task
# load image

# load all image with label
label_csv = np.loadtxt(BASE_DIR+"/label.csv", dtype=str,
                       delimiter=",", skiprows=1)

idx = label_csv[:,0]
labels = label_csv[:,1:]
labels = np.array(labels, dtype=np.uint32)
# load all image and convert to numpy array
images = []
for img_name in idx:
    img = Image.open(BASE_DIR+"/images/"+img_name)
    img = np.array(img)
    images.append(img)

all_img = np.array(images, dtype=np.uint8)
all_labels = labels

# show an example
example_img = all_img[EXAMPLE_ID]
plt.title("Task 1: Example Image")
plt.imshow(example_img,cmap='gray')
plt.show()


# %% task 2 of the basic task, train/test/val split
train_X, val_X, train_Y, val_Y = train_test_split(all_img, all_labels, test_size=VAL_SIZE, random_state=RND_SEED)
print("Task 2: Train/Val split")
print("Train size:", train_X.shape[0])
print("Val size:", val_X.shape[0])
# %%
