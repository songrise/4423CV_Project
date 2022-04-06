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
import keras
# %% Constants Definition
BASE_DIR = "./dataset/"
IMG_H = IMG_W = 128
EXAMPLE_ID = 42
RND_SEED = 42
VAL_SIZE = 0.2
TEST_START = 9000
N_CHAR = 499  # max id of characters
LABEL = {'id_1': 0, 'font_1': 1, 'c_1': 2, 'r_1': 3,
         'id_2': 4, 'font_2': 5, 'c_2': 6, 'r_2': 7}
# %% Task 1 of the basic task
# load image

# load all image with label
label_csv = np.loadtxt(BASE_DIR+"/label.csv", dtype=str,
                       delimiter=",", skiprows=1)

idx = label_csv[:, 0]
labels = label_csv[:, 1:]
labels = np.array(labels, dtype=np.int32)
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
plt.imshow(example_img, cmap='gray')
plt.show()


# %% task 2 of the basic task, train/test/val split
# task 2 of the basic task, train/test/val split
# extract test set first
test_X = all_img[TEST_START:]
test_Y = all_labels[TEST_START:]
all_img = all_img[:TEST_START]
all_labels = all_labels[:TEST_START]
# split train/val set
train_X, val_X, train_Y, val_Y = train_test_split(
    all_img, all_labels, test_size=VAL_SIZE, random_state=RND_SEED)
print("Task 2: Train/Val split")
print("Train size:", train_X.shape[0])
print("Val size:", val_X.shape[0])


# %%baseline of Task 3
# Task 3: baseline of Task 3
# First, extract the labels for all dataset
# we only need the two id for the basic task
def extract_char_ids(labels):
    """
    Input: Ndarray of shape (N, 8), all labels
    Output: Ndarray of shape (N, 2), only id_1 and id_2
    """
    id_1 = labels[:, LABEL['id_1']]
    id_2 = labels[:, LABEL['id_2']]
    id_1 = np.reshape(id_1, (-1, 1))
    id_2 = np.reshape(id_2, (-1, 1))
    label = np.concatenate((id_1, id_2), axis=1)
    return label


train_Y, val_Y, test_Y = extract_char_ids(
    train_Y), extract_char_ids(val_Y), extract_char_ids(test_Y)


def one_hot_encode(labels):
    """
    One hot encoding for classification
    Input: Ndarray [N, 2]
    Output Ndarray [N, 2, N_CHAR]
    """
    n_type = N_CHAR+1
    labels = np.reshape(labels, (-1, 2))
    one_hot = np.zeros((labels.shape[0], labels.shape[1], n_type))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            one_hot[i, j, labels[i, j]] = 1
    return one_hot

# feature engineering on the input image
# todo: find convex hull
# spilt the image into 4 parts and stack them


def split_img(imgs):
    """
    Split an image into 4 parts
    Input: Ndarray of shape (N, H, W), all images
    Output: Ndarray of shape (N, H/2, W/2, 4), split images
    """
    imgs = np.reshape(imgs, (-1, IMG_H, IMG_W, 1))
    upper_left = imgs[:, :IMG_H//2, :IMG_W//2, :]
    upper_right = imgs[:, :IMG_H//2, IMG_W//2:, :]
    lower_left = imgs[:, IMG_H//2:, :IMG_W//2, :]
    lower_right = imgs[:, IMG_H//2:, IMG_W//2:, :]
    imgs_split = np.concatenate(
        (upper_left, upper_right, lower_left, lower_right), axis=3)
    return imgs_split


train_X, val_X, val_Y = split_img(train_X), split_img(val_X), split_img(val_Y)
# todo visualization the split image
# plt.imshow(train_X[0, :, :, 1], cmap='gray')
# %% Task 3 Cont. Define Classifier
# define the classifier


def get_model():
    model = keras.Sequential()
