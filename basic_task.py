# -*- coding : utf-8 -*-
# @FileName  : basic_task.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Apr 03, 2022
# @Github    : https://github.com/songrise
# @Description: script for basic task (task 1-5)

# %% import modules
import PIL.Image as Image
from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
import os
# import tensorflow as tf
import cv2
import sklearn as sk
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import skimage
# %% Constants Definition
BASE_DIR = "./dataset/"
IMG_H = IMG_W = 128
EXAMPLE_ID = 42
RND_SEED = 42
VAL_SIZE = 0.2
TEST_START = 9000
EPOCH = 100
BATCH_SIZE = 256
ALPHA = 1e-4  # learning rate
N_CHAR = 499  # max id of characters
LABEL = {'id_1': 0, 'font_1': 1, 'c_1': 2, 'r_1': 3,
         'id_2': 4, 'font_2': 5, 'c_2': 6, 'r_2': 7}
MODEL_PATH = "model.h5"
# set random seed
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)


def s(a): plt.imshow(a)
# %% Task 1 of the basic task
# load image


def load_all(base: str):
    """
    Input: base, base directory of dataset
    Return: all_img: Ndarray, [N,H,W]
            all_label: Ndarray, [N,8]
    """
    # load all image with label
    label_csv = np.loadtxt(base+"/label.csv", dtype=str,
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
    return all_img, all_labels


all_img, all_labels = load_all(BASE_DIR)


# show an example
example_img = all_img[EXAMPLE_ID]
plt.title("Task 1: Example Image")
plt.imshow(example_img, cmap='gray')
plt.show()


# %% task 2 of the basic task, train/test/val split
# task 2 of the basic task, train/test/val split
# extract test set first

def split_dataset(all_img, all_labels, val_size=0.2, random_state=42):
    """
    Input: all_img: Ndarray, [N,H,W]
            all_labels: Ndarray, [N,8]
            test_size: float, size of val set (portion of all_img)
            random_state: int, random seed
    """

    test_X = all_img[TEST_START:]
    test_Y = all_labels[TEST_START:]
    all_img = all_img[:TEST_START]
    all_labels = all_labels[:TEST_START]
    # split train/val set
    train_X, val_X, train_Y, val_Y = train_test_split(
        all_img, all_labels, test_size=val_size, random_state=random_state)
    return train_X, val_X, test_X, train_Y, val_Y, test_Y


train_X, val_X, test_X, train_Y, val_Y, test_Y = split_dataset(
    all_img, all_labels, VAL_SIZE, RND_SEED)

print("Task 2: Train/Val split")
print("Train size:", train_X.shape[0])
print("Val size:", val_X.shape[0])


# %%baseline of Task 3
# Task 3: baseline of Task 3

# feature engineering on the input image
# extract two characters from the image

def extract_char(imgs, labels):
    """
    extract two characters from the image, and stack them along the channel axis
    Input: Ndarray of shape (N, H, W), all images
    Output: Ndarray of shape (N, H//2, W//2, 2), split images
    """
    imgs = np.reshape(imgs, (-1, IMG_H, IMG_W, 1))
    r_1, c_1, r_2, c_2 = labels[:, LABEL['r_1']], labels[:,
                                                         LABEL['c_1']], labels[:, LABEL['r_2']], labels[:, LABEL['c_2']]
    char_1 = []
    char_2 = []
    for i in range(imgs.shape[0]):
        char_1.append(imgs[i, c_1[i]:c_1[i]+64, r_1[i]:r_1[i]+64])
        char_2.append(imgs[i, c_2[i]:c_2[i]+64, r_2[i]:r_2[i]+64])
    char_1 = np.array(char_1)
    char_2 = np.array(char_2)
    # normalize the image
    char_1 = char_1/255.
    char_2 = char_2/255.
    # stack the two characters along the channel axis
    imgs_split = np.concatenate((char_1, char_2), axis=3)
    return imgs_split


train_X, test_X, val_X = extract_char(
    train_X, train_Y), extract_char(test_X, test_Y), extract_char(val_X, val_Y)

# First, extract the labels for all dataset
# we only need the two id for the basic task as labels


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


train_Y, val_Y, test_Y = one_hot_encode(
    train_Y), one_hot_encode(val_Y), one_hot_encode(test_Y)


# %%
def pipeline(raw_imgs, raw_labels):
    """
    ! FOR MARKING PURPOSE
    call this function to preprocess the imgs and its corresponding labels
    Input: raw_imgs: [N,H,W]
           raw_labels [N, 8]
    Output:
            x: [N, H//2, W//2, 2]
            y: [N, 2, N_CHAR+1]
            Notice that you can only pass one image into the model
            So you have to split them when evaluation
            Such as pred = model(x[...,0]), instead of model(x)
            The same for labels
    """
    x = extract_char(raw_imgs, raw_labels)
    y = extract_char_ids(raw_labels)
    y = one_hot_encode(y)
    return x, y

# %% Task 3 Cont. Define Classifier
# define the classifier


def get_model():
    input_ = Input(shape=(IMG_H//2, IMG_W//2, 1))
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    pred = Dense(N_CHAR+1, activation='softmax')(x)
    model = Model(inputs=input_, outputs=[pred])
    model.compile(loss='categorical_crossentropy')
    return model


model = get_model()


# %% Task 3 Cont. Train
# train the model


def augment_img(imgs):
    """
    randomly apply the following augmentations on batch of images
    1. translation
    2. erosion
    3. dilation
    4. shear
    5. noise
    6. combination of all
    Input: imgs: Ndarray of shape (N, H, W, 2), all images
    Output: Ndarray of shape (N, H, W, 2), all images
    """
    ch_1, ch_2 = imgs[..., 0].copy(), imgs[..., 1].copy()
    # # translate the image
    # tr_1, tr_2 = np.ones_like(ch_1), np.ones_like(ch_2)
    # offset = np.random.randint(-10, 10, size=(2,))
    # tr_1[:, offset[0]:] = ch_1[:, :-offset[0]]
    # tr_2[:, offset[1]:] = ch_2[:, :-offset[1]]
    # # erode the image

    kernel = np.ones((2, 2))
    # erode the image
    eroded_img = []
    for i in range(imgs.shape[0]):
        er_1 = cv2.erode(ch_1[i], kernel, iterations=2)
        er_2 = cv2.erode(ch_2[i], kernel, iterations=2)
        eroded_img.append(np.concatenate(
            (er_1[..., np.newaxis], er_2[..., np.newaxis]), axis=2))
    eroded_img = np.array(eroded_img)

    # dilate the image
    dilated_img = []
    for i in range(imgs.shape[0]):
        dil_1 = cv2.dilate(ch_1[i], kernel, iterations=2)
        dil_2 = cv2.dilate(ch_2[i], kernel, iterations=2)
        dilated_img.append(np.concatenate(
            (dil_1[..., np.newaxis], dil_2[..., np.newaxis]), axis=2))
    dilated_img = np.array(dilated_img)

    # shear the image
    sheared_img = []
    for i in range(imgs.shape[0]):
        scale = np.random.randint(-15, 15) / 100.
        shear_mat = np.array([[1, scale, 0], [scale, 1, 0]], dtype=np.float32)
        shear_1 = cv2.warpAffine(ch_1[i], shear_mat, (IMG_W//2, IMG_H//2))
        shear_2 = cv2.warpAffine(ch_2[i], shear_mat, (IMG_W//2, IMG_H//2))
        sheared_img.append(np.concatenate(
            (shear_1[..., np.newaxis], shear_2[..., np.newaxis]), axis=2))
    sheared_img = np.array(sheared_img)
    # add noise
    # randomly generate gaussian noise
    noise_1 = np.random.normal(0, 0.1, size=ch_1.shape)
    noise_2 = np.random.normal(0, 0.1, size=ch_2.shape)
    noisy_1 = ch_1 + noise_1
    noisy_2 = ch_2 + noise_2
    noisy_imgs = np.concatenate(
        (noisy_1[..., np.newaxis], noisy_2[..., np.newaxis]), axis=3)

    # combine shear, dialate, translate
    combined_img = []
    for i in range(imgs.shape[0]):
        scale = np.random.randint(-15, 15) / 100.
        shear_mat = np.array([[1, scale, 0], [scale, 1, 0]], dtype=np.float32)
        shear_1 = cv2.warpAffine(ch_1[i], shear_mat, (IMG_W//2, IMG_H//2))
        shear_2 = cv2.warpAffine(ch_2[i], shear_mat, (IMG_W//2, IMG_H//2))
        combined_img.append(np.concatenate(
            (shear_1[..., np.newaxis], shear_2[..., np.newaxis]), axis=2))
    combined_img = np.array(combined_img)
    # dilate
    combined_img_res = []
    for i in range(imgs.shape[0]):
        dil_1 = cv2.dilate(combined_img[i, :, :, 0], kernel, iterations=2)
        dil_2 = cv2.dilate(combined_img[i, :, :, 1], kernel, iterations=2)
        combined_img_res.append(np.concatenate(
            (dil_1[..., np.newaxis], dil_2[..., np.newaxis]), axis=2))
    combined_img_res = np.array(combined_img_res)

    return imgs, eroded_img, dilated_img, sheared_img, noisy_imgs, combined_img_res


def bachify(X, Y, batchsize, augment=True):
    """
    Input: X: Ndarray of shape (N, H, W, C), all images
           Y: Ndarray of shape (N, 2, N_CHAR), all labels
           batchsize: int, batch size
    Output: Ndarray of shape (batchsize, H, W, C), batch images
            Ndarray of shape (batchsize, 2, N_CHAR), batch labels
    """
    n_batch = X.shape[0]//batchsize
    for i in range(n_batch):
        if augment:
            raw, ero, dil, she, nsy, com = augment_img(
                X[i*batchsize:(i+1)*batchsize])
        yield np.concatenate((raw, ero, dil, she, nsy, com), axis=0), np.tile(Y[i*batchsize:(i+1)*batchsize], (6, 1, 1))


def validation(model, val_X, val_Y):
    """
    Input: model: keras model
           val_X: Ndarray of shape (N, H, W, C), all images
           val_Y: Ndarray of shape (N, 2, N_CHAR), all labels
    Output: Ndarray of shape (N, 2, N_CHAR), all labels
    """
    pred_1, pred_2 = model(val_X[..., 0], training=False), model(
        val_X[..., 1], training=False)
    pred_1_type = tf.argmax(pred_1, axis=1)
    pred_2_type = tf.argmax(pred_2, axis=1)

    gt_1_type = tf.argmax(val_Y[:, 0, :], axis=1)
    gt_2_type = tf.argmax(val_Y[:, 1, :], axis=1)
    acc_1 = tf.reduce_mean(
        tf.cast(tf.equal(pred_1_type, gt_1_type), tf.float32))
    acc_2 = tf.reduce_mean(
        tf.cast(tf.equal(pred_2_type, gt_2_type), tf.float32))
    print('Accuracy of the first type: {}'.format(acc_1))
    print('Accuracy of the second type: {}'.format(acc_2))
    print("Overall accuracy: {}".format((acc_1+acc_2)/2))


def log_val_acc():
    global model, val_X, val_Y
    validation(model, val_X, val_Y)


def train(model, train_X, train_Y, n_epochs, batch_size):
    """
    optimize the model using Adam optimizer, two patches of the image are used in one iteration
    Input: n_epochs: int, number of epochs
           batch_size: int, batch size
    """
    print("--Start training--")
    optimizer = tf.keras.optimizers.Adam(lr=ALPHA)
    for i_epoch in range(n_epochs):
        for batch_X, batch_Y in bachify(train_X, train_Y, batch_size, augment=True):
            with tf.GradientTape() as tape:
                pred_1, pred_2 = model(batch_X[..., 0], training=True), model(
                    batch_X[..., 1], training=True)
                loss_1 = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(batch_Y[:, 0, :], pred_1))
                loss_2 = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(batch_Y[:, 1, :], pred_2))
                loss = loss_1 + loss_2
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i_epoch % 10 == 0:
            log_val_acc()
            print("---------------")
        print('Epoch: {}, Loss: {}'.format(i_epoch, loss))
    print("--Training finished--")
    # save weights
    model.save_weights(MODEL_PATH)
    print("Model saved to {}".format(MODEL_PATH))


try:
    model.load_weights(MODEL_PATH)
except:
    model = get_model()
    model.summary()
    train(model, train_X, train_Y, EPOCH, BATCH_SIZE)

train(model, train_X, train_Y, EPOCH, BATCH_SIZE)
validation(model, val_X, val_Y)
