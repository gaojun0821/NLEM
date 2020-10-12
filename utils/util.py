import os
import sys

from PIL import Image, ImageEnhance
from keras.preprocessing.image import Iterator
from scipy.ndimage import rotate
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
import random


def print_metrics(itr, **kargs):
    print("*** Round {}  ====> ".format(itr)),
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value)),
    print("")
    sys.stdout.flush()


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def make_g_net_trainable(net, val):
    for l in net.layers:
        l.trainable = val

    if val==False:
        net.get_layer('classifer_layer_1').trainable = True
        net.get_layer('classifer_layer_3').trainable = True
        net.get_layer('classifer_layer_4').trainable = True
        net.get_layer('classifer_layer_5').trainable = True
    else:
        net.get_layer('classifer_layer_1').trainable = False
        net.get_layer('classifer_layer_3').trainable = False
        net.get_layer('classifer_layer_4').trainable = False
        net.get_layer('classifer_layer_5').trainable = False



def discriminator_shape(n, d_out_shape):
    if len(d_out_shape) == 1:  # image gan
        return (n, d_out_shape[0])
    elif len(d_out_shape) == 3:  # pixel, patch gan
        return (n, d_out_shape[0], d_out_shape[1], d_out_shape[2])
    return None


def input2discriminator(real_img_patches, real_shaft_mask_patches, real_tip_mask_patches, fake_shaft_mask_patches,fake_tip_mask_patches, d_out_shape):
    real = np.concatenate((real_img_patches, real_shaft_mask_patches,real_tip_mask_patches), axis=3)
    fake = np.concatenate((real_img_patches, fake_shaft_mask_patches,fake_tip_mask_patches), axis=3)

    d_x_batch = np.concatenate((real, fake), axis=0)

    # real : 1, fake : 0
    d_y_batch = np.ones(discriminator_shape(d_x_batch.shape[0], d_out_shape))
    d_y_batch[real.shape[0]:, ...] = 0

    return d_x_batch, d_y_batch


def input2gan(real_img_patches, real_shaft_mask_patches,real_tip_mask_patches,real_labels_patches,d_out_shape):
    g_x_batch = [real_img_patches, real_shaft_mask_patches,real_tip_mask_patches,real_labels_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch = np.ones(discriminator_shape(real_shaft_mask_patches.shape[0], d_out_shape))
    return g_x_batch, g_y_batch




class TrainBatchFetcher(Iterator):
    """
    fetch batch of original images and vessel images
    """

    def __init__(self, train_imgs, train_shaft_masks , train_tip_masks ,train_labels,batch_size):
        self.train_imgs = train_imgs
        self.train_shaft_masks = train_shaft_masks
        self.train_tip_masks = train_tip_masks
        self.train_labels = train_labels
        self.n_train_imgs = self.train_imgs.shape[0]
        self.batch_size = batch_size

    def next(self):
        indices = list(np.random.choice(self.n_train_imgs, self.batch_size))
        return self.train_imgs[indices, :, :, :], self.train_shaft_masks[indices, :, :, :],self.train_tip_masks[indices, :, :, :],self.train_labels[indices]



class Classifer_TrainBatchFetcher(Iterator):
    """
    fetch batch of original images and vessel images
    """

    def __init__(self, train_imgs, train_shaft_masks,train_tip_masks,train_labels,batch_size):
        self.train_imgs = train_imgs
        self.train_shaft_masks = train_shaft_masks
        self.train_tip_masks = train_tip_masks
        self.train_labels = train_labels
        self.n_train_imgs = self.train_imgs.shape[0]
        self.batch_size = batch_size

    def next(self):
        indices = list(np.random.choice(self.n_train_imgs, self.batch_size))
        return self.train_imgs[indices, :, :, :], self.train_shaft_masks[indices, :, :, :],self.train_tip_masks[indices, :, :, :],self.train_labels[indices]


def input2needleClassifer(real_img_patches, real_shaft_mask_patches,real_tip_mask_patches,real_labels_patches):
    g_x_batch = [real_img_patches, real_shaft_mask_patches,real_tip_mask_patches,real_labels_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch = real_labels_patches
    return g_x_batch, g_y_batch


class Scheduler:
    def __init__(self, n_itrs_per_epoch_d, n_itrs_per_epoch_g, schedules, init_lr):
        self.schedules = schedules
        self.init_dsteps = n_itrs_per_epoch_d
        self.init_gsteps = n_itrs_per_epoch_g
        self.init_lr = init_lr
        self.dsteps = self.init_dsteps
        self.gsteps = self.init_gsteps
        self.lr = self.init_lr

    def get_dsteps(self):
        return int(self.dsteps)

    def get_gsteps(self):
        return int(self.gsteps)

    def get_g_class_steps(self):
        return int(self.gsteps/4)

    def get_lr(self):
        return float(self.lr)

    def update_steps(self, n_round):
        key = str(n_round)
        if key in self.schedules['lr_decay']:
            self.lr = self.init_lr * self.schedules['lr_decay'][key]
        if key in self.schedules['d_step_decay']:
            self.dsteps = max(int(self.init_dsteps * self.schedules['d_step_decay'][key]), 1)
        if key in self.schedules['g_step_decay']:
            self.gsteps = max(int(self.init_gsteps * self.schedules['g_step_decay'][key]), 1)