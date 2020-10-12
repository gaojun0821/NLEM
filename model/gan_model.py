
import os
from keras.models import Sequential
from skimage.transform import resize,rescale
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Flatten, GlobalAveragePooling2D,Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.losses import binary_crossentropy,MSE
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,Flatten
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
import tensorflow as tf

img_rows = 256
img_cols = 256
n_filters = 4

def generator_model(name='g'):
    """
    generate network based on unet
    """
    # set image specifics
    k = 3  # kernel size
    s = 2  # stride
    img_ch = 1  # image channels
    out_ch = 1  # output channel
    img_height, img_width = img_rows, img_cols
    padding = 'same'

    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16 * n_filters, (k, k), padding=padding,)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)


    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8 * n_filters, (k, k), padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(4 * n_filters, (k, k), padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)


    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(2 * n_filters, (k, k), padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)


    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(up4)

    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k), padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)


    up5 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv10 = Conv2D(n_filters, (k, k), padding=padding)(up5)
    conv10 = BatchNormalization(scale=False, axis=3)(conv10)
    conv10 = Activation('relu')(conv10)

    conv10 = Conv2D(n_filters, (k, k), padding=padding)(conv10)
    conv10 = BatchNormalization(scale=False, axis=3)(conv10)
    conv10 = Activation('relu')(conv10)


    conv11 = MaxPooling2D(pool_size=(s, s),name='classifer_layer_1')(conv5)
    conv11 = Flatten(name='classifer_layer_3')(conv11)
    conv11 = Dense(8, activation='relu',name='classifer_layer_4')(conv11)
    outputs3 = Dense(1, activation='sigmoid',name='classifer_layer_5')(conv11)


    outputs1 = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid',name='needle_shaft_out')(conv9)
    outputs2 = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid',name='needle_tip_out')(conv10)

    g = Model(inputs, outputs =[outputs1,outputs2,outputs3], name=name)

    return g



def build_discriminator(init_lr, name='d'):

    def d_layer(layer_input, filters, f_size=3, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        return d


    img = Input(shape=(img_rows, img_cols,3))
    df = 32
    d1 = d_layer(img, df)
    d2 = d_layer_no_strides(d1, df*2)
    d3 = d_layer(d2, df*4)
    d4 = d_layer_no_strides(d3, df*8)
    d5 = d_layer(d4, df*16)


    validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d5)
    model = Model(img, validity)
    model.compile(loss='mse',optimizer=Adam(lr=init_lr, beta_1=0.9, beta_2=0.999),metrics=['accuracy'])

    return model,model.layers[-1].output_shape[1:]



def needle_class(g,init_lr,name="needle_classfier"):
    img_h, img_w = img_rows, img_cols
    img_ch = 1
    seg_ch = 1

    img = Input(shape= (img_h, img_w, img_ch))
    shaft_mask = Input((img_h, img_w, seg_ch))
    tip_mask = Input((img_h, img_w, seg_ch))
    needle_label = Input(shape=(1,))
    shaft_pre, tip_pre, needle_pred = g(img)

    needle_classfier = Model(inputs=[img,shaft_mask,tip_mask,needle_label],outputs= [needle_pred], name=name)

    def Classifer_loss(y_true, y_pred):
        class_loss = binary_crossentropy(needle_label, needle_pred)

        shaft_mask_flat = K.batch_flatten(shaft_mask)
        fake_shaft_mask_flat = K.batch_flatten(shaft_pre)

        tip_mask_flat = K.batch_flatten(tip_mask)
        fake_tip_mask_flat = K.batch_flatten(tip_pre)

        bce_loss1 = weighted_bce(shaft_mask_flat,fake_shaft_mask_flat)
        bce_loss2 = weighted_bce_tip(tip_mask_flat,fake_tip_mask_flat)

        bce_loss = 0.5*bce_loss1 + 0.5*bce_loss2

        return class_loss + 0.0*bce_loss

    needle_classfier.compile(optimizer=Adam(lr=init_lr, beta_1=0.9, beta_2=0.999), loss=Classifer_loss, metrics=['accuracy'])
    return needle_classfier



def weighted_bce(y_true, y_pred):
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

  coe = ((pt_0 + pt_1)/(2*pt_1 + 1))
  weights = (y_true * coe) + 1.
  bce = K.binary_crossentropy(y_true, y_pred)
  print('bce:',coe)
  weighted_bce = K.mean(bce * weights)
  return weighted_bce

def weighted_bce_tip(y_true, y_pred):
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

  coe = ((pt_0 + pt_1)/pt_1*2)

  weights = (y_true * coe) + 1.
  bce = K.binary_crossentropy(y_true, y_pred)
  print('bce:',coe)
  weighted_bce = K.mean(bce * weights)

  c = tf.square(y_true - y_pred) * coe
  mse = tf.reduce_mean(c)
  return mse




def GAN(g, d, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w = img_rows, img_cols

    img_ch = 1
    seg_ch = 1

    img = Input((img_h, img_w, img_ch))
    shaft_mask = Input((img_h, img_w, seg_ch))
    tip_mask = Input((img_h, img_w, seg_ch))
    needle_label = Input(shape=(1,))

    fake_shaft_mask,fake_tip_mask,pre_labels = g(img)

    fake_pair = Concatenate(axis=3)([img, fake_shaft_mask,fake_tip_mask])

    gan = Model(inputs=[img, shaft_mask,tip_mask,needle_label],outputs= [d(fake_pair)], name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = MSE(y_true_flat, y_pred_flat)


        shaft_mask_flat = K.batch_flatten(shaft_mask)
        fake_shaft_mask_flat = K.batch_flatten(fake_shaft_mask)

        tip_mask_flat = K.batch_flatten(tip_mask)
        fake_tip_mask_flat = K.batch_flatten(fake_tip_mask)

        bce_loss1 = weighted_bce(shaft_mask_flat,fake_shaft_mask_flat)
        bce_loss2 = weighted_bce_tip(tip_mask_flat,fake_tip_mask_flat)

        bce_loss = 0.5*bce_loss1 + 0.5*bce_loss2

        classifer_loss = binary_crossentropy(needle_label, pre_labels)


        return  alpha_recip*L_adv + bce_loss  +  0*classifer_loss



    def shaft_dice(y_true, y_pred):
        return  dice_coef(shaft_mask, fake_shaft_mask)

    def tip_dice(y_true, y_pred):
        return  dice_coef(tip_mask, fake_tip_mask)

    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.9, beta_2=0.999), loss=gan_loss, metrics=['accuracy'])

    return gan


def pretrain_g(g,init_lr):
    img_h, img_w = img_rows, img_cols

    img_ch = 1
    img = Input((img_h, img_w, img_ch))
    generator = Model(img, g(img))

    def g_loss(y_true, y_pred):
        L_seg = binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L_seg

    generator.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=g_loss, metrics=['accuracy'])

    return generator


