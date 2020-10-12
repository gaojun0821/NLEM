import sys
import os
import time
import random

import  cv2
import numpy as np
from skimage.transform import resize,rescale
from skimage.io import imsave
import utils.util as util
import tensorflow as tf
from gan_model import generator_model,discriminator_model,GAN,pretrain_g,build_discriminator,needle_class
from utils.data import load_train_data,load_test_data
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras import backend as K
from keras.models import Model



K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 256
img_cols = 256

n_rounds = 30
batch_size = 8
rounds_for_evaluation=range(n_rounds)


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


print('-' * 30)
print('Loading and preprocessing train data...')
print('-' * 30)
imgs_train, shafts_mask_train ,tips_mask_train,needle_label_train = load_train_data()


imgs_train = preprocess(imgs_train)
shafts_mask_train = preprocess(shafts_mask_train)
tips_mask_train = preprocess(tips_mask_train)

imgs_train = imgs_train.astype('float32')
mean = np.mean(imgs_train)  # mean for data centering
std = np.std(imgs_train)  # std for data normalization

imgs_train -= mean
imgs_train /= std

shafts_mask_train = shafts_mask_train.astype('float32')
shafts_mask_train /= 255.  # scale masks to [0, 1]

tips_mask_train = tips_mask_train.astype('float32')
tips_mask_train /= 255.  # scale masks to [0, 1]

# set training and validation dataset
val_ratio= 0.2
n_all_imgs=imgs_train.shape[0]
n_train_imgs=int((1-val_ratio)*n_all_imgs)
train_indices=np.random.choice(n_all_imgs,n_train_imgs,replace=False)
train_batch_fetcher=util.TrainBatchFetcher(imgs_train[train_indices,...], shafts_mask_train[train_indices,...],tips_mask_train[train_indices,...],needle_label_train[train_indices,...],batch_size)
val_imgs, val_shaft_masks,val_tip_masks,val_labels=imgs_train[np.delete(range(n_all_imgs),train_indices),...], shafts_mask_train[np.delete(range(n_all_imgs),train_indices),...], tips_mask_train[np.delete(range(n_all_imgs),train_indices),...], needle_label_train[np.delete(range(n_all_imgs),train_indices),...]

train_classifer_batch_fetcher=util.Classifer_TrainBatchFetcher(imgs_train[train_indices,...], shafts_mask_train[train_indices,...],tips_mask_train[train_indices,...],needle_label_train[train_indices,...],batch_size*4)


# create networks
alpha_recip = 1./5
g = generator_model()
generator=pretrain_g(g, 2e-4)
d,d_out_shape= build_discriminator(1e-4)
gan = GAN(g,d,alpha_recip,1e-4)
needle_classfier = needle_class(g,5e-4)


schedules={'lr_decay':{},
           'd_step_decay':{},
           'g_step_decay':{}}

scheduler=util.Scheduler(n_train_imgs//batch_size , n_train_imgs//batch_size, schedules, init_lr) if alpha_recip>0 else util.Scheduler(0, n_train_imgs//batch_size, schedules, init_lr)

util.make_trainable(d, False)
# Start training l
for n_round in range(n_rounds):

    # train D
    util.make_trainable(d, True)
    for i in range(scheduler.get_dsteps()):
        real_imgs, shaft_masks ,tip_masks,_= next(train_batch_fetcher)
        fake_shaft_masks, fake_tip_masks,_= g.predict(real_imgs, batch_size=batch_size)
        d_x_batch, d_y_batch = util.input2discriminator(real_imgs, shaft_masks,tip_masks,fake_shaft_masks, fake_tip_masks , d_out_shape)
        d_loss, d_acc = d.train_on_batch(d_x_batch, d_y_batch)
        print('d_loss = %f,d_acc = %f'%(d_loss,d_acc))

    #train G
    g.summary()
    util.make_trainable(d, False)
    for i in range(scheduler.get_gsteps()):
        real_imgs, shaft_masks, tip_masks,real_labels= next(train_batch_fetcher)
        g_x_batch, g_y_batch=util.input2gan(real_imgs, shaft_masks,tip_masks ,real_labels,d_out_shape)
        g_loss, g_acc = gan.train_on_batch(g_x_batch, g_y_batch)
        print('g_loss = %f,g_acc = %f' % (g_loss, g_acc))


    if n_round >=10:
        #train Classifer
        util.make_g_net_trainable(g, False)
        g.summary()
        for i in range(scheduler.get_g_class_steps()):
            real_imgs, shaft_masks, tip_masks, real_labels= next(train_classifer_batch_fetcher)
            g_x_batch, g_y_batch=util.input2needleClassifer(real_imgs, shaft_masks, tip_masks, real_labels)
            classifer_loss, classifer_acc = needle_classfier.train_on_batch(g_x_batch, g_y_batch)
            print('classifer_loss = %f,classifer_acc = %f' % (classifer_loss, classifer_acc))
        util.make_g_net_trainable(g, True)
        g.summary()


    # evaluate on validation set
    if n_round in rounds_for_evaluation:

        # D
        fake_shaft_masks, fake_tip_masks, _ = g.predict(val_imgs, batch_size=batch_size)
        d_x_test, d_y_test = util.input2discriminator(val_imgs, val_shaft_masks, val_tip_masks, fake_shaft_masks,
                                                       fake_tip_masks, d_out_shape)

        loss_value, acc = d.evaluate(d_x_test, d_y_test, batch_size=batch_size, verbose=0)
        util.print_metrics(n_round + 1, loss=loss_value, acc=acc, type='D')
        D_LOSS.append(loss_value)
        # G
        gan_x_test, gan_y_test = util.input2gan(val_imgs, val_shaft_masks, val_tip_masks, val_labels, d_out_shape)
        loss_value, acc = gan.evaluate(gan_x_test, gan_y_test, batch_size=1, verbose=0)
        util.print_metrics(n_round + 1, acc=acc, loss=loss_value, type='GAN')

        G_LOSS.append(loss_value)

        # save the weights
        g.save(os.path.join('../saved_model/', "g_model_{}_{}.h5".format(n_round,loss_value)))


    # update step sizes, learning rates
    scheduler.update_steps(n_round)
    K.set_value(d.optimizer.lr, scheduler.get_lr())
    K.set_value(gan.optimizer.lr, scheduler.get_lr())




#######################   test       ################################################
g = generator_model()
g.summary()

print('-'*30)
print('Loading and preprocessing test data...')
print('-'*30)
imgs_test, imgs_id_test = load_test_data()
imgs_test = preprocess(imgs_test)

mean = np.mean(imgs_test)  # mean for data centering
std = np.std(imgs_test)  # std for data normalization

imgs_test = imgs_test.astype('float32')
imgs_test -= mean
imgs_test /= std

print('-'*30)
print('Loading saved weights...')
print('-'*30)
g.load_weights('../saved_model/best.h5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)

imgs_mask_test = g.predict(imgs_test, batch_size=1,verbose=1)

print('-' * 30)
print('Saving predicted masks to files...')
print('-' * 30)
pred_dir = '../examples/'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for image, image_id in zip(imgs_mask_test[0], imgs_id_test):
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, str(image_id) + '_shaft_pred.png'), image)

for image, image_id in zip(imgs_mask_test[1], imgs_id_test):
    image = (image[:, :, 0] * 255.).astype(np.uint8)
    imsave(os.path.join(pred_dir, str(image_id) + '_tip_pred.png'), image)



