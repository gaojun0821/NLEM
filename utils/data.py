from __future__ import print_function

import os
import numpy as np
import random

from skimage.io import imsave, imread

data_path = './dataset/'

image_rows = 256
image_cols = 256


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    random.shuffle(images)
    total = len(images) // 3

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    shafts_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    tips_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    needle_label = np.ndarray(total,dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if '_shaft_mask' in image_name:
            continue

        if '_tip_mask' in image_name:
            continue

        shaft_mask_name = image_name[:-4] + '_shaft_mask.jpg'
        tip_mask_name = image_name[:-4] + '_tip_mask.jpg'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        shaft_mask = imread(os.path.join(train_data_path, shaft_mask_name), as_grey=True)
        tip_mask = imread(os.path.join(train_data_path, tip_mask_name), as_grey=True)
        img = np.array([img])
        shaft_mask = np.array([shaft_mask])
        tip_mask = np.array([tip_mask])
        print(image_name)
        imgs[i] = img
        shafts_mask[i] = shaft_mask
        tips_mask[i] = tip_mask

        if 'no_' in image_name:
            needle_label[i] = 0
        else:
            needle_label[i] = 1

        if i % 1 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('shafts_mask_train.npy', shafts_mask)
    np.save('tips_mask_train.npy', tips_mask)
    np.save('needle_label.npy', needle_label)

    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('../utils/imgs_train.npy')
    shafts_mask_train = np.load('../utils/shafts_mask_train.npy')
    tips_mask_train = np.load('../utils/tips_mask_train.npy')
    needle_label = np.load('../utils/needle_label.npy')
    return imgs_train, shafts_mask_train,tips_mask_train,needle_label


def create_test_data():
    train_data_path = os.path.join('./dataset', 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    i = 0
    for image_name in images:
        img_id = int(image_name.split('_')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')



def load_test_data():
    imgs_test = np.load('../utils/imgs_test.npy')
    imgs_id = np.load('../utils/imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
