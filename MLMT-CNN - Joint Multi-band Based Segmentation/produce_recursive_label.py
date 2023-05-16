import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import keras
from keras.layers import *
from keras import optimizers
from keras import metrics
import keras.backend.tensorflow_backend as k
from utils.provider import get_session
from utils.provider import get_img_seg_from_dir
from utils.provider import get_image_arr
from utils.provider import resize_mask
from MLMT_UNet.MLMT_UNet import get_mlmt_unet
from MLMT_UNet.loss import weighted_categorical_crossentropy


full_disk = True ##  True -> full disk and False -> patches

if full_disk:
    from MLMT_UNet.config_full_disk import config as cfg
else:
    from MLMT_UNet.config_patches import config as cfg



## training data dirs (image/label):
train_dir_img_1 = cfg['train_dir_img_1']
train_dir_img_2 = cfg['train_dir_img_2']
train_dir_img_3 = cfg['train_dir_img_3']
train_dir_img_4 = cfg['train_dir_img_4']

train_dir_seg_1 = cfg['train_dir_seg_1']
train_dir_seg_2 = cfg['train_dir_seg_2']
train_dir_seg_3 = cfg['train_dir_seg_3']
train_dir_seg_4 = cfg['train_dir_seg_4']


## validation data dirs (image/label):
val_dir_img_1 = cfg['val_dir_img_1']
val_dir_img_2 = cfg['val_dir_img_2']
val_dir_img_3 = cfg['val_dir_img_3']
val_dir_img_4 = cfg['val_dir_img_4']

val_dir_seg_1 = cfg['val_dir_seg_1']
val_dir_seg_2 = cfg['val_dir_seg_2']
val_dir_seg_3 = cfg['val_dir_seg_3']
val_dir_seg_4 = cfg['val_dir_seg_4']

n_classes = cfg['n_classes']
print('number of classes {}'.format(n_classes))


input_height, input_width = cfg['input_height'], cfg['input_width']


if full_disk:
    output_dir = './recursive_label/full_disk/'
else:
    output_dir = './recursive_label/patches/'

spect_ID_1, spect_ID_2, spect_ID_3, spect_ID_4 = cfg['spect_1'], cfg['spect_2'], cfg['spect_3'], cfg['spect_4']



## get images/labels path lists of both train and val
train_images_1, train_segmentations_1 = get_img_seg_from_dir(train_dir_img_1, train_dir_seg_1)
train_images_2, train_segmentations_2 = get_img_seg_from_dir(train_dir_img_2, train_dir_seg_2)
train_images_3, train_segmentations_3 = get_img_seg_from_dir(train_dir_img_3, train_dir_seg_3)
train_images_4, train_segmentations_4 = get_img_seg_from_dir(train_dir_img_4, train_dir_seg_4)

val_images_1, val_segmentations_1 = get_img_seg_from_dir(val_dir_img_1, val_dir_seg_1)
val_images_2, val_segmentations_2 = get_img_seg_from_dir(val_dir_img_2, val_dir_seg_2)
val_images_3, val_segmentations_3 = get_img_seg_from_dir(val_dir_img_3, val_dir_seg_3)
val_images_4, val_segmentations_4 = get_img_seg_from_dir(val_dir_img_4, val_dir_seg_4)


assert len(train_images_1) == len(train_images_2) == len(train_images_3) == len(train_images_4) == len(train_segmentations_1) == len(train_segmentations_2) == len(train_segmentations_3) == len(train_segmentations_4)
assert len(val_images_1) == len(val_images_2) == len(val_images_3) == len(val_images_4) == len(val_segmentations_1) == len(val_segmentations_2) == len(val_segmentations_3) == len(val_segmentations_4)


train_num_samples = len(train_images_1)
val_num_samples = len(val_images_1)

print ('found {} training and {} validation samples'.format(train_num_samples, val_num_samples))




model = get_mlmt_unet(n_classes=n_classes,
                      input_height = input_height, input_width = input_width,
                      merge='add')

assert cfg['pre_trained_model'] is not ''
try:
    wp = cfg['saving_dir'] + cfg['pre_trained_model']
    model.load_weights(wp)
    print('weights loaded successfully {}'.format(wp))
except Exception as e:
    print(e)

adam = optimizers.Adam(lr=cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

class_weights = cfg['class_weights']
model.compile(loss = [weighted_categorical_crossentropy(class_weights), weighted_categorical_crossentropy(class_weights),
                      weighted_categorical_crossentropy(class_weights), weighted_categorical_crossentropy(class_weights)],
              optimizer=adam,
              metrics=[metrics.categorical_accuracy])





def prodcue_recursive_label(dir_img_1,
                            dir_img_2,
                            dir_img_3,
                            dir_img_4,
                            images_1, segmentations_1,
                            images_2, segmentations_2,
                            images_3, segmentations_3,
                            images_4, segmentations_4,
                            input_width, input_height,
                            spect_ID_1, spect_ID_2, spect_ID_3, spect_ID_4,
                            phase = '',
                            ntw=None):

    assert phase == 'train' or phase == 'test'
    assert ntw != None

    out_path_1 = output_dir + phase + '/' + spect_ID_1
    out_path_2 = output_dir + phase + '/' + spect_ID_2
    out_path_3 = output_dir + phase + '/' + spect_ID_3
    out_path_4 = output_dir + phase + '/' + spect_ID_4

    for p in [out_path_1, out_path_2, out_path_3, out_path_4]:
        if not os.path.exists(p):
            os.makedirs(p)


    for im1, seg1, im2, seg2, im3, seg3, im4, seg4 in zip(images_1, segmentations_1,
                                                          images_2, segmentations_2,
                                                          images_3, segmentations_3,
                                                          images_4, segmentations_4):

        img_1, h, w = get_image_arr(dir_img_1 + im1, input_width, input_height, ret_hw=True)
        img_2 = get_image_arr(dir_img_2 + im2, input_width, input_height)
        img_3 = get_image_arr(dir_img_3 + im3, input_width, input_height)
        img_4 = get_image_arr(dir_img_4 + im4, input_width, input_height)

        img_1 = np.expand_dims(img_1, axis=0)
        img_2 = np.expand_dims(img_2, axis=0)
        img_3 = np.expand_dims(img_3, axis=0)
        img_4 = np.expand_dims(img_4, axis=0)

        y_pred_1, y_pred_2, y_pred_3, y_pred_4 = model.predict([img_1, img_2, img_3, img_4])

        y_pred_1 = np.uint8(np.squeeze(np.argmax(y_pred_1, axis=3)))
        y_pred_2 = np.uint8(np.squeeze(np.argmax(y_pred_2, axis=3)))
        y_pred_3 = np.uint8(np.squeeze(np.argmax(y_pred_3, axis=3)))
        y_pred_4 = np.uint8(np.squeeze(np.argmax(y_pred_4, axis=3)))

        y_pred_1 = resize_mask(y_pred_1*50, w, h)
        y_pred_2 = resize_mask(y_pred_2*50, w, h)
        y_pred_3 = resize_mask(y_pred_3*50, w, h)
        y_pred_4 = resize_mask(y_pred_4*50, w, h)


        cv2.imwrite(out_path_1 + '/' + im1[:-4] + ".png", y_pred_1)
        cv2.imwrite(out_path_2 + '/' + im2[:-4] + ".png", y_pred_2)
        cv2.imwrite(out_path_3 + '/' + im3[:-4] + ".png", y_pred_3)
        cv2.imwrite(out_path_4 + '/' + im4[:-4] + ".png", y_pred_4)







prodcue_recursive_label(val_dir_img_1,
                        val_dir_img_2,
                        val_dir_img_3,
                        val_dir_img_4,
                        val_images_1, val_segmentations_1,
                        val_images_2, val_segmentations_2,
                        val_images_3, val_segmentations_3,
                        val_images_4, val_segmentations_4,
                        input_width, input_height,
                        spect_ID_1, spect_ID_2, spect_ID_3, spect_ID_4,
                        phase='test',
                        ntw=model)


prodcue_recursive_label(train_dir_img_1,
                        train_dir_img_2,
                        train_dir_img_3,
                        train_dir_img_4,
                        train_images_1, train_segmentations_1,
                        train_images_2, train_segmentations_2,
                        train_images_3, train_segmentations_3,
                        train_images_4, train_segmentations_4,
                        input_width, input_height,
                        spect_ID_1, spect_ID_2, spect_ID_3, spect_ID_4,
                        phase='train',
                        ntw=model)








