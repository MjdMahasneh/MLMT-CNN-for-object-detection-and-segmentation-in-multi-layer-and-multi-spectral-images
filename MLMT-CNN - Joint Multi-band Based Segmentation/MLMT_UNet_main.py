import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
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
from keras.utils import plot_model

from utils.provider import get_session
from utils.provider import get_img_seg_from_dir
from utils.provider import get_img_seg_data
from utils.provider import vis_multi_spect
from utils.provider import give_color_to_seg_img
from MLMT_UNet.MLMT_UNet import get_mlmt_unet
from utils.data_loader import generate_data_III
from utils.metrics import IoU
from utils.get_contours import get_contours
from MLMT_UNet.loss import weighted_categorical_crossentropy


test_phase = True ## True -> testing and False -> training.
full_disk = True ##  True -> full disk and False -> patches
if full_disk:
    from MLMT_UNet.config_full_disk import config as cfg
else:
    from MLMT_UNet.config_patches import config as cfg




if cfg['GPU_fraction'] != None and cfg['GPU_fraction'] != 1.0:
    k.set_session(get_session(cfg['GPU_fraction']))


if not os.path.exists(cfg['saving_dir']):
    os.makedirs(cfg['saving_dir'])

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
output_height, output_width = cfg['output_height'], cfg['output_width']







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

print('found {} training and {} validation samples'.format(train_num_samples, val_num_samples))



## parse val images/labels of all spects
print('parsing validation data ..')
X_1, Y_1 = get_img_seg_data(images_names=val_images_1, segmentations_names=val_segmentations_1,
                            dir_img=val_dir_img_1, dir_seg=val_dir_seg_1,
                            n_classes = n_classes,
                            input_width = input_width, input_height = input_height,
                            output_width = output_width, output_height = output_height)
print('X_1, Y_1 shape : ', X_1.shape, Y_1.shape)


X_2, Y_2 = get_img_seg_data(images_names=val_images_2, segmentations_names=val_segmentations_2,
                            dir_img=val_dir_img_2, dir_seg=val_dir_seg_2,
                            n_classes = n_classes,
                            input_width = input_width, input_height = input_height,
                            output_width = output_width, output_height = output_height)
print('X_2, Y_2 shape : ', X_2.shape, Y_2.shape)


X_3, Y_3 = get_img_seg_data(images_names=val_images_3, segmentations_names=val_segmentations_3,
                            dir_img=val_dir_img_3, dir_seg=val_dir_seg_3,
                            n_classes = n_classes,
                            input_width = input_width, input_height = input_height,
                            output_width = output_width, output_height = output_height)
print('X_3, Y_3 shape : ', X_3.shape, Y_3.shape)


X_4, Y_4 = get_img_seg_data(images_names=val_images_4, segmentations_names=val_segmentations_4,
                            dir_img=val_dir_img_4, dir_seg=val_dir_seg_4,
                            n_classes = n_classes,
                            input_width = input_width, input_height = input_height,
                            output_width = output_width, output_height = output_height)
print('X_4, Y_4 shape : ', X_4.shape, Y_4.shape)
print('finished pasrsing validation data.')





## get network and compile
model = get_mlmt_unet(n_classes=n_classes,
                      input_height = input_height, input_width = input_width,
                      merge='add')

if cfg['model_summary']:
    model.summary()
    plot_model(model, to_file=cfg['saving_dir']+'model.png')

if cfg['fine_tune']:
    try:
        model.load_weights(cfg['fine_tune'])
    except Exception as e:
        print(e)


adam = optimizers.Adam(lr=cfg['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

class_weights = cfg['class_weights']
model.compile(loss = [weighted_categorical_crossentropy(class_weights), weighted_categorical_crossentropy(class_weights),
                      weighted_categorical_crossentropy(class_weights), weighted_categorical_crossentropy(class_weights)],
              optimizer=adam,
              metrics=[metrics.categorical_accuracy])

Checkpoint = keras.callbacks.ModelCheckpoint(cfg['weights_path'], monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)


## train
if not test_phase :
    hist1 = model.fit_generator(generate_data_III(train_dir_img_1, train_dir_seg_1,
                                                  train_dir_img_2, train_dir_seg_2,
                                                  train_dir_img_3, train_dir_seg_3,
                                                  train_dir_img_4, train_dir_seg_4,
                                                  input_height, input_width,
                                                  n_classes,
                                                  train_images_1, train_segmentations_1,
                                                  train_images_2, train_segmentations_2,
                                                  train_images_3, train_segmentations_3,
                                                  train_images_4, train_segmentations_4,
                                                  batch_size = cfg['batch_size']),
                                steps_per_epoch = cfg['steps_per_epoch'],
                                validation_data = generate_data_III(val_dir_img_1, val_dir_seg_1,
                                                                    val_dir_img_2, val_dir_seg_2,
                                                                    val_dir_img_3, val_dir_seg_3,
                                                                    val_dir_img_4, val_dir_seg_4,
                                                                    input_height, input_width,
                                                                    n_classes,
                                                                    val_images_1, val_segmentations_1,
                                                                    val_images_2, val_segmentations_2,
                                                                    val_images_3, val_segmentations_3,
                                                                    val_images_4, val_segmentations_4,
                                                                    batch_size = cfg['batch_size']),
                                validation_steps = (val_num_samples // cfg['batch_size']),
                                verbose=cfg['verbose'],
                                callbacks=[Checkpoint],
                                epochs=cfg['epochs'])



## test
if test_phase:
    assert cfg['pre_trained_model'] is not ''
    try:
        wp = cfg['saving_dir']+cfg['pre_trained_model']
        model.load_weights(wp)
        print('weights loaded successfully {}'.format(wp))
    except Exception as e:
        print(e)

    t0= time.clock()

    y_pred_1, y_pred_2, y_pred_3, y_pred_4 = model.predict([X_1, X_2, X_3, X_4])

    t= time.clock() - t0
    print('process terminiated in {} secounds'.format(t))

    y_predi_1  = np.argmax(y_pred_1, axis=3)
    y_predi_2  = np.argmax(y_pred_2, axis=3)
    y_predi_3  = np.argmax(y_pred_3, axis=3)
    y_predi_4  = np.argmax(y_pred_4, axis=3)


    y_testi_1 = np.argmax(Y_1, axis=3)
    y_testi_2 = np.argmax(Y_2, axis=3)
    y_testi_3 = np.argmax(Y_3, axis=3)
    y_testi_4 = np.argmax(Y_4, axis=3)




    print('y_testi_1 {} y_predi_1 {} y_testi_2 {} y_predi_2 {} y_testi_3 {} y_predi_3 {} y_testi_4 {} y_predi_4 {} '.format(y_testi_1.shape ,y_predi_1.shape,
                                                                                                                            y_testi_2.shape ,y_predi_2.shape,
                                                                                                                            y_testi_3.shape ,y_predi_3.shape,
                                                                                                                            y_testi_4.shape ,y_predi_4.shape))
    IoU(y_testi_1, y_predi_1, spect_ID=cfg['spect_1'])
    IoU(y_testi_2, y_predi_2, spect_ID=cfg['spect_2'])
    IoU(y_testi_3, y_predi_3, spect_ID=cfg['spect_3'])
    IoU(y_testi_4, y_predi_4, spect_ID=cfg['spect_4'])





    if cfg['vis_at_test']:
        for i in range(40):

            vis_1 = get_contours(X_1[i][:, :, 0], y_predi_1[i])
            vis_2 = get_contours(X_2[i][:, :, 0], y_predi_2[i])
            vis_3 = get_contours(X_3[i][:, :, 0], y_predi_3[i])
            vis_4 = get_contours(X_4[i][:, :, 0], y_predi_4[i])

            vis_multi_spect(images=[vis_1, vis_2, vis_3, vis_4], titles=[cfg['spect_1'], cfg['spect_2'], cfg['spect_3'], cfg['spect_4']],
                            save=False, out_path='./figures', fig_name=str(i))






