import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os



def get_session(gpu_fraction):
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                               allow_growth=True)
   print('assigning GPU fraction {}'.format(gpu_fraction))
   return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape) == 3:
        seg = seg[:, :, 0]

    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][2]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][0]))

    return seg_img

def get_image_arr(path, width, height, ret_hw = False):
    img = Image.open(path)
    img = np.array(img)

    h, w = img.shape

    img = cv2.resize(img, (width, height))
    img = np.repeat(img[..., np.newaxis], 3, -1)
    if ret_hw:
        return img, h, w
    else:
        return img



def get_segmentation_arr(path, n_classes, width, height):
    seg_labels = np.zeros((height, width, n_classes))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(n_classes):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels



def get_img_seg_from_dir(dir_img, dir_seg):
    images = os.listdir(dir_img)
    images.sort()

    segmentations = os.listdir(dir_seg)
    segmentations.sort()

    return images, segmentations


def get_img_seg_data(images_names, segmentations_names, dir_img, dir_seg, n_classes, input_width, input_height, output_width, output_height):
    X, Y = [], []
    for im, seg in zip(images_names, segmentations_names):
        X.append(get_image_arr(dir_img + im, input_width, input_height))
        Y.append(get_segmentation_arr(dir_seg + seg, n_classes, output_width, output_height))
    return np.array(X), np.array(Y)



def vis_multi_spect(images=[], titles=[], save=False, out_path='', fig_name=''):

    fig = plt.figure(figsize=(len(titles), 2), dpi=300)

    for i in range(len(images)):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(titles[i])

    if save:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        path = out_path + '/' + fig_name + '.png'
        plt.savefig(fname=path, dpi=300,
                    bbox_inches='tight', transparent='True', pad_inches=0)
    plt.show()




def resize_mask(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)







