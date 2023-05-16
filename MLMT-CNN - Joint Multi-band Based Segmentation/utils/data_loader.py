import random
import numpy as np
from utils.provider import get_image_arr
from utils.provider import get_segmentation_arr

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def unison_shuffled_lists(a, b, c, d, e, f, g, h):
    combined = list(zip(a, b, c, d, e, f, g, h))
    random.shuffle(combined)
    a[:], b[:], c[:], d[:], e[:], f[:], g[:], h[:] = zip(*combined)
    return a, b, c, d, e, f, g, h

def generate_data_III(dir_img_1, dir_seg_1, dir_img_2, dir_seg_2,
                      dir_img_3, dir_seg_3, dir_img_4,
                      dir_seg_4, input_height, input_width, n_classes,
                      images_1, masks_1, images_2, masks_2,
                      images_3, masks_3, images_4, masks_4,
                      batch_size):

    assert len(images_1) == len(masks_1)
    assert batch_size <= len(images_1)

    assert len(images_2) == len(masks_2)
    assert batch_size <= len(images_2)

    assert len(images_3) == len(masks_3)
    assert batch_size <= len(images_3)

    assert len(images_4) == len(masks_4)
    assert batch_size <= len(images_4)


    number_of_batches = int(np.ceil(len(images_1) / batch_size))
    counter = 0

    while True:
        image_batch_1 = []
        mask_batch_1 = []

        image_batch_2 = []
        mask_batch_2 = []

        image_batch_3 = []
        mask_batch_3 = []

        image_batch_4 = []
        mask_batch_4 = []

        ## define the starting and ending point of the batch in the images array :
        idx_start = batch_size * counter
        idx_end = batch_size * (counter + 1)

        for idx in range(idx_start, idx_end):

            img_1 = get_image_arr(dir_img_1 + images_1[idx], input_width, input_height)
            img_2 = get_image_arr(dir_img_2 + images_2[idx], input_width, input_height)
            img_3 = get_image_arr(dir_img_3 + images_3[idx], input_width, input_height)
            img_4 = get_image_arr(dir_img_4 + images_4[idx], input_width, input_height)




            mask_1 = get_segmentation_arr(dir_seg_1 + masks_1[idx], n_classes, input_width, input_height)
            mask_2 = get_segmentation_arr(dir_seg_2 + masks_2[idx], n_classes, input_width, input_height)
            mask_3 = get_segmentation_arr(dir_seg_3 + masks_3[idx], n_classes, input_width, input_height)
            mask_4 = get_segmentation_arr(dir_seg_4 + masks_4[idx], n_classes, input_width, input_height)

            image_batch_1.append(img_1)
            mask_batch_1.append(mask_1)

            image_batch_2.append(img_2)
            mask_batch_2.append(mask_2)

            image_batch_3.append(img_3)
            mask_batch_3.append(mask_3)

            image_batch_4.append(img_4)
            mask_batch_4.append(mask_4)

        counter += 1

        if (counter == number_of_batches - 1):
            images_1, masks_1,\
            images_2, masks_2,\
            images_3, masks_3,\
            images_4, masks_4 = unison_shuffled_lists(images_1, masks_1,
                                                      images_2, masks_2,
                                                      images_3, masks_3,
                                                      images_4, masks_4)

            ## wrap up epoch and exit current loop, this will reset the counter and the indecies :
            counter = 0 # reset counter


        yield [np.array(image_batch_1), np.array(image_batch_2), np.array(image_batch_3), np.array(image_batch_4)],\
              [np.array(mask_batch_1), np.array(mask_batch_2), np.array(mask_batch_3), np.array(mask_batch_4)]

