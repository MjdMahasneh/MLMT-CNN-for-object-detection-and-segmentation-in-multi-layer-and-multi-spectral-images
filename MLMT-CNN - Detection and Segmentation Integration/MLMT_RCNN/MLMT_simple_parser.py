import numpy as np
import os
from PIL import Image

def get_imlist_starting_with(path, starting_str):
  return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(starting_str)]

def parse_label_file(input_path, spect, data_mode, main_flag = False):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    
    with open(input_path, 'r') as f:
        print('Parsing annotation file ', spect)
        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg:
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True

                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                img = Image.open(filename)
                img = np.array(img)
                img *= 255
                img = np.repeat(img[..., np.newaxis], 3, -1)

                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                if main_flag == True:
                    all_imgs[filename]['bboxes'] = []
                    all_imgs[filename]['bboxes_2'] = []
                    all_imgs[filename]['bboxes_3'] = []
                    all_imgs[filename]['bboxes_4'] = []
                else:
                    all_imgs[filename]['bboxes'] = []

                if data_mode == 'train' :
                    all_imgs[filename]['imageset'] = 'train'
                elif data_mode == 'test' :
                    all_imgs[filename]['imageset'] = 'test'
                else:
                    raise Exception('data_mode error : data_mode must be either train or test.')

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(float(x1)), 'x2': int(float(x2)), 'y1': int(float(y1)),
                 'y2': int(float(y2))})

    return all_imgs, found_bg, classes_count, class_mapping

def get_data(input_path_1, input_path_2,
             input_path_3, input_path_4,
             images_dir,
             spect_1, spect_2,
             spect_3, spect_4,
             data_set):

    all_images_1, found_bg, classes_count, class_mapping = parse_label_file(input_path_1, spect_1, data_mode = data_set, main_flag = True)
    all_images_2, _, _, _ = parse_label_file(input_path_2, spect_2, data_mode = data_set)
    all_images_3, _, _, _ = parse_label_file(input_path_3, spect_3, data_mode=data_set)
    all_images_4, _, _, _ = parse_label_file(input_path_4, spect_4, data_mode=data_set)

    for filename_1 in all_images_1:

        image_ID = filename_1.split('/')[-1].split('_')[0]
        spect_dir_2 = images_dir + '/' + spect_2 + '/'
        filename_2 = get_imlist_starting_with(spect_dir_2, image_ID)[0]
        spect_dir_3 = images_dir + '/' + spect_3 + '/'
        filename_3 = get_imlist_starting_with(spect_dir_3, image_ID)[0]
        spect_dir_4 = images_dir + '/' + spect_4 + '/'
        filename_4 = get_imlist_starting_with(spect_dir_4, image_ID)[0]

        all_images_1[filename_1]['bboxes_2'] = all_images_2[filename_2]['bboxes']
        all_images_1[filename_1]['bboxes_3'] = all_images_3[filename_3]['bboxes']
        all_images_1[filename_1]['bboxes_4'] = all_images_4[filename_4]['bboxes']

        all_data = []
        for key in all_images_1:
            all_data.append(all_images_1[key])

        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

    return all_data, classes_count, class_mapping
