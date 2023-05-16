import cv2
import numpy as np
import copy
from PIL import Image
import os

def get_imlist_starting_with(path, starting_str):
  return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(starting_str)]

def augment_all_bands_simultaneously(img_1, img_2, img_data_aug, config, bands):
  
  rows, cols = img_1.shape[:2]

  if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
          img_1 = cv2.flip(img_1, 1)
          img_2 = cv2.flip(img_2, 1)

          for band in bands :
            if band == '1' :
              bboxes_ID = 'bboxes'

            elif band == '2' :
              bboxes_ID = 'bboxes_2'
            else :
              raise Exception('Error in Band ID')
            
            for bbox in img_data_aug[bboxes_ID]:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2

  if config.use_vertical_flips and np.random.randint(0, 2) == 0:
          img_1 = cv2.flip(img_1, 0)
          img_2 = cv2.flip(img_2, 0)

          for band in bands :
            if band == '1' :
              bboxes_ID = 'bboxes'

            elif band == '2' :
              bboxes_ID = 'bboxes_2'
            else :
              raise Exception('Error in Band ID')

            for bbox in img_data_aug[bboxes_ID]:
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2

  if config.rot_90:
          angle = np.random.choice([0,90,180,270],1)[0]
          if angle == 270:
                  img_1 = np.transpose(img_1, (1,0,2))
                  img_1 = cv2.flip(img_1, 0)
                  img_2 = np.transpose(img_2, (1,0,2))
                  img_2 = cv2.flip(img_2, 0)
          elif angle == 180:
                  img_1 = cv2.flip(img_1, -1)
                  img_2 = cv2.flip(img_2, -1)
          elif angle == 90:
                  img_1 = np.transpose(img_1, (1,0,2))
                  img_1 = cv2.flip(img_1, 1)
                  img_2 = np.transpose(img_2, (1,0,2))
                  img_2 = cv2.flip(img_2, 1)
          elif angle == 0:
                  pass

          for band in bands :
            if band == '1' :
              bboxes_ID = 'bboxes'
            elif band == '2' :
              bboxes_ID = 'bboxes_2'
            else :
              raise Exception('Error in Band ID')

            for bbox in img_data_aug[bboxes_ID]:
                    x1 = bbox['x1']
                    x2 = bbox['x2']
                    y1 = bbox['y1']
                    y2 = bbox['y2']
                    if angle == 270:
                            bbox['x1'] = y1
                            bbox['x2'] = y2
                            bbox['y1'] = cols - x2
                            bbox['y2'] = cols - x1
                    elif angle == 180:
                            bbox['x2'] = cols - x1
                            bbox['x1'] = cols - x2
                            bbox['y2'] = rows - y1
                            bbox['y1'] = rows - y2
                    elif angle == 90:
                            bbox['x1'] = rows - y2
                            bbox['x2'] = rows - y1
                            bbox['y1'] = x1
                            bbox['y2'] = x2        
                    elif angle == 0:
                            pass
  return img_data_aug, img_1, img_2

def augment(img_data, config, images_dir, spect_1, spect_2, augment=True):
  assert 'filepath' in img_data
  assert 'bboxes' in img_data
  assert 'bboxes_2' in img_data
  assert 'width' in img_data
  assert 'height' in img_data

  img_data_aug = copy.deepcopy(img_data)

  spect_1_img = Image.open(img_data_aug['filepath'])
  spect_1_img = np.array(spect_1_img)
  spect_1_img *= 255
  spect_1_img = np.repeat(spect_1_img[..., np.newaxis], 3, -1)

  line = img_data_aug['filepath']
  image_name = line.split('/')[-1]
  image_ID = image_name.split('_')[0]

  ## get 2nd spect :
  spect_dir = images_dir + '/' + spect_2 + '/'
  corrosponding_image_spect_2 = get_imlist_starting_with(spect_dir, image_ID)
  spect_2_img_path = corrosponding_image_spect_2[0]
  spect_2_img = Image.open(spect_2_img_path)
  spect_2_img = np.array(spect_2_img)
  spect_2_img *= 255
  spect_2_img = np.repeat(spect_2_img[..., np.newaxis], 3, -1)

  if augment:
    img_data_aug, img_spect_1, img_spect_2 = augment_all_bands_simultaneously(spect_1_img, spect_2_img,
                                                                              img_data_aug,
                                                                              config,
                                                                              bands = ('1', '2'))
  img_data_aug['width'] = img_spect_1.shape[1]
  img_data_aug['height'] = img_spect_1.shape[0]
  return img_data_aug, img_spect_1, img_spect_2
