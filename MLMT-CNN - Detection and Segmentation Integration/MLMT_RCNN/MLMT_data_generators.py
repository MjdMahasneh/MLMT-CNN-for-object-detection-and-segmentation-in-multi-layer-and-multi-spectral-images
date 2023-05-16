from __future__ import absolute_import
import numpy as np
import cv2
import random
from . import MLMT_data_augment
import threading
import itertools

def union(au, bu, area_intersection):
        area_a = (au[2] - au[0]) * (au[3] - au[1])
        area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
        area_union = area_a + area_b - area_intersection
        return area_union

def intersection(ai, bi):
        x = max(ai[0], bi[0])
        y = max(ai[1], bi[1])
        w = min(ai[2], bi[2]) - x
        h = min(ai[3], bi[3]) - y
        if w < 0 or h < 0:
                return 0
        return w*h

def iou(a, b):
        if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
                return 0.0
        area_i = intersection(a, b)
        area_u = union(a, b, area_i)
        return float(area_i) / float(area_u + 1e-6)

def get_new_img_size(width, height, img_min_side=600):
        if width <= height:
                f = float(img_min_side) / width
                resized_height = int(f * height)
                resized_width = img_min_side
        else:
                f = float(img_min_side) / height
                resized_width = int(f * width)
                resized_height = img_min_side
        return resized_width, resized_height

class SampleSelector:
        def __init__(self, class_count):
                self.classes = [b for b in class_count.keys() if class_count[b] > 0]
                self.class_cycle = itertools.cycle(self.classes)
                self.curr_class = next(self.class_cycle)

        def skip_sample_for_balanced_class(self, img_data):
                class_in_img = False
                for bbox in img_data['bboxes']:
                        cls_name = bbox['class']
                        if cls_name == self.curr_class: 
                                class_in_img = True
                                self.curr_class = next(self.class_cycle)
                                break

                if class_in_img:
                        return False
                else:
                        return True

def calc_rpn_mul_spect(C, img_data, width, height, resized_width, resized_height, img_length_calc_function, spect):
        
        if spect == '1' :
                img_data_bboxes = img_data['bboxes']
        elif spect == '2' :
                img_data_bboxes = img_data['bboxes_2']
        else :
                raise ValueError('spect is not defined in calc_rpn_mul_spect()')

        downscale = float(C.rpn_stride)
        anchor_sizes = C.anchor_box_scales
        anchor_ratios = C.anchor_box_ratios
        num_anchors = len(anchor_sizes) * len(anchor_ratios)    

        # calculate the output map size based on the network architecture
        (output_width, output_height) = img_length_calc_function(resized_width, resized_height)

        n_anchratios = len(anchor_ratios)
        
        # initialise empty output objectives
        y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
        y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
        y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

        num_bboxes = len(img_data_bboxes)

        num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
        best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
        best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

        # get the GT box coordinates, and resize to account for image resizing
        gta = np.zeros((num_bboxes, 4))
        for bbox_num, bbox in enumerate(img_data_bboxes):
                gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
                gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
                gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
                gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
        
        # rpn ground truth
        for anchor_size_idx in range(len(anchor_sizes)):
                for anchor_ratio_idx in range(n_anchratios):
                        anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
                        anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]   

                        for ix in range(output_width):                                  
                                # x-coordinates of the current anchor box       
                                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                                x2_anc = downscale * (ix + 0.5) + anchor_x / 2  
                                
                                # ignore boxes that go across image boundaries                                  
                                if x1_anc < 0 or x2_anc > resized_width:
                                        continue
                                        
                                for jy in range(output_height):
                                        # y-coordinates of the current anchor box
                                        y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                                        y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                                        # ignore boxes that go across image boundaries
                                        if y1_anc < 0 or y2_anc > resized_height:
                                                continue

                                        bbox_type = 'neg'

                                        best_iou_for_loc = 0.0

                                        for bbox_num in range(num_bboxes):
                                                curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])

                                                if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                                                        cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                                                        cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                                                        cxa = (x1_anc + x2_anc)/2.0
                                                        cya = (y1_anc + y2_anc)/2.0

                                                        tx = (cx - cxa) / (x2_anc - x1_anc)
                                                        ty = (cy - cya) / (y2_anc - y1_anc)
                                                        tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                                                        th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
                                                if img_data_bboxes[bbox_num]['class'] != 'bg':
                                                        # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                                                        if curr_iou > best_iou_for_bbox[bbox_num]:
                                                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                                                best_iou_for_bbox[bbox_num] = curr_iou
                                                                best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                                                best_dx_for_bbox[bbox_num,:] = [tx, ty, tw, th]
                                                        # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                                                        if curr_iou > C.rpn_max_overlap:
                                                                bbox_type = 'pos'
                                                                num_anchors_for_bbox[bbox_num] += 1
                                                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                                                if curr_iou > best_iou_for_loc:
                                                                        best_iou_for_loc = curr_iou
                                                                        best_regr = (tx, ty, tw, th)
                                                        # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                                                        if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                                                # gray zone between neg and pos
                                                                if bbox_type != 'pos':
                                                                        bbox_type = 'neutral'

                                        # turn on or off outputs depending on IOUs
                                        if bbox_type == 'neg':
                                                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                                                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                                        elif bbox_type == 'neutral':
                                                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                                                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                                        elif bbox_type == 'pos':
                                                y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                                                y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                                                start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                                                y_rpn_regr[jy, ix, start:start+4] = best_regr

        # we ensure that every bbox has at least one positive RPN region
        for idx in range(num_anchors_for_bbox.shape[0]):
                if num_anchors_for_bbox[idx] == 0:
                        # no box with an IOU greater than zero ...
                        if best_anchor_for_bbox[idx, 0] == -1:
                                continue
                        y_is_box_valid[
                                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                                best_anchor_for_bbox[idx,3]] = 1
                        y_rpn_overlap[
                                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios *
                                best_anchor_for_bbox[idx,3]] = 1
                        start = 4 * (best_anchor_for_bbox[idx,2] + n_anchratios * best_anchor_for_bbox[idx,3])
                        y_rpn_regr[
                                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], start:start+4] = best_dx_for_bbox[idx, :]

        y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
        y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

        y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
        y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

        y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
        y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

        pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
        neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

        num_pos = len(pos_locs[0])

        # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
        # regions. We also limit it to 256 regions.
        num_regions = 256

        if len(pos_locs[0]) > num_regions/2:
                val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
                y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
                num_pos = num_regions/2

        if len(neg_locs[0]) + num_pos > num_regions:
                val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
                y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

        y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
        y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

        return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
        """Takes an iterator/generator and makes it thread-safe by
        serializing call to the `next` method of given iterator/generator.
        """
        def __init__(self, it):
                self.it = it
                self.lock = threading.Lock()

        def __iter__(self):
                return self

        def next(self):
                with self.lock:
                        return next(self.it)            

def threadsafe_generator(f):
        """A decorator that takes a generator function and makes it thread-safe.
        """
        def g(*a, **kw):
                return threadsafe_iter(f(*a, **kw))
        return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, images_dir, spect_1_name, spect_2_name, mode='train'):

        sample_selector = SampleSelector(class_count)

        while True:
                if mode == 'train':
                        random.shuffle(all_img_data)

                for img_data in all_img_data:
                        if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                                continue

                        if mode == 'train':
                                img_data_aug, spect_1_img, spect_2_img  = MLMT_data_augment.augment(img_data,
                                                                                                    C,
                                                                                                    images_dir,
                                                                                                    spect_1_name, spect_2_name,
                                                                                                    augment=True)

                        else:
                                img_data_aug, spect_1_img, spect_2_img = MLMT_data_augment.augment(img_data,
                                                                                                   C,
                                                                                                   images_dir,
                                                                                                   spect_1_name, spect_2_name,
                                                                                                   augment=False)

                                
                        (width, height) = (img_data_aug['width'], img_data_aug['height'])
                        (rows, cols, _) = spect_1_img.shape

                        assert cols == width
                        assert rows == height

                        (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                        spect_1_img = cv2.resize(spect_1_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                        spect_2_img = cv2.resize(spect_2_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                        try:
                                y_rpn_cls_1, y_rpn_regr_1 = calc_rpn_mul_spect(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function, spect = '1')
                                y_rpn_cls_2, y_rpn_regr_2 = calc_rpn_mul_spect(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function, spect = '2')
                        except Exception as e:
                                print('Ops! somthing went wrong! ',e)

                        # Zero-center by mean pixel, and preprocess image
                        spect_1_img = spect_1_img[:,:, (2, 1, 0)]  # BGR -> RGB
                        spect_1_img = spect_1_img.astype(np.float32)
                        spect_1_img[:, :, 0] -= C.img_channel_mean[0]
                        spect_1_img[:, :, 1] -= C.img_channel_mean[1]
                        spect_1_img[:, :, 2] -= C.img_channel_mean[2]
                        spect_1_img /= C.img_scaling_factor

                        spect_1_img = np.transpose(spect_1_img, (2, 0, 1))
                        spect_1_img = np.expand_dims(spect_1_img, axis=0)

                        # Zero-center by mean pixel, and preprocess Spect 2 image
                        spect_2_img = spect_2_img[:,:, (2, 1, 0)]  # BGR -> RGB
                        spect_2_img = spect_2_img.astype(np.float32)
                        spect_2_img[:, :, 0] -= C.img_channel_mean[0]
                        spect_2_img[:, :, 1] -= C.img_channel_mean[1]
                        spect_2_img[:, :, 2] -= C.img_channel_mean[2]
                        spect_2_img /= C.img_scaling_factor

                        spect_2_img = np.transpose(spect_2_img, (2, 0, 1))
                        spect_2_img = np.expand_dims(spect_2_img, axis=0)

                        ## std scaling : 
                        y_rpn_regr_1[:, y_rpn_regr_1.shape[1]//2:, :, :] *= C.std_scaling
                        y_rpn_regr_2[:, y_rpn_regr_2.shape[1]//2:, :, :] *= C.std_scaling

                        if backend == 'tf':
                                spect_1_img = np.transpose(spect_1_img, (0, 2, 3, 1))
                                spect_2_img = np.transpose(spect_2_img, (0, 2, 3, 1))

                                y_rpn_cls_1 = np.transpose(y_rpn_cls_1, (0, 2, 3, 1))
                                y_rpn_cls_2 = np.transpose(y_rpn_cls_2, (0, 2, 3, 1))

                                y_rpn_regr_1 = np.transpose(y_rpn_regr_1, (0, 2, 3, 1))
                                y_rpn_regr_2 = np.transpose(y_rpn_regr_2, (0, 2, 3, 1))

                        yield np.copy(spect_1_img), np.copy(spect_2_img), [np.copy(y_rpn_cls_1), np.copy(y_rpn_regr_1), np.copy(y_rpn_cls_2), np.copy(y_rpn_regr_2)], img_data_aug
