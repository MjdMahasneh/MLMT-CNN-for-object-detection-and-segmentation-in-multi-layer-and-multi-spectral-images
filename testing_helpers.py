import tensorflow as tf
import cv2
import numpy as np
from MLMT_RCNN import roi_helpers
import os
from PIL import Image
import matplotlib.pyplot as plt

def get_session(gpu_fraction):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def get_imlist_starting_with(path, starting_str):
  return [os.path.join(path, f) for f in os.listdir(path) if f.startswith(starting_str)]

def format_img_size(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio

def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy

def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return real_x1, real_y1, real_x2, real_y2

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

def box_area(box):
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    return area_a

def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)

def overlap_over_GT_area(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    intesection_area = intersection(a, b)
    GT_box_area = box_area(b)
    return float(intesection_area) / float(GT_box_area)

def overlap_over_predicted_box_area(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    intesection_area = intersection(a, b)
    Predicted_box_area = box_area(a)
    return float(intesection_area) / float(Predicted_box_area)

def get_map(img_name, img, pred, gt, f, spect_ID, ratio, vis_dir='./visualization/', report_dir='./log/'):

    report_dir = report_dir + spect_ID + '/'
    vis_dir =  vis_dir + '/' + spect_ID + '/'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    TP, FP, FN, TN = 0, 0, 0, 0
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])

    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']

        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
        real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio,
                                                                                      int_pred_x1, int_pred_y1,
                                                                                      int_pred_x2, int_pred_y2)
        line = img_name + ',' + str(int(real_pred_x1)) + ',' + str(int(real_pred_y1)) + ',' + str(int(real_pred_x2)) + ',' + str(int(real_pred_y2)) + ',AR' + '\n'

        with open(report_dir+'/'+img_name[:-4]+'.txt', 'a') as outf:
            outf.write(line)

        found_match = False
        count = 0

        pick_intersected_GT = []
        for gt_box in gt:
            gt_class = gt_box['class']
            gt_x1 = gt_box['x1'] / fx
            gt_x2 = gt_box['x2'] / fx
            gt_y1 = gt_box['y1'] / fy
            gt_y2 = gt_box['y2'] / fy

            if gt_class != pred_class:
                continue

            intersection_pred_GT = intersection((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            gt_box_area = box_area((gt_x1, gt_y1, gt_x2, gt_y2))
            pred_box_area = box_area((pred_x1, pred_y1, pred_x2, pred_y2))

            if (intersection_pred_GT / pred_box_area) >= 0.5 or (intersection_pred_GT / gt_box_area) >= 0.5:
                pick_intersected_GT.append(count)
            count += 1

        if len(pick_intersected_GT) != 0:
            found_match = True
            for idx in pick_intersected_GT:
                gt[idx]['bbox_matched'] = True
                T[pred_class].append(int(found_match))
                P[pred_class].append(pred_prob)
            TP = TP + len(pick_intersected_GT)

            int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
            real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio,
                                                                                          int_pred_x1, int_pred_y1,
                                                                                          int_pred_x2, int_pred_y2)
            cv2.rectangle(img, (real_pred_x1, real_pred_y1), (real_pred_x2, real_pred_y2), (0, 0, 255), 2)

        if int(found_match) == 0:
            int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
            real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio,
                                                                                          int_pred_x1, int_pred_y1,
                                                                                          int_pred_x2, int_pred_y2)
            cv2.rectangle(img, (real_pred_x1, real_pred_y1), (real_pred_x2, real_pred_y2), (0, 0, 255), 2)
            FP += 1
            T[pred_class].append(int(found_match))
            P[pred_class].append(pred_prob)

    for gt_box in gt:
        if not gt_box['bbox_matched']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []
            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)
            FN += 1

    result_path = vis_dir + img_name[:-4] + '.png'
    cv2.imwrite(result_path, img)

    return T, P, TP, FP, FN

def process_dets(P_cls, P_regr, ROIs, bboxes, probs, bbox_threshold, C, class_mapping):
    for ii in range(P_cls.shape[1]):
        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue

        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
        if cls_name not in bboxes:
            bboxes[cls_name] = []
            probs[cls_name] = []

        (x, y, w, h) = ROIs[0, ii, :]

        cls_num = np.argmax(P_cls[0, ii, :])
        try:
            (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
            tx /= C.classifier_regr_std[0]
            ty /= C.classifier_regr_std[1]
            tw /= C.classifier_regr_std[2]
            th /= C.classifier_regr_std[3]
            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
        except:
            pass
        bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
        probs[cls_name].append(np.max(P_cls[0, ii, :]))
    return bboxes, probs

def pad_rois(mul_res, C, jk):
    short = False
    ROIs = np.expand_dims(mul_res[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
    if ROIs.shape[1] == 0:
        short = True
        pass
    elif jk == mul_res.shape[0] // C.num_rois:
        curr_shape = ROIs.shape
        target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = ROIs
        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
        ROIs = ROIs_padded
    return ROIs, short

def perform_nms(bboxes, probs, spect_ID):
    all_dets = []
    for key in bboxes:
        bbox = np.array(bboxes[key])
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast_with_probs(bbox, np.array(probs[key]), overlap_thresh=.05)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk], 'spect_ID': str(spect_ID)}
            all_dets.append(det)
    return all_dets

def get_det_image(images_dir, spect, image_ID):
    spect_dir = images_dir + '/' + spect + '/'
    corrosponding_image_spect_2 = get_imlist_starting_with(spect_dir, image_ID)
    assert len(corrosponding_image_spect_2) == 1
    Spect_filepath = corrosponding_image_spect_2[0]
    img_name = Spect_filepath.split('/')[-1]
    img_raw = Image.open(Spect_filepath)
    img_raw = np.array(img_raw)
    img_proc = img_raw*255
    img_proc = img_proc.copy()
    img_proc = np.repeat(img_proc[..., np.newaxis], 3, -1)
    return img_raw, img_proc, img_name

def calc_F1(cnt_TP, cnt_FP, cnt_FN, spect, report=False):
    precision = cnt_TP/(cnt_TP + cnt_FP)
    recall = cnt_TP/(cnt_TP + cnt_FN)
    F1 = (2*precision*recall)/(precision+recall)
    if report:
        print('-' * 30, 'spect {}'.format(spect))
        print('precision'.ljust(15), '{}'.format(round(precision, 3)))
        print('recall'.ljust(15), '{}'.format(round(recall, 3)))
        print('F1'.ljust(15), '{}'.format(round(F1, 3)))
    return

def visualize_detections(img_vis, y_predi, box_idx, class_ID, pred_box,
                         x1, y1, x2, y2,
                         vis_contours=True,
                         ret_contours=False):
    AR_idx = np.where(y_predi[box_idx] != class_ID)
    y_predi[box_idx][AR_idx] = 0
    new_height, new_width = pred_box['height'], pred_box['width']
    patch_resized = cv2.resize(y_predi[box_idx].astype(np.float32), (new_width, new_height),
                               interpolation=cv2.INTER_CUBIC)
    patch_resized = np.uint8(patch_resized)
    if vis_contours:
        contours, _ = cv2.findContours(patch_resized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            contours[i][:, 0, 0] = (contours[i][:, 0, 0] + x1).astype(int)
            contours[i][:, 0, 1] = (contours[i][:, 0, 1] + y1).astype(int)
            cv2.drawContours(img_vis, contours, i, (255, 0, 0), 1)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if ret_contours:
        return img_vis, contours
    else:
        return img_vis

def save_image(img, vis_dir, spect, img_name):
    res_dir = vis_dir + 'integrated' + '/' + spect + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    img_path = res_dir + img_name[:-4] + '.png'
    cv2.imwrite(img_path, img)

def save_contour(path, spect, img_name, contours):
    res_dir = path + 'contours' + '/' + spect + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    img_path = res_dir + img_name[:-4] + '.npy'
    np.save(img_path, np.array(contours))

def format_segmentation_img(img , width, height):
    img = np.float32(cv2.resize(img, ( width , height )))
    img = np.repeat(img[..., np.newaxis], 3, -1)
    return img

def vis_contours_from_npy(path, img):
    cnts = np.load(path)
    for i, c in enumerate(cnts):
        if np.shape(c)[0] > 1:
            for ii, cc in enumerate(cnts[i]):
                cv2.drawContours(img, cnts[i], ii, (0, 0, 255), 2)
        else:
            cv2.drawContours(img, cnts[i], 0, (0, 0, 255), 2)
    return img








