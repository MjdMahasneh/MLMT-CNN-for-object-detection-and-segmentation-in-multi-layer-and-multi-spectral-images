import os
import shutil
import sys
import pickle
import time
from optparse import OptionParser
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from MLMT_RCNN import roi_helpers
from MLMT_RCNN import MLMT_ResNet as nn
import numpy as np
import matplotlib.pyplot as plt
from testing_helpers import get_session,\
    format_img_size, format_img, process_dets,\
    calc_F1, get_map, save_contour, pad_rois,\
    perform_nms, get_det_image, format_segmentation_img,\
    get_real_coordinates, visualize_detections, save_image
from MLMT_RCNN.MLMT_simple_parser import get_data
from MLMT_UNet.MLMT_UNet import get_mlmt_unet

sys.setrecursionlimit(40000)

parser = OptionParser()
parser.add_option('--path_1', dest='test_path_1', default ='F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/my_datasimple_label_testing_TIFF_284.txt' , help='Path to test data 1.')
parser.add_option('--path_2', dest='test_path_2', default ='F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/my_datasimple_label_testing_TIFF_171.txt' , help='Path to test data 2.')
parser.add_option('--path_3', dest='test_path_3', default ='F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/my_datasimple_label_testing_TIFF_195.txt' , help='Path to test data 3.')
parser.add_option('--path_4', dest='test_path_4', default ='F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/my_datasimple_label_testing_TIFF_304.txt' , help='Path to test data 4.')
parser.add_option('--config_filename', dest='config_filename', default ='config.pickle', help='Location to read the metadata related to the training (generated when training).')
parser.add_option('--test_images_dir', default='F:/Deep_projects/MulSpect IMG Cls/Arch_two/Data/testing_images',  dest='test_images_dir', help='')
parser.add_option('--spect_1', dest='spect_1', default='284', help='spect_1 name')
parser.add_option('--spect_2', dest='spect_2', default='171', help='spect_2 name')
parser.add_option('--spect_3', dest='spect_3', default='195', help='spect_3 name')
parser.add_option('--spect_4', dest='spect_4', default='304', help='spect_4 name')
parser.add_option('--RPN_model_path_A', dest='RPN_model_path_A', default='./model/detection_stage/171+284/1571_RPN.h5', help='RPN A weights')
parser.add_option('--DET_model_path_A', dest='DET_model_path_A', default='./model/detection_stage/171+284/360_DET.h5', help='CLS A weights')
parser.add_option('--RPN_model_path_B', dest='RPN_model_path_B', default='./model/detection_stage/195+304/1437_RPN.h5', help='RPN B weights')
parser.add_option('--DET_model_path_B', dest='DET_model_path_B', default='./model/detection_stage/195+304/1649_DET.h5', help='CLS B weights')
parser.add_option('--MLMT_UNet_model_path', dest='MLMT_UNet_model_path', default='model/segmentation_stage/244_SEG.h5', help='segmentation weights')
parser.add_option('--vis_dir', dest='vis_dir', default='./visualization/', help='visualization path')
parser.add_option('--report_dir', dest='report_dir', default='./log/', help='results report path')
(options, args) = parser.parse_args()
test_images_dir = options.test_images_dir
spect_1 = options.spect_1
spect_2 = options.spect_2
spect_3 = options.spect_3
spect_4 = options.spect_4
RPN_model_path_A = options.RPN_model_path_A
DET_model_path_A = options.DET_model_path_A
RPN_model_path_B = options.RPN_model_path_B
DET_model_path_B = options.DET_model_path_B
MLMT_UNet_model_path = options.MLMT_UNet_model_path
vis_dir = options.vis_dir
report_dir = options.report_dir
if os.path.exists(report_dir):
    shutil.rmtree(report_dir)
config_output_filename = options.config_filename
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

K.set_session(get_session(C.gpu_fraction))

C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 100
bbox_threshold = 0.5
MLMT_UNet_input_height , MLMT_UNet_input_width = 224 , 224
MLMT_UNet_n_classes = 3
merge_res_across_RPNs = True ## False

class_mapping = C.class_mapping
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)
class_mapping = {v: k for k, v in class_mapping.items()}
print('class mapping {}'.format(class_mapping))
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_feat = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_feat = (None, None, 1024)
image_input_1 = Input(shape=input_shape_img)
image_input_2 = Input(shape=input_shape_img)
image_input_3 = Input(shape=input_shape_img)
image_input_4 = Input(shape=input_shape_img)
feature_input_1 = Input(shape=input_shape_feat)
feature_input_2 = Input(shape=input_shape_feat)
feature_input_3 = Input(shape=input_shape_feat)
feature_input_4 = Input(shape=input_shape_feat)
roi_input_1 = Input(shape=(None, 4))
roi_input_2 = Input(shape=(None, 4))
roi_input_3 = Input(shape=(None, 4))
roi_input_4 = Input(shape=(None, 4))

shared_layers_merged_A, shared_layers_x1_A, shared_layers_x2_A  = nn.mlmt_base_nn(image_input_1, image_input_2, trainable=True)
shared_layers_merged_B, shared_layers_x1_B, shared_layers_x2_B  = nn.mlmt_base_nn(image_input_3, image_input_4, trainable=True)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers_A = nn.rpn(shared_layers_merged_A, shared_layers_x1_A, shared_layers_x2_A, num_anchors)
rpn_layers_B = nn.rpn(shared_layers_merged_B, shared_layers_x1_B, shared_layers_x2_B, num_anchors)

classifier_A = nn.classifier(feature_input_1, feature_input_2, roi_input_1, roi_input_2, C.num_rois, nb_classes=len(class_mapping), trainable=True)
classifier_B = nn.classifier(feature_input_3, feature_input_4, roi_input_3, roi_input_4, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn_A = Model([image_input_1, image_input_2], rpn_layers_A)
model_rpn_B = Model([image_input_3, image_input_4], rpn_layers_B)
model_classifier_A = Model([feature_input_1, feature_input_2, roi_input_1, roi_input_2], classifier_A)
model_classifier_B = Model([feature_input_3, feature_input_4, roi_input_3, roi_input_4], classifier_B)

print('Loading RPNs, A and B, weights from {} and {}'.format(RPN_model_path_A, RPN_model_path_B))
print('Loading Det, A and B, weights from {} and {}'.format(DET_model_path_A, DET_model_path_B))
model_rpn_A.load_weights(RPN_model_path_A, by_name=True)
model_classifier_A.load_weights(DET_model_path_A, by_name=True)
model_rpn_B.load_weights(RPN_model_path_B, by_name=True)
model_classifier_B.load_weights(DET_model_path_B, by_name=True)

model_rpn_A.compile(optimizer='sgd', loss='mse')
model_classifier_A.compile(optimizer='sgd', loss='mse')
model_rpn_B.compile(optimizer='sgd', loss='mse')
model_classifier_B.compile(optimizer='sgd', loss='mse')

MLMT_Unet = get_mlmt_unet(n_classes=MLMT_UNet_n_classes,
                          input_height=MLMT_UNet_input_height, input_width=MLMT_UNet_input_width,
                          merge='add')

print('Loading MLMT UNet weights from {}'.format(MLMT_UNet_model_path))
MLMT_Unet.load_weights(MLMT_UNet_model_path)

MLMT_Unet.compile(optimizer='sgd', loss='mse')

all_images,  _, _  = get_data(options.test_path_1, options.test_path_2,
                              options.test_path_3, options.test_path_4,
                              images_dir=test_images_dir,
                              spect_1=spect_1, spect_2=spect_2,
                              spect_3=spect_3, spect_4=spect_4,
                              data_set = 'test')

test_imgs = [s for s in all_images if s['imageset'] == 'test']

T = {}
P = {}
cnt_TP_1, cnt_FP_1, cnt_FN_1 = 0, 0, 0
cnt_TP_2, cnt_FP_2, cnt_FN_2 = 0, 0, 0
cnt_TP_3, cnt_FP_3, cnt_FN_3 = 0, 0, 0
cnt_TP_4, cnt_FP_4, cnt_FN_4 = 0, 0, 0

st = time.time()
log = []
for idx, img_data in enumerate(test_imgs):
    print('processing {}/{}'.format(idx, len(test_imgs)))

    Spect_1_filepath = img_data['filepath']
    image_ID = Spect_1_filepath.split('/')[-1].split('_')[0]
    img_Spect_1_raw, img_Spect_1, image_1_name = get_det_image(test_images_dir, spect_1, image_ID)
    img_Spect_2_raw, img_Spect_2, image_2_name = get_det_image(test_images_dir, spect_2, image_ID)
    img_Spect_3_raw, img_Spect_3, image_3_name = get_det_image(test_images_dir, spect_3, image_ID)
    img_Spect_4_raw, img_Spect_4, image_4_name = get_det_image(test_images_dir, spect_4, image_ID)

    img_1_copy = img_Spect_1.copy()
    img_2_copy = img_Spect_2.copy()
    img_3_copy = img_Spect_3.copy()
    img_4_copy = img_Spect_4.copy()

    X_1, fx_1, fy_1 = format_img(img_Spect_1, C)
    X_2, fx_2, fy_2 = format_img(img_Spect_2, C)
    X_3, fx_3, fy_3 = format_img(img_Spect_3, C)
    X_4, fx_4, fy_4 = format_img(img_Spect_4, C)

    _, ratio = format_img_size(img_Spect_1, C)

    if K.image_dim_ordering() == 'tf':
        X_1 = np.transpose(X_1, (0, 2, 3, 1))
        X_2 = np.transpose(X_2, (0, 2, 3, 1))
        X_3 = np.transpose(X_3, (0, 2, 3, 1))
        X_4 = np.transpose(X_4, (0, 2, 3, 1))

    start = time.time()

    [Y1_1, Y2_1, Y1_2, Y2_2, F1, F2] = model_rpn_A.predict([X_1, X_2])
    result_1 = roi_helpers.rpn_to_roi(Y1_1, Y2_1, C, K.image_dim_ordering(), overlap_thresh=1, max_boxes=150)
    result_2 = roi_helpers.rpn_to_roi(Y1_2, Y2_2, C, K.image_dim_ordering(), overlap_thresh=1, max_boxes=150)

    [Y1_3, Y2_3, Y1_4, Y2_4, F3, F4] = model_rpn_B.predict([X_3, X_4])
    result_3 = roi_helpers.rpn_to_roi(Y1_3, Y2_3, C, K.image_dim_ordering(), overlap_thresh=1, max_boxes=150)
    result_4 = roi_helpers.rpn_to_roi(Y1_4, Y2_4, C, K.image_dim_ordering(), overlap_thresh=1, max_boxes=150)

    if merge_res_across_RPNs:
        temp_mul_res = np.concatenate((result_1, result_2, result_3, result_4), axis=0)
        temp_mul_res[:, 2] -= temp_mul_res[:, 0]
        temp_mul_res[:, 3] -= temp_mul_res[:, 1]
        mul_res_A = temp_mul_res
        mul_res_B = temp_mul_res
    else:
        mul_res_A = np.concatenate((result_1, result_2), axis=0)
        mul_res_A[:, 2] -= mul_res_A[:, 0]
        mul_res_A[:, 3] -= mul_res_A[:, 1]
        mul_res_B = np.concatenate((result_3, result_4), axis=0)
        mul_res_B[:, 2] -= mul_res_B[:, 0]
        mul_res_B[:, 3] -= mul_res_B[:, 1]

    bboxes_1 = {}
    bboxes_2 = {}
    bboxes_3 = {}
    bboxes_4 = {}
    probs_1 = {}
    probs_2 = {}
    probs_3 = {}
    probs_4 = {}

    for jk in range(mul_res_A.shape[0] // C.num_rois + 1):
        ROIs_A, short_A = pad_rois(mul_res_A, C, jk)
        if short_A :
            break
        [P_cls_1, P_regr_1, P_cls_2, P_regr_2] = model_classifier_A.predict([F1, F2, ROIs_A, ROIs_A])
        bboxes_1, probs_1 = process_dets(P_cls_1, P_regr_1, ROIs_A, bboxes_1, probs_1, bbox_threshold, C, class_mapping)
        bboxes_2, probs_2 = process_dets(P_cls_2, P_regr_2, ROIs_A, bboxes_2, probs_2, bbox_threshold, C, class_mapping)
    for jk in range(mul_res_B.shape[0] // C.num_rois + 1):
        ROIs_B, short_B = pad_rois(mul_res_B, C, jk)
        if short_B :
            break
        [P_cls_3, P_regr_3, P_cls_4, P_regr_4] = model_classifier_B.predict([F3, F4, ROIs_B, ROIs_B])
        bboxes_3, probs_3 = process_dets(P_cls_3, P_regr_3, ROIs_B, bboxes_3, probs_3, bbox_threshold, C, class_mapping)
        bboxes_4, probs_4 = process_dets(P_cls_4, P_regr_4, ROIs_B, bboxes_4, probs_4, bbox_threshold, C, class_mapping)

    all_dets_1 = perform_nms(bboxes=bboxes_1, probs=probs_1, spect_ID=1)
    all_dets_2 = perform_nms(bboxes=bboxes_2, probs=probs_2, spect_ID=2)
    all_dets_3 = perform_nms(bboxes=bboxes_3, probs=probs_3, spect_ID=3)
    all_dets_4 = perform_nms(bboxes=bboxes_4, probs=probs_4, spect_ID=4)

    t_1, p_1, TP_1, FP_1, FN_1 = get_map(img_name = image_1_name,
                                  img = img_1_copy,
                                  pred = all_dets_1,
                                  gt = img_data['bboxes'],
                                  f = (fx_1, fy_1),
                                  spect_ID = spect_1,
                                  ratio = ratio,
                                  vis_dir=vis_dir)
    t_2, p_2, TP_2, FP_2, FN_2 = get_map(img_name = image_2_name,
                                  img = img_2_copy,
                                  pred = all_dets_2,
                                  gt = img_data['bboxes_2'],
                                  f = (fx_2, fy_2),
                                  spect_ID = spect_2,
                                  ratio = ratio,
                                  vis_dir=vis_dir)
    t_3, p_3, TP_3, FP_3, FN_3 = get_map(img_name = image_3_name,
                                  img = img_3_copy,
                                  pred = all_dets_3,
                                  gt = img_data['bboxes_3'],
                                  f = (fx_3, fy_3),
                                  spect_ID = spect_3,
                                  ratio = ratio,
                                  vis_dir=vis_dir)
    t_4, p_4, TP_4, FP_4, FN_4 = get_map(img_name = image_4_name,
                                  img = img_4_copy,
                                  pred = all_dets_4,
                                  gt = img_data['bboxes_4'],
                                  f = (fx_4, fy_4),
                                  spect_ID = spect_4,
                                  ratio = ratio,
                                  vis_dir=vis_dir)

    cnt_TP_1 += TP_1
    cnt_FP_1 += FP_1
    cnt_FN_1 += FN_1
    cnt_TP_2 += TP_2
    cnt_FP_2 += FP_2
    cnt_FN_2 += FN_2
    cnt_TP_3 += TP_3
    cnt_FP_3 += FP_3
    cnt_FN_3 += FN_3
    cnt_TP_4 += TP_4
    cnt_FP_4 += FP_4
    cnt_FN_4 += FN_4

    all_bands_dets = []
    all_bands_dets.extend(all_dets_1+all_dets_2+all_dets_3+all_dets_4)

    X_test_284 = []
    X_test_171 = []
    X_test_195 = []
    X_test_304 = []

    X_pred_284 = []
    X_pred_171 = []
    X_pred_195 = []
    X_pred_304 = []

    for box_idx in all_bands_dets:
        pred_box = box_idx
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']

        pred_prob = pred_box['prob']

        int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
        real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio,
                                                                                      int_pred_x1, int_pred_y1,
                                                                                      int_pred_x2, int_pred_y2)

        patch_band_284 = img_Spect_1_raw[real_pred_y1: real_pred_y2, real_pred_x1: real_pred_x2]
        patch_band_171 = img_Spect_2_raw[real_pred_y1: real_pred_y2, real_pred_x1: real_pred_x2]
        patch_band_195 = img_Spect_3_raw[real_pred_y1: real_pred_y2, real_pred_x1: real_pred_x2]
        patch_band_304 = img_Spect_4_raw[real_pred_y1: real_pred_y2, real_pred_x1: real_pred_x2]

        real_W = real_pred_x2 - real_pred_x1
        real_H = real_pred_y2 - real_pred_y1
        pred_box['height'] = real_H
        pred_box['width'] = real_W

        patch_original_shape = patch_band_284.shape

        patch_band_284 = format_segmentation_img(patch_band_284, MLMT_UNet_input_width, MLMT_UNet_input_height)
        patch_band_171 = format_segmentation_img(patch_band_171, MLMT_UNet_input_width, MLMT_UNet_input_height)
        patch_band_195 = format_segmentation_img(patch_band_195, MLMT_UNet_input_width, MLMT_UNet_input_height)
        patch_band_304 = format_segmentation_img(patch_band_304, MLMT_UNet_input_width, MLMT_UNet_input_height)

        X_test_284.append(patch_band_284)
        X_test_171.append(patch_band_171)
        X_test_195.append(patch_band_195)
        X_test_304.append(patch_band_304)

    X_test_284 = np.array(X_test_284)
    X_test_171 = np.array(X_test_171)
    X_test_195 = np.array(X_test_195)
    X_test_304 = np.array(X_test_304)

    y_pred_284, y_pred_195, y_pred_171, y_pred_304 = MLMT_Unet.predict([X_test_284, X_test_195, X_test_171, X_test_304])
    y_predi_284 = np.argmax(y_pred_284, axis=3)
    y_predi_171 = np.argmax(y_pred_171, axis=3)
    y_predi_195 = np.argmax(y_pred_195, axis=3)
    y_predi_304 = np.argmax(y_pred_304, axis=3)

    MLMT_Unet_AR_class_ID = 2
    img_vis = np.uint8(img_Spect_1)

    img_vis_1 = img_Spect_1.copy()
    img_vis_2 = img_Spect_2.copy()
    img_vis_3 = img_Spect_3.copy()
    img_vis_4 = img_Spect_4.copy()
    contours_284 = []
    contours_195 = []
    contours_171 = []
    contours_304 = []

    for box_idx in range(0, len(all_bands_dets)):
        pred_box = all_bands_dets[box_idx]
        pred_class = pred_box['class']
        pred_x1 = pred_box['x1']
        pred_x2 = pred_box['x2']
        pred_y1 = pred_box['y1']
        pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']

        int_pred_x1, int_pred_y1, int_pred_x2, int_pred_y2 = int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)
        real_pred_x1, real_pred_y1, real_pred_x2, real_pred_y2 = get_real_coordinates(ratio,
                                                                                      int_pred_x1, int_pred_y1,
                                                                                      int_pred_x2, int_pred_y2)
        if pred_box['spect_ID'] == '1':
            img_vis_284, contours = visualize_detections(img_vis=img_vis_1, y_predi=y_predi_284,
                                                         box_idx=box_idx, class_ID=MLMT_Unet_AR_class_ID, pred_box=pred_box,
                                                         x1=real_pred_x1, y1=real_pred_y1, x2=real_pred_x2, y2=real_pred_y2,
                                                         vis_contours=True,
                                                         ret_contours=True)
        elif pred_box['spect_ID'] == '2':
            img_vis_171, contours = visualize_detections(img_vis=img_vis_2, y_predi=y_predi_171,
                                                         box_idx=box_idx, class_ID=MLMT_Unet_AR_class_ID, pred_box=pred_box,
                                                         x1=real_pred_x1, y1=real_pred_y1, x2=real_pred_x2, y2=real_pred_y2,
                                                         vis_contours=True,
                                                         ret_contours=True)
        elif pred_box['spect_ID'] == '3':
            img_vis_195, contours = visualize_detections(img_vis=img_vis_3, y_predi=y_predi_195,
                                                         box_idx=box_idx, class_ID=MLMT_Unet_AR_class_ID, pred_box=pred_box,
                                                         x1=real_pred_x1, y1=real_pred_y1, x2=real_pred_x2, y2=real_pred_y2,
                                                         vis_contours=True,
                                                         ret_contours=True)
        elif pred_box['spect_ID'] == '4':
            img_vis_304, contours = visualize_detections(img_vis=img_vis_4, y_predi=y_predi_304,
                                                         box_idx=box_idx, class_ID=MLMT_Unet_AR_class_ID, pred_box=pred_box,
                                                         x1=real_pred_x1, y1=real_pred_y1, x2=real_pred_x2, y2=real_pred_y2,
                                                         vis_contours=True,
                                                         ret_contours=True)
    save_contour(vis_dir, spect_1, image_1_name, contours_284)
    save_contour(vis_dir, spect_2, image_2_name, contours_171)
    save_contour(vis_dir, spect_3, image_3_name, contours_195)
    save_contour(vis_dir, spect_4, image_4_name, contours_304)

    save_image(img_vis_1, vis_dir, spect_1, image_1_name)
    save_image(img_vis_2, vis_dir, spect_2, image_2_name)
    save_image(img_vis_3, vis_dir, spect_3, image_3_name)
    save_image(img_vis_4, vis_dir, spect_4, image_4_name)

calc_F1(cnt_TP_1, cnt_FP_1, cnt_FN_1, spect=spect_1, report=True)
calc_F1(cnt_TP_2, cnt_FP_2, cnt_FN_2, spect=spect_2, report=True)
calc_F1(cnt_TP_3, cnt_FP_3, cnt_FN_3, spect=spect_3, report=True)
calc_F1(cnt_TP_4, cnt_FP_4, cnt_FN_4, spect=spect_4, report=True)

print('Elapsed time = {}'.format(time.time() - st))




