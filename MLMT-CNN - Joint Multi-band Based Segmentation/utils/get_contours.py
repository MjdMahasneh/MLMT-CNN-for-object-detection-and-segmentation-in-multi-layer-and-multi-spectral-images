from utils.provider import get_image_arr, get_segmentation_arr
import matplotlib.pyplot as plt
import cv2
import numpy as np



def get_contours(image, mask, thickness=1, color = (255, 0, 0), class_ID = 2):

    ## binarize :
    mask[np.where(mask != class_ID)] = 0
    mask[np.where(mask != 0)] = 1

    mask = np.uint8(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = np.uint8(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) * 255.0)

    cv2.drawContours(image, contours, -1, color, thickness)

    return image











