# coding:utf-8
import cv2
import numpy as np
import os

TARGET_WIDTH = 400
TARGET_HEIGHT = 100

def img_padding(img):
    left = TARGET_WIDTH - img.shape[1] - 1
    top = TARGET_HEIGHT - img.shape[0] - 1
    if left < 0 or top < 0:
        img = same_scale(img, TARGET_WIDTH, TARGET_HEIGHT)
        left = TARGET_WIDTH - img.shape[1] - 1
        top = TARGET_HEIGHT - img.shape[0] - 1
    final_img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    start_x = 0
    if left > 0:
        start_x = int(left / 2)
    start_y = 0
    if top > 0:
        start_y = int(top / 2)
    img_h, img_w = img.shape[0], img.shape[1]
    final_img[start_y:start_y + img_h, start_x:start_x + img_w] = img
    return final_img


def same_scale(cv2_img, max_width, max_height):
    cur_height, cur_width = cv2_img.shape[:2]
    ratio_w = float(max_width) / float(cur_width)
    ratio_h = float(max_height) / float(cur_height)
    ratio = min(ratio_w, ratio_h)

    new_size = (min(int(cur_width * ratio), max_width),
                min(int(cur_height * ratio), max_height))

    new_size = (max(new_size[0], 1),
                max(new_size[1], 1),)
    resized_img = cv2.resize(cv2_img, new_size)
    return resized_img

