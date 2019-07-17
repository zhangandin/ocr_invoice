# -*- coding:UTF-8 -*-
import cv2
from PIL import Image
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
import math
import glob

index = 0


def show(img):
    global index
    index += 1
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(index)
        plt.imshow(image)
        plt.show()
    if len(img.shape) == 2:
        plt.title(index)
        plt.imshow(img, cmap=plt.gray())
        plt.show()


def find_image_cbox(img):
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    img = cv2.bitwise_not(img)
    cropped_box = find_image_bbox(img)
    return cropped_box


def find_image_bbox(img):
    height = img.shape[0]
    width = img.shape[1]
    v_sum = np.sum(img, axis=0)
    h_sum = np.sum(img, axis=1)
    left = 0
    right = width - 1
    top = 0
    low = height - 1
    # 从左往右扫描，遇到非零像素点就以此为字体的左边界
    for i in range(width):
        if v_sum[i] > 0:
            left = i
            break
    # 从右往左扫描，遇到非零像素点就以此为字体的右边界
    for i in range(width - 1, -1, -1):
        if v_sum[i] > 0:
            right = i
            break
    # 从上往下扫描，遇到非零像素点就以此为字体的上边界
    for i in range(height):
        if h_sum[i] > 0:
            top = i
            break
    # 从下往上扫描，遇到非零像素点就以此为字体的下边界
    for i in range(height - 1, -1, -1):
        if h_sum[i] > 0:
            low = i
            break
    return left, top, right, low


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


def same_scale2(cv2_img, max_size):
    cur_height, cur_width = cv2_img.shape[:2]
    ratio_h = float(max_size) / float(cur_height)
    new_width = int(cur_width*ratio_h)
    if new_width == 0:
        new_width = 1
    resized_img = cv2.resize(cv2_img, (new_width, max_size))
    return resized_img


def img_crop2(img):
    img = img.copy()
    left, upper, right, lower = find_image_bbox(img)
    img = img[upper: lower + 1, left: right + 1]
    return img


def img_item_crop(img):
    img = img.copy()

    def find_edge(thresh, data):
        length = len(data)
        start = 0
        end = 0
        for i in range(length):
            if data[i] >= thresh:
                start = i
                break
        for i in range(length - 1, -1, -1):
            if data[i] >= thresh:
                end = i
                break
        return start, end

    h, w = img.shape[0], img.shape[1]
    cols = [0 for _ in range(w)]
    rows = [0 for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                cols[j] += 1
                rows[i] += 1
    ls = np.array(cols)
    ls = ls[ls > 0]
    thresh = ls.min(axis=0)
    left, right = find_edge(thresh, cols)
    ls = np.array(rows)
    ls = ls[ls > 0]
    thresh = ls.min(axis=0)
    top, bottom = find_edge(thresh, rows)
    img = img[top:bottom + 1, left:right + 1]
    return img, (top, bottom, left, right)


def horizontal_crop(img):
    img = img.copy()

    def find_edge(thresh, data):
        length = len(data)
        start = 0
        end = length-1
        half = int(length / 2)
        for i in range(half, -1, -1):
            if data[i] < thresh:
                start = i
                break
        for i in range(half + 1, length, 1):
            if data[i] < thresh:
                end = i
                break
        return start, end

    h, w = img.shape[0], img.shape[1]
    rows = [0 for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                rows[i] += 1
    thresh = 1
    top, bottom = find_edge(thresh, rows)
    img = img[top:bottom + 1, :]
    return img, (top, bottom)


def horizontal_crop2(img):
    img = img.copy()

    def find_edge(thresh, data):
        length = len(data)
        start = 0
        end = 0
        last = 0
        final_start = 0
        final_end = 0
        flag = False
        for i in range(length):
            value = data[i]
            if value >= thresh:
                if flag:
                    if i < length - 1:
                        continue
                    else:
                        end = i
                        if end - start > last:
                            last = end - start
                            final_start = start
                            final_end = end
                        flag = False

                else:
                    flag = True
                    start = i
            else:
                if flag:
                    end = i
                    if end - start > last:
                        last = end - start
                        final_start = start
                        final_end = end
                    flag = False

        return final_start, final_end

    h, w = img.shape[0], img.shape[1]
    rows = [0 for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                rows[i] += 1
    ls = np.array(rows)
    ls = ls[ls > 0]
    thresh = ls.min(axis=0)
    top, bottom = find_edge(thresh, rows)
    img = img[top:bottom + 1, :]
    return img, (top, bottom)


def vertical_crop(img):
    img = img.copy()

    def find_edge(thresh, data):
        length = len(data)
        start = 0
        end = length-1
        for i in range(length):
            if data[i] >= thresh:
                start = i
                break
        for i in range(length - 1, -1, -1):
            if data[i] >= thresh:
                end = i
                break
        return start, end

    h, w = img.shape[0], img.shape[1]
    cols = [0 for _ in range(w)]
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                cols[j] += 1

    ls = np.array(cols)
    ls = ls[ls > 0]
    thresh = ls.min(axis=0)
    left, right = find_edge(thresh, cols)
    img = img[:, left:right + 1]
    return img, (left, right)


TARGET_HEIGHT = 30
TARGET_WIDTH = 200

class OcrSpliter(object):
    def __init__(self):
        self.num = 0

    def DegreeTrans(self,theda):
        return theda / np.pi * 180

    def getNewSize(self,img, angle):
        q_height, q_width = img.shape[0], img.shape[1]
        angle = angle * math.pi / 180.0
        s = math.fabs(math.sin(angle))
        c = math.fabs(math.cos(angle))
        width = int((q_height * s + q_width * c) / 4) * 4
        height = int((q_height * c + q_width * s) / 4) * 4
        return width, height

    def rotate(self, img, rotate):
        img = img.copy()
        b, g, r = img[5, 5]
        height, width = img.shape[0], img.shape[1]
        if rotate != 0:
            M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rotate, 1.0)
            new_width, new_height = self.getNewSize(img, rotate)
            img = cv2.warpAffine(img, M, (new_width, new_height), borderValue=(int(b), int(g), int(r)))
        return img

    def get_x_y(self, x, y, angle):
        angle = angle * math.pi / 180.0
        x1 = x + math.sqrt(x ** 2 + y ** 2) * math.sin(angle)
        y1 = y
        return [x1, y1]


    def crop(self, path, img_src=None):
        if img_src is not None:
            src = img_src.copy()
        else:
            src = cv2.imread(path)
        # show(src)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel)
        # show(img)
        img = cv2.bitwise_not(img)
        # show(img)
        _, (top, bottom) = horizontal_crop(img)
        if bottom - top < 10:
            _, (top, bottom) = horizontal_crop2(img)
        # show(img)
        img = src[top:bottom + 1, :]
        # show(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        img, (left, right) = vertical_crop(img)
        # img = src[top:bottom + 1, left:right + 1]
        return img

    def ocr_split(self, path, img_src=None):
        if img_src is not None:
            img = self.crop('', img_src)
        else:
            img = self.crop(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        # img = cv2.bitwise_not(img)
        img = same_scale(img, TARGET_WIDTH, TARGET_HEIGHT)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # img = cv2.dilate(img, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # img = cv2.erode(img, kernel)
        img = img_padding(img)
        # show(img)
        return img


def img_padding(img):
    left = TARGET_WIDTH - img.shape[1] - 1
    top = TARGET_HEIGHT - img.shape[0] - 1
    if left < 0 and top < 0:
        return img
    final_img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
    # if left<=0:
    #     start_x=0
    # else:
    #     start_x = int(left / 2)
    start_x = 0
    if top <= 0:
        start_y = 0
    else:
        start_y = int(top / 2)
    img_h, img_w = img.shape[0], img.shape[1]
    final_img[start_y:start_y + img_h, start_x:start_x + img_w] = img
    return final_img




# spliter = OcrSpliter()
#
#
# path = 'IMG_20181120_144931_5.jpg'
# spliter.ocr_split(path)


# paths = glob.glob('images/*.*g')
# print(len(paths))
# for i, path in enumerate(paths):
#     spliter.ocr_split(path)
