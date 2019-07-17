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



TARGET_SIZE = 48


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

    def line_detection(self, image):
        src = image.copy()
        img_r = src.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # show(gray)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # show(edges)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
        if lines is not None:
            sum = 0
            for line in lines:
                rho, theta = line[0]
                sum += theta
            # show(image)
            avg = sum / len(lines)
            angle = self.DegreeTrans(avg) - 90
            # print(angle)
            img_r = self.rotate(src, angle)
            # show(img_r)
        padding = 30
        img = np.ones([img_r.shape[0], img_r.shape[1] + padding * 2, 3], dtype=np.uint8) * 255
        img[:, padding:padding+img_r.shape[1]] = img_r
        # show(img)
        height, width = img.shape[0], img.shape[1]
        p1 = [0, 0]
        p2 = [int((width - 1)/2), int((height-1)/2)]
        p3 = [0, height - 1]
        matSrc = np.float32([p1,p2,p3])  # 需要注意的是 行列 和 坐标 是不一致的
        img_src = img

        def get_count(angle):
            matDst = np.float32(
                [self.get_x_y(p1[0], p1[1], angle), p2,
                 self.get_x_y(p3[0], p3[1], -angle)])
            matAffine = cv2.getAffineTransform(matSrc, matDst)  # mat 1 src 2 dst 形成组合矩阵
            img = cv2.warpAffine(img_src, matAffine, (img_src.shape[1], img_src.shape[0]), borderValue=(255, 255, 255))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            img = cv2.erode(img, kernel)
            # show(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # show(img)
            ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
            # show(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            img = cv2.erode(img, kernel)
            # show(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img = cv2.dilate(img, kernel)
            # show(img)
            img = cv2.bitwise_not(img)
            # show(img)
            img, _ = img_crop2(img)
            h, w = img.shape[0], img.shape[1]
            count = 0
            for i in range(w):
                for j in range(h):
                    if img[j, i] == 255:
                        count += 1
                        break
            if count == 0:
                count = 10000
            return count

        flag = True
        left = -10
        right = 10
        now = 0
        left_count = get_count(left)
        now_count = get_count(now)
        right_count = get_count(right)
        if left_count > now_count and right_count > now_count:
            flag = False

        if flag:
            left = -30
            right = 30
            now = 0
            while left <= right-1:
                left_count = get_count(left)
                now_count = get_count(now)
                right_count = get_count(right)
                if left_count < now_count and right_count < now_count:
                    if abs(left_count - now_count) > abs(right_count - now_count):
                        right = now
                        now = (left + right) / 2
                    else:
                        left = now
                        now = (left + right) / 2
                elif left_count < now_count:
                    right = now
                    now = (left + right) / 2
                elif right_count < now_count:
                    left = now
                    now = (left + right) / 2
                else:
                    left += 2
                    right -= 2

        rigth_angle = now
        print('rigth_angle:', rigth_angle)

        if rigth_angle == 0:
            return img_r
        else:
            matDst = np.float32(
                [self.get_x_y(p1[0], p1[1], rigth_angle), p2,
                 self.get_x_y(p3[0], p3[1], -rigth_angle)])
            matAffine = cv2.getAffineTransform(matSrc, matDst)  # mat 1 src 2 dst 形成组合矩阵
            img = cv2.warpAffine(img_src, matAffine, (img_src.shape[1], img_src.shape[0]), borderValue=(255, 255, 255))
            return img

    # def ocr_split(self, path):
    #     src = cv2.imread(path)
    #     src = same_scale(src,200,30)
    #     # show(src)
    #     # pure_name = os.path.basename(path).split('.')[0]
    #     # dir_name = os.path.join('images', pure_name)
    #     # shutil.rmtree(dir_name, ignore_errors=True)
    #     # os.mkdir(dir_name)
    #     img = src.copy()
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     img = cv2.erode(img, kernel)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     img = cv2.dilate(img, kernel)
    #     img = cv2.bitwise_not(img)
    #     img = img_padding(img)
    #     # show(img)
    #     img = img_crop2(img)
    #     img = same_scale(img, 200, 30)
    #     _, img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
    #     img = img_padding(img)
    #     # show(img)
    #     # src = src[top:bottom + 1, left:right + 1]
    #     # src = same_scale(src, 400, 60)
    #     # show(src)
    #     return img

    def ocr_split(self, path, img_src=None):
        if img_src is not None:
            src = img_src.copy()
        else:
            src = cv2.imread(path)
        src = same_scale(src, 200, 30)
        # show(src)
        # pure_name = os.path.basename(path).split('.')[0]
        # dir_name = os.path.join('images', pure_name)
        # shutil.rmtree(dir_name, ignore_errors=True)
        # os.mkdir(dir_name)
        # img = src.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = src[:, :, 2]
        # show(img)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
        # show(img)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # img = cv2.dilate(img, kernel)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # img = cv2.erode(img, kernel)
        img = cv2.bitwise_not(img)
        img = img_padding(img)
        # show(img)
        img = img_crop2(img)
        img = same_scale(img, 200, 30)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = img_padding(img)
        # show(img)
        # src = src[top:bottom + 1, left:right + 1]
        # src = same_scale(src, 400, 60)
        # show(src)
        return img




def img_padding(img):
    TARGET_HEIGHT = 30
    TARGET_WIDTH = 200
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
    if top<=0:
        start_y=0
    else:
        start_y = int(top / 2)
    img_h, img_w = img.shape[0], img.shape[1]
    final_img[start_y:start_y + img_h, start_x:start_x + img_w] = img
    return final_img




# spliter = OcrSpliter()
#
#
# path = 'IMG_20181120_145731_21.jpg'
# spliter.ocr_split(path)


# paths = glob.glob('images/*.*g')
# print(len(paths))
# for i, path in enumerate(paths):
#     spliter.ocr_split(path)
