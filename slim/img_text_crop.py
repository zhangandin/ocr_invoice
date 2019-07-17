# coding:utf-8

import cv2
import numpy as np
import math
import os
import shutil
import piece_normal

def img_crop(img):
    img = img.copy()
    left = 0
    right = 1
    top = 0
    bottom = 1
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
    thresh = ls.min(axis=0) + 2
    for i in range(w):
        if cols[i] >= thresh:
            left = i
            break
    for i in range(w - 1, -1, -1):
        if cols[i] >= thresh:
            right = i
            break
    ls = np.array(rows)
    ls = ls[ls > 0]
    thresh = ls.min(axis=0) + 2
    for i in range(h):
        if rows[i] >= thresh:
            top = i
            break
    for i in range(h - 1, -1, -1):
        if rows[i] >= thresh:
            bottom = i
            break
    img = img[top:bottom + 1, left:right + 1]
    return img, (top, bottom, left, right)


def item_crop(img_src, points, save_path,  normal_path, save_file=True):
    points = np.asarray([points]).astype(np.int)
    # points = np.reshape(points, [-1, 2])
    # print(points)
    # cv2.drawContours(img_src, points, 0, (0, 0, 255), 2)
    # show(img_src)
    # exit()
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # mask = np.zeros_like(img_src, dtype=np.uint8)
    # cv2.polylines(mask, [box], True, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.fillPoly(mask, [box], (255, 255, 255))
    # result = cv2.bitwise_and(img_src, img_src, mask=mask[:, :, 0])

    x, y, w, h = cv2.boundingRect(box)
    dst = img_src[y:y + h, x:x + w, :]
    degree = rect[2]
    if degree < -45:
        degree += 90
    height, width = dst.shape[:2]
    # 旋转后的尺寸
    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    fill_color = dst[5, 5]

    imgRotation = cv2.warpAffine(dst, matRotation, (widthNew, heightNew),
                                 borderValue=(int(fill_color[0]), int(fill_color[1]), int(fill_color[2])))

    img = imgRotation.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img = cv2.erode(img, kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img = cv2.erode(img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img = cv2.dilate(img, kernel)
    img = cv2.bitwise_not(img)
    # show(img)
    img, (top, bottom, left, right) = img_crop(img)
    # show(img)
    final_result = imgRotation[top:bottom + 1, left:right + 1]
    # show(final_result)
    if save_file:
        cv2.imwrite(save_path, final_result)
    img = piece_normal.img_padding(final_result)
    if save_file:
        cv2.imwrite(normal_path, img)
    return final_result, img


def work(img, input_text, save_dir):
    with open(input_text, "r", encoding='utf-8') as f:
        print(input_text)
        name = os.path.basename(input_text)
        name = str(name.split(".")[0])
        parent = os.path.join(save_dir, name)
        shutil.rmtree(parent, ignore_errors=True)
        os.makedirs(parent, exist_ok=True)
        normal_dir = parent+"_normal"
        shutil.rmtree(normal_dir, ignore_errors=True)
        os.makedirs(normal_dir, exist_ok=True)
        border = 2
        for i, line in enumerate(f.readlines()):
            line = line.split(",")
            points = [[float(line[0])-border, float(line[1])-border],
                      [float(line[2])-border, float(line[3])+border],
                      [float(line[4])+border, float(line[5])+border],
                      [float(line[6])+border, float(line[7])-border]]
            save_path = os.path.join(parent, name + "_" + str(i) + ".jpg")
            normal_path = os.path.join(normal_dir, name + "_" + str(i) + ".jpg")
            item_crop(img, points, save_path, normal_path)


def work_for_array(img, detected_rects={}, img_items={}, img_items_normal={}):
    border = 2
    for key in detected_rects.keys():
        line = detected_rects[key]
        points = [[float(line[0]) - border, float(line[1]) - border],
                  [float(line[2]) - border, float(line[3]) + border],
                  [float(line[4]) + border, float(line[5]) + border],
                  [float(line[6]) + border, float(line[7]) - border]]
        piece, piece_normal = item_crop(img, points, '', '', False)
        img_items[key] = piece
        img_items_normal[key] = piece_normal


def split_list(ls, n):
    if not isinstance(ls, list) or not isinstance(n, int):
        return []
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if ls_len <= n:
        return [ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ### j,j,j,...(前面有n-1个j),j+k
        # 步长j,次数n-1
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        # 算上末尾的j+k
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

