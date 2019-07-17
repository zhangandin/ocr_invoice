# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

import numpy as np
import tensorflow as tf
from preprocessing import preprocessing_factory
import cv2
from tensorflow.python.client import timeline
from utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt
import time
import os
import math
import img_text_crop
import copy
import ocr_code_utils
import ocr_number_utils
import ocr_price_utils
import ocr_date_utils

img_index = 0

def show(img):
    global img_index
    img_index += 1
    if len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.title(img_index)
        plt.imshow(image)
        plt.show()
    if len(img.shape) == 2:
        plt.title(img_index)
        plt.imshow(img, cmap=plt.gray())
        plt.show()

from pse import pse


logger.setLevel(cfg.debug)


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    #ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w


    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time()-start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals, timer

def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0]*255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()


def rotate(img, degree):
    degree = -degree
    img = img.copy()
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(
        width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(
        height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew))
    return imgRotation

def sparse_tuple_from(sequences, dtype=np.int32):
    """得到一个list的稀疏表示，为了直接将数据赋值给tensorflow的tf.sparse_placeholder稀疏矩阵
    Args:
        sequences: 序列的列表
    Returns:
        一个三元组，和tensorflow的tf.sparse_placeholder同结构
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def sparse_tuple_to(tuple_input, max_len):
    indices = tuple_input[0]
    values = tuple_input[1]
    results = []
    size = len(indices)
    i = 0
    while i < size:
        index = indices[i][0]
        item = []
        j = i
        next_index = indices[j][0]
        while next_index == index:
            item.append(values[j])
            j += 1
            if j == size:
                break
            next_index = indices[j][0]
        results.append(item)
        i = j
    while len(results) < max_len:
        results.append([])
    return results


labels_angle = {0: 0, 1: 90, 2: 180, 3: 270}

labels_field = {0: 'invoice_code',
                1: 'invoice_number',
                2: 'invoice_price',
                3: 'invoice_date',
                4: 'invoice_other'}

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
labels_code = {}
labels_code[0] = ' '
for i, ch in enumerate(letters):
    labels_code[i + 1] = ch
labels_code[len(labels_code) + 1] = '<BLANK>'
max_sequence_length_code = 10

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
labels_number = {}
labels_number[0] = ' '
for i, ch in enumerate(letters):
    labels_number[i + 1] = ch
labels_number[len(labels_number) + 1] = '<BLANK>'
max_sequence_length_number = 8

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '￥']
labels_price = {}
labels_price[0] = ' '
for i, ch in enumerate(letters):
    labels_price[i + 1] = ch
labels_price[len(labels_price) + 1] = '<BLANK>'
max_sequence_length_price = 8

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '年', '月', '日']
labels_date = {}
labels_date[0] = ' '
for i, ch in enumerate(letters):
    labels_date[i + 1] = ch
labels_date[len(labels_date) + 1] = '<BLANK>'
max_sequence_length_date = 11

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    input_dir = '../test_images'
    output_dir = '../test_images_result'
    img_path = '../test_images/IMG_20190107_103022.jpg'
    # img_path = '../test_images/IMG_20190108_162848.jpg'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    g1 = tf.Graph()  # 加载到Session 1的graph
    sess1 = tf.Session(graph=g1, config=config)  # Session1
    # 加载第一个模型
    with sess1.as_default():
        with g1.as_default():
            with tf.gfile.FastGFile('../models/ocr_angle.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')
            preprocessing_name1 = 'resnet_v1_50'
            image_preprocessing_fn1 = preprocessing_factory.get_preprocessing(
                preprocessing_name1,
                is_training=False)
            test_image_size1 = 224
            graph = tf.get_default_graph()
            predictions1 = graph.get_tensor_by_name('resnet_v1_50/predictions/Reshape_1:0')
            tensor_input1 = graph.get_tensor_by_name('Placeholder:0')
            tensor_item1 = tf.placeholder(tf.float32, [None, None, 3])
            processed_image1 = image_preprocessing_fn1(tensor_item1, test_image_size1, test_image_size1)

    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2, config=config)
    with sess2.as_default():
        with g2.as_default():
            with tf.gfile.FastGFile('../models/pse_invoice.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            graph = tf.get_default_graph()
            input_images2 = graph.get_tensor_by_name('input_images:0')
            seg_maps_pred2 = graph.get_tensor_by_name('Sigmoid:0')

    g3 = tf.Graph()
    sess3 = tf.Session(graph=g3, config=config)
    with sess3.as_default():
        with g3.as_default():
            with tf.gfile.FastGFile('../models/ocr_field.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            preprocessing_name3 = 'resnet_v1_50'
            image_preprocessing_fn3 = preprocessing_factory.get_preprocessing(
                preprocessing_name3,
                is_training=False)
            output_height3, output_width3 = 100, 400
            graph = tf.get_default_graph()
            predictions3 = graph.get_tensor_by_name('resnet_v1_50/predictions/Reshape_1:0')
            tensor_input3 = graph.get_tensor_by_name('Placeholder:0')
            tensor_item3 = tf.placeholder(tf.float32, [None, None, 3])
            processed_image3 = image_preprocessing_fn3(tensor_item3, output_height3, output_width3)

    g4 = tf.Graph()
    sess4 = tf.Session(graph=g4, config=config)
    with sess4.as_default():
        with g4.as_default():
            with tf.gfile.FastGFile('../models/ocr_code.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            graph = tf.get_default_graph()
            image_inputs4 = graph.get_tensor_by_name('image_inputs:0')
            logits = graph.get_tensor_by_name('transpose_2:0')
            shape = tf.shape(logits)
            seq_len = tf.reshape(shape[0], [-1])
            seq_len = tf.tile(seq_len, [shape[1]])
            greedy_decoder = tf.nn.ctc_greedy_decoder(logits, seq_len)
            decoded4 = greedy_decoder[0]

    g5 = tf.Graph()
    sess5 = tf.Session(graph=g5, config=config)
    with sess5.as_default():
        with g5.as_default():
            with tf.gfile.FastGFile('../models/ocr_number.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            graph = tf.get_default_graph()
            image_inputs5 = graph.get_tensor_by_name('image_inputs:0')
            logits = graph.get_tensor_by_name('transpose_2:0')
            shape = tf.shape(logits)
            seq_len = tf.reshape(shape[0], [-1])
            seq_len = tf.tile(seq_len, [shape[1]])
            greedy_decoder = tf.nn.ctc_greedy_decoder(logits, seq_len)
            decoded5 = greedy_decoder[0]

    g6 = tf.Graph()
    sess6 = tf.Session(graph=g6, config=config)
    with sess6.as_default():
        with g6.as_default():
            with tf.gfile.FastGFile('../models/ocr_price.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            graph = tf.get_default_graph()
            image_inputs6 = graph.get_tensor_by_name('image_inputs:0')
            logits = graph.get_tensor_by_name('transpose_2:0')
            shape = tf.shape(logits)
            seq_len = tf.reshape(shape[0], [-1])
            seq_len = tf.tile(seq_len, [shape[1]])
            greedy_decoder = tf.nn.ctc_greedy_decoder(logits, seq_len)
            decoded6 = greedy_decoder[0]

    g7 = tf.Graph()
    sess7 = tf.Session(graph=g7, config=config)
    with sess7.as_default():
        with g7.as_default():
            with tf.gfile.FastGFile('../models/ocr_date.pb', 'rb') as f:
                # 使用tf.GraphDef()定义一个空的Graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # Imports the graph from graph_def into the current default Graph.
                tf.import_graph_def(graph_def, name='')

            graph = tf.get_default_graph()
            image_inputs7 = graph.get_tensor_by_name('image_inputs:0')
            logits = graph.get_tensor_by_name('transpose_2:0')
            shape = tf.shape(logits)
            seq_len = tf.reshape(shape[0], [-1])
            seq_len = tf.tile(seq_len, [shape[1]])
            greedy_decoder = tf.nn.ctc_greedy_decoder(logits, seq_len)
            decoded7 = greedy_decoder[0]

    # 使用的时候
    with sess1.as_default():
        with sess1.graph.as_default():
            sess1.run(tf.global_variables_initializer())
            src = cv2.imread(img_path)
            image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (test_image_size1, test_image_size1))
            height, width = image.shape[0], image.shape[1]
            image = np.reshape(image, [height, width, 3])
            image = sess1.run(processed_image1, feed_dict={tensor_item1: image})
            logi = sess1.run(predictions1, feed_dict={tensor_input1: [image]})[0]
            prediction = np.argmax(logi)
            degree = labels_angle[prediction]
            print(img_path, prediction, degree)

    # 使用的时候
    with sess2.as_default():
        with sess2.graph.as_default():
            im_src = rotate(src, degree)
            im = copy.deepcopy(im_src)
            im = im[:, :, ::-1]
            # im = cv2.imread(img_path)[:, :, ::-1]
            logger.debug('image file:{}'.format(img_path))
            os.makedirs(output_dir, exist_ok=True)
            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = resize_image(im)
            h, w, _ = im_resized.shape
            timer = {'net': 0, 'pse': 0}
            start = time.time()
            seg_maps = sess2.run(seg_maps_pred2, feed_dict={input_images2: [im_resized]})
            timer['net'] = time.time() - start

            boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)
            logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(
                img_path, timer['net'] * 1000, timer['pse'] * 1000))

            if boxes is not None:
                boxes = boxes.reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h
                h, w, _ = im.shape
                boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

            duration = time.time() - start_time
            logger.info('[timing] {}'.format(duration))

            # save to file
            if boxes is not None:
                res_file = os.path.join(
                    output_dir,
                    '{}.txt'.format(os.path.splitext(
                        os.path.basename(img_path))[0]))

                with open(res_file, 'w') as f:
                    num = 0
                    for i in range(len(boxes)):
                        # to avoid submitting errors
                        box = boxes[i]
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue
                        num += 1
                        f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                            box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                            box[2, 0], box[2, 1], box[3, 0], box[3, 1]))

                        cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                      color=(255, 255, 0), thickness=2)

                img_text_crop.work(im_src, res_file, output_dir)

            img_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(img_path, im[:, :, ::-1])
            # show_score_geo(im_resized, kernels, im)

    ocr_items = {}
    with sess3.as_default():
        with sess3.graph.as_default():
            sess3.run(tf.global_variables_initializer())
            name = os.path.basename(img_path)
            name = str(name.split(".")[0])
            res_file = os.path.join(
                output_dir,
                '{}.txt'.format(os.path.splitext(
                    os.path.basename(img_path))[0]))
            locations = {}
            with open(res_file, "r", encoding='utf-8') as f:
                for i, line in enumerate(f.readlines()):
                    line = line.split(",")
                    points = [[float(line[0]), float(line[1])],
                              [float(line[2]), float(line[3])],
                              [float(line[4]), float(line[5])],
                              [float(line[6]), float(line[7])]]
                    locations[i] = points
            parent = os.path.join(output_dir, name)
            normal_dir = parent + "_normal/*.jpg"
            paths = glob.glob(normal_dir)
            field_result = parent + '_field.txt'
            f = open(field_result, 'w', encoding='utf-8')
            min_value = 5000
            price_index = 0
            price_path = ''
            boxm = []
            for img_path in paths:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (output_width3, output_height3))
                height, width = image.shape[0], image.shape[1]
                image = np.reshape(image, [height, width, 3])
                image = sess3.run(processed_image3, feed_dict={tensor_item3: image})
                logi = sess3.run(predictions3, feed_dict={tensor_input3: [image]})[0]
                prediction = int(np.argmax(logi))
                if prediction < 4:
                    img_name = str(os.path.basename(img_path).split('.')[0])
                    img_index = int(img_name.split('_')[-1])
                    if prediction != 2:
                        info = '%s, %d, %s, %d' % (img_name, prediction, labels_field[prediction], img_index)
                        print(info)
                        f.write(info + '\n')
                        ocr_items[prediction] = os.path.join(parent, img_name+'.jpg')
                        boxm.append(locations[img_index])
                    else:
                        x = int(locations[img_index][0][0])
                        if x < min_value:
                            min_value = x
                            price_index = img_index
                            price_path = img_path
            img_name = str(os.path.basename(price_path).split('.')[0])
            info = '%s, %d, %s, %d' % (img_name, 2, labels_field[2], price_index)
            print(info)
            f.write(info + '\n')
            ocr_items[2] = os.path.join(parent, img_name + '.jpg')
            f.close()
            boxm.append(locations[price_index])
            boxm = np.asarray(boxm)
            im = copy.deepcopy(im_src)
            im = im[:, :, ::-1]
            for box in boxm:
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(0, 0, 255), thickness=3)
            img_path = os.path.join(output_dir, str(os.path.basename(img_path).split('.')[0])+'-detect.jpg')
            cv2.imwrite(img_path, im[:, :, ::-1])


    with sess4.as_default():
        with sess4.graph.as_default():
            spliter4 = ocr_code_utils.OcrSpliter()
            path = ocr_items[0]
            image = spliter4.ocr_split(path)
            # image = cv2.imread(path, 0)
            imgw = np.asarray(image, dtype=np.float32)
            image = (imgw - np.mean(imgw)) / np.std(imgw)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            feed = {image_inputs4: [image]}
            predictions_result = sess4.run(decoded4[0], feed_dict=feed)
            predictions_result = sparse_tuple_to(predictions_result, max_sequence_length_code)
            predictions_result = predictions_result[0]
            predictions_result = [labels_code[label] for label in predictions_result]
            pred = ''
            for ch in predictions_result:
                if ch not in ['<BLANK>']:
                    pred += ch
            print('invoice_code:', pred)

    with sess5.as_default():
        with sess5.graph.as_default():
            spliter5 = ocr_number_utils.OcrSpliter()
            path = ocr_items[1]
            image = spliter5.ocr_split(path)
            # image = cv2.imread(path, 0)
            imgw = np.asarray(image, dtype=np.float32)
            image = (imgw - np.mean(imgw)) / np.std(imgw)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            feed = {image_inputs5: [image]}
            predictions_result = sess5.run(decoded5[0], feed_dict=feed)
            predictions_result = sparse_tuple_to(predictions_result, max_sequence_length_number)
            predictions_result = predictions_result[0]
            predictions_result = [labels_number[label] for label in predictions_result]
            pred = ''
            for ch in predictions_result:
                if ch not in ['<BLANK>']:
                    pred += ch
            print('invoice_number:',  pred)

    with sess6.as_default():
        with sess6.graph.as_default():
            spliter6 = ocr_price_utils.OcrSpliter()
            path = ocr_items[2]
            image = spliter6.ocr_split(path)
            # image = cv2.imread(path, 0)
            imgw = np.asarray(image, dtype=np.float32)
            image = (imgw - np.mean(imgw)) / np.std(imgw)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            feed = {image_inputs6: [image]}
            predictions_result = sess6.run(decoded6[0], feed_dict=feed)
            predictions_result = sparse_tuple_to(predictions_result, max_sequence_length_price)
            predictions_result = predictions_result[0]
            predictions_result = [labels_price[label] for label in predictions_result]
            pred = ''
            for ch in predictions_result:
                if ch not in ['<BLANK>']:
                    pred += ch
            pred = pred[:-2] + '.' + pred[-2:]
            print('invoice_price:', pred)

    with sess7.as_default():
        with sess7.graph.as_default():
            spliter7 = ocr_date_utils.OcrSpliter()
            path = ocr_items[3]
            image = spliter7.ocr_split(path)
            # image = cv2.imread(path, 0)
            imgw = np.asarray(image, dtype=np.float32)
            image = (imgw - np.mean(imgw)) / np.std(imgw)
            image = np.reshape(image, [image.shape[0], image.shape[1], 1])
            feed = {image_inputs7: [image]}
            predictions_result = sess7.run(decoded7[0], feed_dict=feed)
            predictions_result = sparse_tuple_to(predictions_result, max_sequence_length_date)
            predictions_result = predictions_result[0]
            predictions_result = [labels_date[label] for label in predictions_result]
            pred = ''
            for ch in predictions_result:
                if ch not in ['<BLANK>']:
                    pred += ch
            print('invoice_date:',  pred)

    # 关闭sess
    sess1.close()
    sess2.close()
    sess3.close()
    sess4.close()
    sess5.close()
    sess6.close()
    sess7.close()


if __name__ == '__main__':
    tf.app.run()
