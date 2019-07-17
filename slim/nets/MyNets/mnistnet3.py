# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib import slim

import tensorflow as tf

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def mnistnet(images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='MnistNet'):
    end_points = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}, scope=scope):
        conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
        conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
        conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
        net = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(
                axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net

        conv3_4 = slim.conv2d(net, 512, [3, 3], padding='SAME', scope='conv3_4')
        conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

        flatten = slim.flatten(max_pool_4)
        fc1 = slim.fully_connected(slim.dropout(flatten, dropout_keep_prob), 1024,
                                   activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(slim.dropout(fc1, dropout_keep_prob), num_classes, activation_fn=None,
                                      scope='fc2')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


mnistnet.default_image_size = 28


def mnistnet_arg_scope(weight_decay=0.004):
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            activation_fn=tf.nn.relu):
        with slim.arg_scope(
                [slim.fully_connected],
                biases_initializer=tf.constant_initializer(0.1),
                weights_initializer=trunc_normal(0.04),
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu) as sc:
            return sc
