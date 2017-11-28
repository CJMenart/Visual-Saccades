########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
#hacked up by Chris Menart for visual saccades project and 
# refactored to avoid common initialization errors. 10-21-17
#WARNING: No regularization is currently included. However, you have the option to make
#variables trainable or not
#OTHER WARNING: Makes use of Spatial Pyramid Pooling, which breaks on batch sizes greater than 1.
#Code was written assuming fully stochastic training.

import tensorflow as tf
import numpy as np
from poolToFixed import *

class vgg16:
    def __init__(self,inputs,weightFile,trainable,fov):
        
        weights = np.load(weightFile)
        self.inputs = inputs
        
        with tf.variable_scope('vgg') as scope:
            self.convlayers(inputs,weights,trainable)
        
        self.out = poolToFixed(self.conv2_2, fov,'max')
            
    def convlayers(self,inputs,weights,trainable):
        self.parameters = []

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights',shape=[3,3,3,64],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv1_1_W'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            biases = tf.get_variable('biases',shape=[64],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv1_1_b'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights',shape=[3,3,64,64],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv1_2_W'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            biases = tf.get_variable('biases',shape=[64],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv1_2_b'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)        
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights',shape=[3,3,64,128],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv2_1_W'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            biases = tf.get_variable('biases',shape=[128],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv2_1_b'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights',shape=[3,3,128,128],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv2_2_W'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            biases = tf.get_variable('biases',shape=[128],dtype=tf.float32,
                initializer=tf.constant_initializer(weights['conv2_2_b'],dtype=tf.float32,verify_shape=True),
                trainable=trainable,validate_shape=True)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out)
            self.parameters += [kernel, biases]