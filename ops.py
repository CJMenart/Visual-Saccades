from contextlib import contextmanager
import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.contrib.layers import xavier_initializer

def affine_layer(inputs, out_dim, name = 'affine_layer'):
    in_dim=inputs.get_shape().as_list()[1]
    with tf.variable_scope(name):
        init = tf.random_uniform_initializer(-0.08, 0.08)
        weights = tf.get_variable(name = 'weights',shape = [in_dim,out_dim]
                                , dtype = tf.float32, initializer = init)
        outputs = tf.matmul(inputs, weights)
    return outputs

def conv_layer(inputs, filter_shape, stride, name = 'conv_layer'):
    with tf.variable_scope(name):
        init = tf.contrib.layers.xavier_initializer()
        filter1 = tf.get_variable(name = 'filt_weights', shape = filter_shape, dtype = tf.float32, initializer = init)
        output = tf.nn.conv2d(inputs, filter1, strides = stride, padding = 'SAME')
        return output
def average_gradients(tower_grads):
    """ Calculate the average gradient for each shared variable across towers.
    Note that this function provides a sync point across al towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer
        list is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):

            # each grad is ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:

                # Add 0 dim to gradients to represent tower
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension that we will average over below
                grads.append(expanded_g)

            # Build the tensor and average along tower dimension
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # The Variables are redundant because they are shared across towers
            # just return first tower's pointer to the Variable
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    
@contextmanager
def variables_on_first_device(device_name):
    old_fn = tf.get_variable
    def new_fn(*args, **kwargs):
        with tf.device(device_name):
            return old_fn(*args, **kwargs)
    tf.get_variable = new_fn
    yield
    tf.get_variable = old_fn
    
def scalar_summary(name, x):
    try:
        summ = tf.summary.scalar(name, x)
    except AttributeError:
        summ = tf.scalar_summary(name, x)
    return summ

def histogram_summary(name, x):
    try:
        summ = tf.summary.histogram(name, x)
    except AttributeError:
        summ = tf.histogram_summary(name, x)
    return summ
def leakyrelu(x, alpha=0.3, name='lrelu'):
    with tf.name_scope(name):
        return tf.maximum(x, alpha * x, name=name)
def downconv(x, output_dim, k=[5, 5], pool=[2, 2], name='downconv'):
    """ Downsampled convolution 2d """
    w_init = xavier_initializer()
    with tf.variable_scope(name):
        W = tf.get_variable('W', k + [x.get_shape()[-1], output_dim], initializer=w_init)
        conv = tf.nn.conv2d(x, W, strides=[1] + pool + [1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, b)
        return conv
    
def deconv(x, output_dim, output_shape, k=[5, 5], pool=[2, 2], name='downconv'):
    """ Deconvolution 2d """
    w_init = xavier_initializer()
    with tf.variable_scope(name):
        W = tf.get_variable('W', k + [output_dim, x.get_shape()[-1]], initializer=w_init)
        conv = tf.nn.conv2d_transpose(x, W, strides=[1] + pool + [1], 
                                      output_shape = output_shape, 
                                      padding='SAME')
        b = tf.get_variable('b', [output_dim], 
                            initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, b)
        return conv
    
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(out_file, var_list, name_list):
    dict1 = {}
    for i in range(len(var_list)):
        dict1[name_list[i]] = _bytes_feature(var_list[i].tostring())
    example = tf.train.Example(features = tf.train.Features(feature = dict1))
    out_file.write(example.SerializeToString())
    
def write_tfrecords_val(out_file, var_list, name_list):
    dict1 = {}
    for i in range(3):
        dict1[name_list[i]] = _bytes_feature(var_list[i].tostring())
    dict1[name_list[3]] = _bytes_feature(var_list[3])
    dict1[name_list[4]] = _bytes_feature(var_list[4])
    example = tf.train.Example(features = tf.train.Features(feature = dict1))
    out_file.write(example.SerializeToString())

def read_data(filepath, name_list, shape_list, dtype_list):
    with tf.name_scope('read_data'):
        filename_queue = tf.train.string_input_producer([filepath])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        dict1={}
        for i in range(len(name_list)):
            dict1[name_list[i]] = tf.FixedLenFeature([], tf.string)
        features = tf.parse_single_example(serialized_example, features = dict1)
        outputs = []
        for i in range(len(name_list)):
            temp = tf.decode_raw(features[name_list[i]], dtype_list[i])
            temp = tf.reshape(temp, shape_list[i])
            outputs.append(temp)
        return outputs
def read_val_data(filepath, name_list, shape_list, dtype_list):
    with tf.name_scope('read_data'):
        filename_queue = tf.train.string_input_producer([filepath], shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        dict1={}
        for i in range(len(name_list)):
            dict1[name_list[i]] = tf.FixedLenFeature([], tf.string)
        features = tf.parse_single_example(serialized_example, features = dict1)
        outputs = []
        for i in range(3):
            temp = tf.decode_raw(features[name_list[i]], dtype_list[i])
            temp = tf.reshape(temp, shape_list[i])
            outputs.append(temp)
        temp = features[name_list[3]]
        outputs.append(temp)
        temp = features[name_list[4]]
        outputs.append(temp)
        return outputs 
def batch_data(data, batch_size):
    with tf.name_scope('batch_and_shuffle_data'):
        output = tf.train.shuffle_batch(data, batch_size = batch_size, 
                                        num_threads = 8,
                                        capacity=1000 + 3 * batch_size,
                                        min_after_dequeue = 1000,
                                        name='in_and_out')
        return output