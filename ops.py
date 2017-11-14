from contextlib import contextmanager
import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.contrib.layers import xavier_initializer
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
    