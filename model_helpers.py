import tensorflow as tf
import numpy as np
import os 
import sys
from ops import *
def lstm_layer(inputs, states):
    lstm = tf.nn.rnn_cell(num_units = lstm_layer_size, use_peepholes = use_peepholes)
    lstm_out = lstm(inputs, states)

def affine_layer(inputs, out_dim, name = 'affine_layer'):
    in_dim=inputs.get_shape().as_list()[1]
    with tf.variable_scope(name):
        init = tf.random_uniform_initializer(-0.08, 0.08)
        weights = tf.get_variable(name = 'weights',shape = [in_dim,out_dim]
                                , dtype = tf.float32, initializer = init)
        outputs = tf.matmul(inputs, weights)
    return outputs

