import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import batch_norm

tf.reset_default_graph()
batch_size = 32
height = 256
width = 256
lr_img_size = [32, 32]
patch_size = [32, 32]
is_training=True
input_img = tf.placeholder(dtype = tf.float32, shape = [batch_size, height, width, 3])
input_patch = tf.placeholder(dtype = tf.float32, shape = [batch_size] + patch_size + [3])
lr_img = tf.image.resize_images(input_img, size = lr_img_size, method=tf.image.ResizeMethod.BICUBIC)
print('Low resolution full image shape: ' + str(lr_img.get_shape().as_list()))
num_lstm_layer = 2
lstm_layer_size = 1024
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
    
def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)
def conv_net_lr_img(lr_img):
    print('*'*20 + '  Building convolutional network for low resolution image  ' + '*'*20)
    with tf.variable_scope('conv_net_lr_img'):
        out1 = downconv(lr_img, 16, [5, 5], name = 'down_conv1')
        out1 = batch_norm(out1, scale=True, is_training=is_training, scope='bnorm1')
        out1 = leakyrelu(out1)
        print('First Layer: Input ==> ' + str(lr_img.get_shape().as_list()) + ', Output ==> ' \
              + str(out1.get_shape().as_list()))
        
        out2 = downconv(out1, 32, [5, 5], name = 'down_conv2')
        out2 = batch_norm(out2, scale=True, is_training=is_training, scope='bnorm2')
        out2 = leakyrelu(out2)
        print('Second Layer: Input ==> ' + str(out1.get_shape().as_list()) + ', Output ==> ' \
              + str(out2.get_shape().as_list()))
        
        out3 = downconv(out2, 64, [3, 3], name = 'down_conv3')
        out3 = batch_norm(out3, scale=True, is_training=is_training, scope='bnorm3')
        out3 = leakyrelu(out3)
        print('Third Layer: Input ==> ' + str(out2.get_shape().as_list()) + ', Output ==> ' \
              + str(out3.get_shape().as_list()))
        
        return out3
    
def conv_net_patch(input_patch, i):
    if i==1:
        print('*'*20 + '  Building convolutional network for patches  ' + '*'*20)
    with tf.variable_scope('conv_net_patch'):
        out1 = downconv(input_patch, 16, [5, 5], name = 'down_conv1')
        out1 = batch_norm(out1, scale=True, is_training=is_training, scope='bnorm1')
        out1 = leakyrelu(out1)
        if i==1:
            print('First Layer: Input ==> ' + str(input_patch.get_shape().as_list()) + ', Output ==> '\
                  + str(out1.get_shape().as_list()))
        
        out2 = downconv(out1, 32, [5, 5], name = 'down_conv2')
        out2 = batch_norm(out2, scale=True, is_training=is_training, scope='bnorm2')
        out2 = leakyrelu(out2)
        if i==1:
            print('Second Layer: Input ==> ' + str(out1.get_shape().as_list()) + ', Output ==> '\
                  + str(out2.get_shape().as_list()))
              
        out3 = downconv(out2, 64, [3, 3], name = 'down_conv3')
        out3 = batch_norm(out3, scale=True, is_training=is_training, scope='bnorm3')
        out3 = leakyrelu(out3)
        if i==1:
            print('Third Layer: Input ==> ' + str(out2.get_shape().as_list()) + ', Output ==> '\
                  + str(out3.get_shape().as_list()))
            
        return out3
    
def deconv_net_patch(input_code, i):
    if i==0:
        print('*'*20 + '  Building Deconvolutional network for patches  ' + '*'*20)
    with tf.variable_scope('deconv_net_patch'):
        input_code = tf.reshape(input_code, [batch_size, 4, 4, 64])
        out1 = deconv(input_code, 32, [batch_size, 8, 8, 32], [3, 3], name = 'deconv1')
        out1 = batch_norm(out1, scale=True, is_training=is_training, scope='bnorm1')
        out1 = leakyrelu(out1)
        if i==0:
            print('First Layer: Input ==> ' + str(input_code.get_shape().as_list()) + ', Output ==> '\
                  + str(out1.get_shape().as_list()))
            
        out2 = deconv(out1, 16, [batch_size, 16, 16, 16], [5, 5], name = 'deconv2')
        out2 = batch_norm(out2, scale=True, is_training=is_training, scope='bnorm2')
        out2 = leakyrelu(out2)
        if i==0:
            print('Second Layer: Input ==> ' + str(out1.get_shape().as_list()) + ', Output ==> '\
                  + str(out2.get_shape().as_list()))
            
        out3 = deconv(out2, 3, [batch_size, 32, 32, 3], [5, 5], name = 'down_conv3')
        out3 = batch_norm(out3, scale=True, is_training=is_training, scope='bnorm3')
        out3 = leakyrelu(out3)
        if i==0:
            print('Third Layer: Input ==> ' + str(out2.get_shape().as_list()) + ', Output ==> '\
                  + str(out3.get_shape().as_list()))
            
        return out3
		
with tf.name_scope('extract_patches'):
    patches = tf.extract_image_patches(input_img, ksizes = [1] + patch_size + [3], 
                                     strides = [1, 16, 16, 1], 
                                     rates = [1, 1, 1, 1 ], padding = 'SAME')
    patches = tf.reshape(patches, [-1] + [batch_size] + patch_size + [3])
    print('Extracted patches shape ==> ' + str(patches.get_shape().as_list()))
lr_img_code =  conv_net_lr_img(lr_img)
lr_img_code = tf.reshape(lr_img_code, [batch_size, -1])
LSTMs = []
for i in range(num_lstm_layer):
    LSTMs.append(tf.contrib.rnn.LSTMCell(lstm_layer_size, use_peepholes = True))
stacked_LSTM = tf.contrib.rnn.MultiRNNCell(LSTMs)
states = stacked_LSTM.zero_state(batch_size, tf.float32)
with tf.variable_scope('lstm_layer'):
    for i in range(10):
        if i == 0:
            inputs = lr_img_code
        else:
            inputs = tf.reshape(conv_net_patch(output_patch, i), [batch_size, -1])
        if i == 1:
            tf.get_variable_scope().reuse_variables()
        output, states = stacked_LSTM(inputs, states)
        output_patch = deconv_net_patch(output, i)
        difference = output_patch - patches
        if i == 0:
            loss = tf.reduce_min(tf.reduce_mean(difference, [1, 2, 3, 4]))
        else:
            loss += tf.reduce_min(tf.reduce_mean(difference, [1, 2, 3, 4]))
        if i == 1:
            print('LSTM Input ==> ' + str(inputs.get_shape().as_list()))
            print('LSTM Output ==> ' + str(output.get_shape().as_list()))