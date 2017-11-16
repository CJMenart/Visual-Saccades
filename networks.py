import tensorflow as tf
import numpy as np
from ops import *
import tensorflow as tf
import numpy as np
from ops import *
from model_helpers import *
from tensorflow.contrib.layers import fully_connected
import sys

class QuestionEmbeddingNet:
    def __init__(self, lstm_layer_size, num_lstm_layer, use_peepholes = True, is_bnorm = True, name = 'ques_embed'):
        self.name = name
        self.lstm_layer_size = lstm_layer_size
        self.num_lstm_layer = num_lstm_layer
        self.use_peepholes = use_peepholes
        self.LSTMs = []
        self.is_bnorm = is_bnorm
        if is_bnorm:
            self.batch_norm = BatchNorm()
        for i in range(self.num_lstm_layer):
            self.LSTMs.append(tf.nn.rnn_cell.LSTMCell(self.lstm_layer_size, forget_bias = 1.0, 
                                                      use_peepholes = self.use_peepholes))
     
    def __call__(self, ques_inp, ques_len_inp, vocab_size, word_embed_size, max_ques_length, batch_size, 
                 is_train = True, keep_prob = 0.5,
                 scope = 'ques_embed'):
        if is_train:
            self.LSTMs_drop = []
            for i in range(self.num_lstm_layer):
                self.LSTMs_drop.append(tf.nn.rnn_cell.DropoutWrapper(self.LSTMs[i], 
                                                            output_keep_prob = keep_prob))
            self.stacked_LSTM = tf.nn.rnn_cell.MultiRNNCell(self.LSTMs_drop)
        else:
            self.stacked_LSTM = tf.nn.rnn_cell.MultiRNNCell(self.LSTMs)
        with tf.variable_scope(scope):
            init = tf.random_uniform_initializer(-0.08, 0.08)
                
            self.ques_embed_W = tf.get_variable(name = 'embedding_matrix', 
                                              shape = [vocab_size, word_embed_size],
                                              dtype = tf.float32, initializer = init)
            print('ques_embed_W', self.ques_embed_W.get_shape().as_list(),  self.ques_embed_W.dtype)
                
            states = self.stacked_LSTM.zero_state(batch_size, tf.float32)
            inputs = []
            for i in range(max_ques_length):
                inputs.append(tf.nn.embedding_lookup(self.ques_embed_W, ques_inp[:, i]))
            lstm_inputs = tf.stack(inputs, axis = 1)
            outputs, states = tf.nn.dynamic_rnn(cell=self.stacked_LSTM,
                                             inputs=lstm_inputs,
                                             sequence_length=ques_len_inp,
                                             initial_state=states,
                                             parallel_iterations=32,
                                             swap_memory=False)
            '''
            for i in range(max_ques_length):
                inputs = tf.nn.embedding_lookup(self.ques_embed_W, ques_inp[:,i])
                inputs = tf.tanh(inputs)
                if is_train:
                    inputs = tf.nn.dropout(inputs, keep_prob = keep_prob)
                    
                output, states = self.stacked_LSTM(inputs, states)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()
            '''
            print(outputs[:, i, :].get_shape().as_list())
            concat_list = []
            for i in range(self.num_lstm_layer):
                concat_list.append(states[i].c)
                concat_list.append(states[i].h)
            ques_embed = tf.concat(concat_list, axis = 1)
            print('ques_embed', ques_embed.get_shape().as_list(), ques_embed.dtype)
            return ques_embed

class PatchGenerator:
    def __init__(self, batch_size, use_peepholes = True, is_bnorm = True):
        self.lr_img_size = [32, 32]
        self.patch_size = self.lr_img_size
        self.lstm_layer_size = 1024
        self.num_lstm_layers = 2
        self.batch_size = batch_size
        self.is_bnorm = is_bnorm
        self.LSTMs = []
        self.num_lstm_unroll = 10
        self.use_peepholes = use_peepholes
        self.output_patch = []
        if is_bnorm:
            self.batch_norm = BatchNorm()
        for i in range(self.num_lstm_layers):
            self.LSTMs.append(tf.nn.rnn_cell.LSTMCell(self.lstm_layer_size, forget_bias = 1.0, 
                                                      use_peepholes = self.use_peepholes))
            
    def __call__(self, input_img, is_train = True):
        with tf.name_scope('extract_patches'):
            patches = tf.extract_image_patches(input_img, ksizes = [1] + self.patch_size + [1], 
                                             strides = [1, 16, 16, 1], 
                                             rates = [1, 1, 1, 1 ], padding = 'SAME')
            patches = tf.reshape(patches, [-1] + [self.batch_size] + self.patch_size + [3])
            print('Extracted patches shape ==> {}'.format(patches.get_shape().as_list()))
            
            lr_img = tf.image.resize_images(input_img, size = self.lr_img_size, method=tf.image.ResizeMethod.BICUBIC)
            lr_img_code =  self.conv_net(lr_img, 0, is_training = is_train, name = 'lr_img_conv_net')
            lr_img_code = tf.reshape(lr_img_code, [self.batch_size, -1])
         
            self.stacked_LSTM = tf.nn.rnn_cell.MultiRNNCell(self.LSTMs)
            states = self.stacked_LSTM.zero_state(self.batch_size, tf.float32)
            with tf.variable_scope('lstm_layer'):
                for i in range(self.num_lstm_unroll):
                    if i == 0:
                        inputs = lr_img_code
                    else:
                        inputs = tf.reshape(self.conv_net(output_patch, i, is_training = is_train, 
                                                          name = 'patc_conv_net'), [self.batch_size, -1])
                    if i == 1:
                        tf.get_variable_scope().reuse_variables()
                    output, states = self.stacked_LSTM(inputs, states)
                    output_patch = self.deconv_net(output, i, is_training = is_train)
                    difference = tf.abs(output_patch - patches)
                    if i == 0:
                        loss = tf.reduce_min(tf.reduce_mean(difference, [1, 2, 3, 4]))
                    else:
                        loss += tf.reduce_min(tf.reduce_mean(difference, [1, 2, 3, 4]))
                    if i == 1:
                        print('LSTM Input ==> ' + str(inputs.get_shape().as_list()))
                        print('LSTM Output ==> ' + str(output.get_shape().as_list()))
                        
                self.output_patch.append(output_patch)
                img_feature = tf.reshape(output_patch, [self.batch_size, -1])
                return img_feature, loss
            
    def deconv_net(self, input_code, i, is_training = True, name = 'deconvnet'):
        if i==0:
            print('*'*20 + '  Building Deconvolutional network for patches  ' + '*'*20)
        with tf.variable_scope(name):
            input_code = tf.reshape(input_code, [self.batch_size, 4, 4, 64])
            out1 = deconv(input_code, 32, [self.batch_size, 8, 8, 32], [3, 3], name = 'deconv1')
            if self.is_bnorm:
                out1 = self.batch_norm(out1, is_train=is_training, name = 'bnorm1')
            out1 = leakyrelu(out1)
            if i==0:
                print('First Layer: Input ==> ' + str(input_code.get_shape().as_list()) + ', Output ==> '\
                      + str(out1.get_shape().as_list()))

            out2 = deconv(out1, 16, [self.batch_size, 16, 16, 16], [5, 5], name = 'deconv2')
            if self.is_bnorm:
                out2 =  self.batch_norm(out2, is_train=is_training, name = 'bnorm2')
            out2 = leakyrelu(out2)
            if i==0:
                print('Second Layer: Input ==> ' + str(out1.get_shape().as_list()) + ', Output ==> '\
                      + str(out2.get_shape().as_list()))

            out3 = deconv(out2, 3, [self.batch_size, 32, 32, 3], [5, 5], name = 'down_conv3')
            if self.is_bnorm:
                out3 = self.batch_norm(out3, is_train=is_training, name = 'bnorm3')
            out3 = leakyrelu(out3)
            if i==0:
                print('Third Layer: Input ==> ' + str(out2.get_shape().as_list()) + ', Output ==> '\
                      + str(out3.get_shape().as_list()))

            return out3
        
    def conv_net(self, input_code, i, is_training = True, name = 'conv_net'):
        if i==1:
            print('*'*20 + '  Building convolutional network for patches  ' + '*'*20)
        with tf.variable_scope(name):
            out1 = downconv(input_code, 16, [5, 5], name = 'down_conv1')
            if self.is_bnorm:
                out1 = self.batch_norm(out1, is_train = is_training, name ='bnorm1')
            out1 = leakyrelu(out1)
            if i==1:
                print('First Layer: Input ==> ' + str(input_code.get_shape().as_list()) + ', Output ==> '\
                      + str(out1.get_shape().as_list()))

            out2 = downconv(out1, 32, [5, 5], name = 'down_conv2')
            if self.is_bnorm:
                out2 = self.batch_norm(out2, is_train = is_training, name ='bnorm2')
            out2 = leakyrelu(out2)
            if i==1:
                print('Second Layer: Input ==> ' + str(out1.get_shape().as_list()) + ', Output ==> '\
                      + str(out2.get_shape().as_list()))

            out3 = downconv(out2, 64, [3, 3], name = 'down_conv3')
            if self.is_bnorm:
                out3 = self.batch_norm(out3, is_train = is_training, name ='bnorm3')
            out3 = leakyrelu(out3)
            if i==1:
                print('Third Layer: Input ==> ' + str(out2.get_shape().as_list()) + ', Output ==> '\
                      + str(out3.get_shape().as_list()))

            return out3
        
class ImagePlusQuesFeatureNet:
    def __init__(self, final_feat_size, out_layer_size, 
                 activation_fn = tf.tanh, feat_join = 'mul', is_bnorm = True):
        self.final_feat_size = final_feat_size
        self.activation_fn = activation_fn
        self.out_layer_size = out_layer_size
        self.is_bnorm = is_bnorm
        self.batch_norm = BatchNorm()
        self.feat_join = feat_join
    def __call__(self, img_inp, ques_inp, is_train = True, keep_prob = 0.5):
       
        with tf.variable_scope('feature_combination'):
            if is_train:
                img_inp = tf.nn.dropout(img_inp, keep_prob = keep_prob)
                ques_inp = tf.nn.dropout(ques_inp, keep_prob = keep_prob)
            if self.is_bnorm:
                with tf.variable_scope('img_embed_W'):
                    with tf.variable_scope('hidden1'):
                        final_img_feat = affine_layer(img_inp, self.final_feat_size)
                        final_img_feat = self.batch_norm(final_img_feat, is_train)
                        final_img_feat = tf.tanh(final_img_feat)
                      
                with tf.variable_scope('ques_embed_W'):
                    final_ques_feat = affine_layer(ques_inp, self.final_feat_size)
                    final_ques_feat = self.batch_norm(final_ques_feat, is_train)
                    final_ques_feat = tf.tanh(final_ques_feat)
                
            else:
                final_img_feat = fully_connected(img_inp, self.final_feat_size, 
                                                 self.activation_fn, 
                                                 scope = 'img_embed_reduce_W1')

                final_ques_feat = fully_connected(ques_inp, 
                                                  self.final_feat_size, 
                                                  self.activation_fn,
                                                  scope = 'ques_embed_reduce_W')
            if self.feat_join == 'mul':
                final_feat = tf.multiply(final_img_feat, final_ques_feat )
            elif self.feat_join == 'add':
                final_feat = tf.add(final_img_feat, final_ques_feat )
            elif self.feat_join == 'concat':
                final_feat = tf.concat([final_img_feat, final_ques_feat], axis = 1)
                
            print('final_feat', final_feat.get_shape().as_list(), final_feat.dtype)
            
            if is_train:
                final_feat = tf.nn.dropout(final_feat, keep_prob = keep_prob)
            init = tf.random_uniform_initializer(-0.08, 0.08)
            if self.is_bnorm:
                with tf.variable_scope('multi_model_W'):
                    
                    with tf.variable_scope('hidden0'):
                        final_feat = affine_layer(final_feat, final_feat.get_shape().as_list()[1])
                        final_feat = self.batch_norm(final_feat, is_train)
                        final_feat = leakyrelu(final_feat)
                        final_feat = tf.nn.dropout(final_feat, keep_prob = keep_prob)
                    
                    with tf.variable_scope('hidden1'):
                        final_feat = affine_layer(final_feat, self.out_layer_size)
                        final_feat = self.batch_norm(final_feat, is_train)
            else:
                final_feat = fully_connected(final_feat, 1024, activation_fn = None,
                                                 weights_initializer = init, scope = 'multi_modal_W')
                final_feat = fully_connected(final_feat, self.out_layer_size, activation_fn = None,
                                                 weights_initializer = init, scope = 'multi_modal_W1')
            return final_img_feat, final_ques_feat, final_feat
        
class BatchNorm:
    def __init__(self, name = 'Bnorm'):
        self.name = name
        self.pop_mean = []
        self.pop_var = []
    
    def __call__(self, inputs, is_train,  decay = 0.999, name = 'Bnorm'):
        epsilon=0.001
        with tf.variable_scope(name):
            scale_init = tf.constant_initializer(1.)
            beta_init = tf.constant_initializer(0.)
            scale = tf.get_variable(name = 'scale', shape = inputs.get_shape()[-1],
                                    dtype = tf.float32, initializer = scale_init)
            beta = tf.get_variable(name = 'beta', shape = inputs.get_shape()[-1], 
                                  dtype = tf.float32, initializer = beta_init)

            pop_mean_init = tf.constant_initializer(0.)
            pop_mean = tf.get_variable(name = 'pop_mean', shape = inputs.get_shape()[1:],
                                          dtype = tf.float32, initializer = pop_mean_init, trainable = False)
            pop_var_init = tf.constant_initializer(1.)
            pop_var = tf.get_variable(name = 'pop_var', shape = inputs.get_shape()[1:],
                                       dtype = tf.float32, initializer = pop_mean_init, trainable = False)
            self.pop_mean.append(pop_mean)
            self.pop_var.append(pop_var)
            if is_train:
                batch_mean, batch_var = tf.nn.moments(inputs,[0])
                train_mean = tf.assign(pop_mean,
                                           pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.assign(pop_var,
                                          pop_var * decay + batch_var * (1 - decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 
                                                         epsilon)
            else:
                return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

        
        
class FeedForwardNet:
    def __init__(self, activation_fn = tf.tanh, normalization_fn = BatchNorm, is_bnorm = True):
        self.batch_norm = BatchNorm()
        self.activation_fn = activation_fn
        self.is_bnorm = is_bnorm
    def __call__(self, inputs, num_hidden_layer, hidden_layer_size, out_layer_size, is_train, keep_prob):
        with tf.variable_scope('FeedForwardNet'):
            output = inputs
            init = tf.random_uniform_initializer(-0.08, 0.08)
            for i in range(num_hidden_layer):
                with tf.variable_scope('hidden' + str(i)):
                    if self.is_bnorm:
                        output = affine_layer(output, hidden_layer_size)
                        output = self.batch_norm(output, is_train)
                        output = tf.tanh(output)
                    else:
                        
                        output = fully_connected(output, hidden_layer_size, 
                                                 activation_fn=tf.tanh, weights_initializer = init)
                    if is_train:
                        output = tf.nn.dropout(output, keep_prob = keep_prob)
                        
            
            output = fully_connected(output, out_layer_size, activation_fn = None,  weights_initializer = init)
            return output
        
