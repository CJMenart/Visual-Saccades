from __future__ import print_function
from model import Model
import timeit
import numpy as np
import tensorflow as tf
from ops import *
from itertools import cycle
from helpers import *
from tensorflow.contrib.layers import fully_connected
from networks import QuestionEmbeddingNet, ImagePlusQuesFeatureNet, FeedForwardNet,  PatchGenerator
import sys
from sklearn.externals import joblib
temp = '*'*10
import os
import time
class QUES_NET(Model):
    def __init__(self, sess, devices, args, infer = False):
        super(QUES_NET, self).__init__('QUES_NET')
        self.sess = sess
        self.devices = devices
        self.epoch = args.epoch
        self.lr = tf.Variable(args.lr, name = 'lr', trainable = False)
        self.batch_size = args.batch_size
        self.num_hidden_layer = args.num_hidden_layer
        self.hidden_layer_size = args.hidden_layer_size
        self.word_embed_size = args.word_embed_size
        self.out_layer_size = args.out_layer_size
        self.is_train = args.is_train
        self.save_path = os.path.join(args.save_dir, self.name)
        self.results_path = args.result_path
        self.is_bnorm = args.is_bnorm
        self.train_data_path = args.train_path
        self.val_data_path = args.val_path
        self.labelencoder = joblib.load(args.lbl_enc_file)
        self.keep_prob = tf.Variable(args.keep_prob, dtype = tf.float32,
                                                trainable = False, 
                                                name = 'keep_prob')
        self.is_vars_summ = True
        self.is_grads_summ = True
        self.ffn = FeedForwardNet(activation_fn = leakyrelu)
        self.build_model(devices, args)
        self.train_writer, self.val_writer = self.summary_writer(sess.graph)
        self.merge_summaries()
        
        
       
    def build_model(self, devices, args):
        
        all_grads = []
        opt = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope(self.name):
            for idx, device in enumerate(devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_first_device(devices[0]):
                            self.build_model_single_gpu(idx, args)
                            grads = opt.compute_gradients(self.loss[-1], var_list=self.vars)
                            all_grads.append(grads)
                            tf.get_variable_scope().reuse_variables()
            avg_grads = average_gradients(all_grads)
            self.avg_grads = avg_grads
        self.opt = opt.apply_gradients(avg_grads)
        
       
     
        
    def build_model_single_gpu(self, idx, args):
        if idx == 0:
            self.ques = []
            self.ans = []
            self.out_logit = []
            self.loss = []
            
            with tf.name_scope('train_pipe'):
                self.train_data = read_data(self.train_data_path, ['ques', 'ans'], 
                                            [(self.word_embed_size, ), ()], 
                                            [tf.float32, tf.int32])
            
            with tf.name_scope('val_pipe'):
                self.val_data = read_val_data(self.val_data_path, 
                                        ['ques', 'ans_all', 'ques_str'], 
                                        [(self.word_embed_size, ), (), ()], 
                                        [tf.float32, tf.string, tf.string])
            with tf.name_scope('batch_val_data'):
                self.val_batch = tf.train.batch(self.val_data, 
                                                batch_size = args.batch_size, 
                                                num_threads = 6,
                                                capacity = 1000 + 3 * self.batch_size,
                                                enqueue_many = False, 
                                                allow_smaller_final_batch=False)
            
                self.val_ques = self.val_batch[0]
 
                self.val_answers = self.val_batch[1]
                self.val_ques_str = self.val_batch[2]
        
        with tf.name_scope('batch_and_shuffle_data'):
            data = tf.train.shuffle_batch([self.train_data[0], self.train_data[1]], 
                                           batch_size = self.batch_size, 
                                           num_threads = 4,
                                           capacity = 1000 + 3* self.batch_size,
                                           min_after_dequeue = 1000,
                                           enqueue_many = False, 
                                           name='in_and_out')
              
        
        self.ques.append(data[0])
        self.ans.append(data[1])
        print('ques shape ==> {}'.format(self.ques[0]))
        print('ans shape ==> {}'.format(self.ans[0]))
        
        if idx == 0:
            with tf.name_scope('Test_Model'):
              
                
   
                self.out_logit_test = self.ffn(self.val_ques, num_hidden_layer = self.num_hidden_layer, 
                                               hidden_layer_size = self.hidden_layer_size, 
                                               out_layer_size = self.out_layer_size, is_train = False)
                self.out_proba_test = tf.nn.softmax(self.out_logit_test)
                
                self.val_accuracy = tf.placeholder(dtype = tf.float32, shape = ())
            tf.get_variable_scope().reuse_variables()
            print(tf.get_variable_scope())
        with tf.name_scope('Train_Model'):
            print(tf.get_variable_scope())
            
            out_logit = self.ffn(self.ques[idx], num_hidden_layer = self.num_hidden_layer, 
                               hidden_layer_size = self.hidden_layer_size, 
                               out_layer_size = self.out_layer_size, is_train = True,  
                               keep_prob = self.keep_prob)
            self.out_logit.append(out_logit)
            if idx == 0:
                self.out_proba_train = tf.nn.softmax(self.out_logit[0])
            
           
        with tf.name_scope('cross_entropy_loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.out_logit[idx], 
                                                              labels = self.ans[idx]))
            self.loss.append(loss)
       
            
        self.get_vars()
       
        print('ques', self.ques[idx].get_shape().as_list(), 
              self.ques[idx].dtype)
        print('answer', self.ans[idx].get_shape().as_list(), 
              self.ans[idx].dtype)
        print('out_logit', self.out_logit[idx].get_shape().as_list(), self.out_logit[idx].dtype)
        print('loss', self.loss[idx].get_shape().as_list(), self.loss[idx].dtype)
        #sys.exit()   
    def get_vars(self):
        vars1 = tf.trainable_variables()
        self.vars_dict = {}
        for var in vars1:
            self.vars_dict[var.name] = var
        self.vars = self.vars_dict.values()
        print('{} printing variable names {}'.format('*'*20,'*'*20))
        for name in self.vars_dict.keys():
            print(name)
        print()
  
    def merge_summaries(self):
        with tf.name_scope('summaries'):
            self.loss_summ = []
            for idx, loss in enumerate(self.loss):
                self.loss_summ.append(scalar_summary('device' + str(idx) + '/loss_summ', loss))

            if self.is_vars_summ:
                self.vars_summ = []
                for var in self.vars:
                    self.vars_summ.append(histogram_summary(var.name.replace(':','_'), var))
    
            if self.is_grads_summ:
                self.grads_summ = []
                for grad, var in self.avg_grads:
                    self.grads_summ.append(histogram_summary(var.name.replace(':','_') + '/gradients', grad))
               
            self.val_accuracy_summ = scalar_summary('val_accuracy',  self.val_accuracy)
       
            summ_lst = self.vars_summ + self.grads_summ + self.loss_summ 
            if self.is_bnorm:       
                self.pop_mean_summ = []
                self.pop_var_summ = []
                self.pop_mean = self.ffn.batch_norm.pop_mean
                self.pop_var = self.ffn.batch_norm.pop_var
            
                for i in range(len(self.pop_mean)):
                    self.pop_mean_summ.append(histogram_summary(self.pop_mean[i].name, self.pop_mean[i]))
                    self.pop_var_summ.append(histogram_summary(self.pop_var[i].name, self.pop_var[i]))
                summ_lst += self.pop_mean_summ + self.pop_var_summ
            self.summ = tf.summary.merge(summ_lst)
            
    def summary_writer(self, graph):

        save_path = self.save_path
        if not os.path.exists(os.path.join(save_path, 'train')):
                os.makedirs(os.path.join(save_path, 'train'))
        if not os.path.exists(os.path.join(save_path, 'val')):
                os.makedirs(os.path.join(save_path, 'val'))
        train_writer = tf.summary.FileWriter(os.path.join(save_path,
                                                         'train'), graph)
        val_writer = tf.summary.FileWriter(os.path.join(save_path,
                                                         'val'))
        return train_writer, val_writer

    def train(self, model = None):
        devices = self.devices
   
        num_devices = len(devices)
        print('num_devices', num_devices)
        sess = self.sess
        init1 = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        sess.run([init1, init2])
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        time.sleep(3)
        epoch = self.epoch

        '''
        num_train_examples = 0
        for record in tf.python_io.tf_record_iterator(self.train_data_path):
            num_train_examples += 1
        '''
        num_train_examples = 364085
        num_train_batches = num_train_examples / self.batch_size
        print('Number of Train examples: ', num_train_examples)
        print('Batches per train epoch: ', num_train_batches)
        '''
        num_val_examples = 0
        for record in tf.python_io.tf_record_iterator(self.val_data_path):
            num_val_examples += 1
        '''
        num_val_examples = 214354

        num_val_batches = num_val_examples / self.batch_size
        print('Number of val examples: ', num_val_examples)
        print('Batches per val epoch: ', num_val_batches)
        self.num_val_batches = num_val_batches
        if self.load(self.save_path, model):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load Failed')
        counter = 0
        save_counter = 0
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        losses = []
        loss =0.
        try:
            while not coord.should_stop():
                
  
                if batch_idx >= num_train_batches:
                        
                    mean_losses = np.mean(losses, 0)
                    fdict = {}
                    for idx in range(num_devices):
                        fdict[self.loss[idx]] = mean_losses[idx]
                     
                    _summ = sess.run(self.summ, feed_dict = fdict) 
                    losses = []
                    curr_epoch += 1
                    batch_idx = 0
                    self.save(self.save_path, self.global_step)
                    sess.run(self.increment_op)
                    self.train_writer.add_summary(_summ, sess.run(self.global_step))
                    self.train_writer.flush()
                    if (curr_epoch % 1) == 0:
                        val_accuracy_test = get_validation_score(self, curr_epoch)
                        #val_accuracy_train = self.get_validation_score(curr_epoch, True)
                        
                        val_summ_test = sess.run(self.val_accuracy_summ, feed_dict = {self.val_accuracy:val_accuracy_test})
                        #val_summ_train = sess.run(self.val_accuracy_summ, feed_dict = {self.val_accuracy:val_accuracy_train})
                        self.val_writer.add_summary(val_summ_test, sess.run(self.global_step))
                        self.val_writer.flush()
                        #self.train_writer.add_summary(val_summ_train, sess.run(self.global_step))
                        self.train_writer.flush()
                    
                    
                else:
                    start = timeit.default_timer()
                    #time.sleep(0.5)
                    _, loss = sess.run([self.opt, self.loss])
                    losses.append(loss)    
                    batch_idx += num_devices
                    counter += num_devices
                
                    end = timeit.default_timer()
                    batch_timings.append(end - start)
                    print('{}/{} (epoch {}), loss = {:.5f}, '
                          'time/batch = {:.3f}, '
                           'mtime/batch = {:.3f}'.format(counter,
                                                    epoch * num_train_batches,
                                                    curr_epoch,
                                                    np.mean(loss),
                                                    end - start,
                                                    np.mean(batch_timings)))
                
                if curr_epoch >= self.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(epoch))
                    print('Saving last model at epcoh {}'.format(curr_epoch))
                    self.save(self.save_path, self.global_step)
                    break

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
        
