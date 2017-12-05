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
import os
temp = '*'*10
class QUES_IMG_NET(Model):
    def __init__(self, sess, devices, args, infer = False):
        super(QUES_IMG_NET, self).__init__('QUES_IMG_NET')
        self.sess = sess
        self.devices = devices
        self.epoch = args.epoch
        self.lr = tf.Variable(args.lr, name = 'lr', trainable = False)
        self.batch_size = args.batch_size
        self.num_lstm_layer = args.num_lstm_layer
        self.lstm_layer_size = args.lstm_layer_size
        self.num_hidden_layer = args.num_hidden_layer
        self.hidden_layer_size = args.hidden_layer_size
        self.final_feat_size = args.final_feat_size
        self.word_embed_size = args.word_embed_size
        self.out_layer_size = args.out_layer_size
        self.is_train = args.is_train
        self.save_path = os.path.join(args.save_dir, self.name)
        self.results_path = args.result_path
        self.is_bnorm = args.is_bnorm
        self.feat_join = args.feat_join
        self.train_data_path = args.train_path
        self.val_data_path = args.val_path
        self.train_stats_path = args.train_stats_path
        self.labelencoder = joblib.load(args.lbl_enc_file)
        self.ques_with_img = args.ques_with_img
        self.hidden_keep_prob = tf.Variable(args.hidden_keep_prob, dtype = tf.float32,
                                                trainable = False, 
                                                name = 'hidden_keep_prob')
        self.lstm_keep_prob = tf.Variable(args.lstm_keep_prob, dtype = tf.float32, 
                                              trainable = False, 
                                              name = 'lstm_keep_prob')
        self.ce_loss_lambda = args.ce_loss_lambda
        self.patch_loss_lambda = args.patch_loss_lambda
        self.diff_loss_lambda = args.diff_loss_lambda
        self.is_vars_summ = True
        self.is_grads_summ = True
        self.d_opt = tf.train.AdamOptimizer(self.lr)
        self.use_peepholes = args.use_peepholes
                                                  
        self.patch_generator = PatchGenerator(batch_size = self.batch_size, 
                                              is_bnorm = self.is_bnorm, 
                                              use_peepholes = self.use_peepholes, 
                                              final_feat_size = self.final_feat_size, 
                                              ques_with_img = self.ques_with_img)
        self.combine_feature =  ImagePlusQuesFeatureNet(self.final_feat_size, self.out_layer_size, 
                                                        tf.tanh, self.feat_join, self.is_bnorm)
        self.ffn = FeedForwardNet(activation_fn = tf.nn.relu)
        self.build_model(devices, args)
        self.train_writer, self.val_writer = self.summary_writer(sess.graph)
        self.merge_summaries()
        #self.val_img, self.val_ques, self.val_ans = get_val_data(args.val_path)
        
        
       
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
            self.img = []
            self.img1 = []
            self.ques = []
            self.ques_len = []
            self.ans = []
            self.ques_embed = []
            self.img_feat = []
            self.out_logit = []
            self.loss = []
            self.patch_loss = []
            self.diff_loss = []
            self.lr_img = []
            self.inp_patches_lst = []
            self.out_patches_lst = []
            self.patches = []
            self.ce_loss = []
            tr_stats = np.load(self.train_stats_path)
            self.img_mean = tf.Variable(tr_stats['img_mean'].astype(np.float32), trainable = False)
            self.img_std = tf.Variable(tr_stats['img_std'].astype(np.float32), trainable = False)
            
            with tf.name_scope('train_pipe'):
                self.train_data = read_data(self.train_data_path, ['img', 'ques', 'ans'], 
                                            [(256, 256, 3), (self.word_embed_size, ), ()], 
                                            [tf.float32, tf.float32, tf.int32])
            with tf.name_scope('val_pipe'):
                self.val_data = read_val_data(self.val_data_path, 
                                        ['img', 'ques', 'ans_all', 'ques_str'], 
                                        [(256, 256, 3), (self.word_embed_size, ), (), ()], 
                                        [tf.float32, tf.float32, tf.string, tf.string])
            with tf.name_scope('batch_val_data'):
                self.val_batch = tf.train.batch(self.val_data, batch_size = args.batch_size, 
                                        num_threads = 6,
                                        capacity=1000 + 3 * args.batch_size,
                                        allow_smaller_final_batch=False)
                val_img = self.val_batch[0]
                self.val_img1 = val_img
                self.val_img = tf.divide(tf.subtract(val_img, self.img_mean), self.img_std)
                self.val_ques = self.val_batch[1]
                self.val_answers = self.val_batch[2]
                self.val_ques_str = self.val_batch[3]
        with tf.name_scope('batch_and_shuffle_data'):
            data = tf.train.shuffle_batch(self.train_data, batch_size = self.batch_size, 
                                            num_threads = 4,
                                            capacity = 1000 + 3 * self.batch_size,
                                            min_after_dequeue = 1000,
                                            enqueue_many = False, 
                                            name='in_and_out')
        data = batch_data(self.train_data, args.batch_size)
        img = (data[0] - self.img_mean) / self.img_std
        self.img.append(img)
        self.img1.append(data[0])
        self.ques.append(data[1])
        self.ans.append(data[2])
        print('img shape ==> {}'.format(self.img[0]))
        print('ques shape ==> {}'.format(self.ques[0]))
        print('ans shape ==> {}'.format(self.ans[0]))
     
        if idx == 0:
            with tf.name_scope('Test_Model'):
                self.ques_embed_test = fully_connected(inputs = self.val_ques, num_outputs = self.final_feat_size, 
                                                       activation_fn = tf.tanh, scope = 'ques_embed')
                self.img_feat_test, self.patch_loss_test, \
                self.diff_loss_test, self.lr_img_test, \
                self.inp_patches_lst_test, \
                self.out_patches_lst_test, \
                self.val_patches = self.patch_generator(self.val_img, 
                                                        self.ques_embed_test, 
                                                        is_train = False)
                
                self.final_feat_test = self.combine_feature(self.img_feat_test,
                                                            self.val_ques, 
                                                            is_train = False, 
                                                            keep_prob = 1)
                self.out_logit_test = self.ffn(self.final_feat_test, num_hidden_layer = self.num_hidden_layer, 
                                               hidden_layer_size = self.final_feat_test.get_shape().as_list()[-1], 
                                               out_layer_size = self.out_layer_size, 
                                               is_train = False)
                self.out_proba_test = tf.nn.softmax(self.out_logit_test)
                self.val_accuracy = tf.placeholder(dtype = tf.float32, shape = ())
            tf.get_variable_scope().reuse_variables()
            print(tf.get_variable_scope())
        with tf.name_scope('Train_Model'):
            print(tf.get_variable_scope())
     
            ques_embed = fully_connected(inputs = self.ques[idx], num_outputs = self.final_feat_size, 
                                         activation_fn = tf.tanh, scope = 'ques_embed')
            img_feat, patch_loss, diff_loss, \
            lr_img, inp_patches_lst, \
            out_patches_lst, patches = self.patch_generator(self.img[idx], ques_embed, is_train = True)
            
            final_feat = self.combine_feature(img_feat,
                                              self.ques[idx], 
                                              is_train = True, 
                                              keep_prob = self.hidden_keep_prob)
            out_logit = self.ffn(final_feat, num_hidden_layer = self.num_hidden_layer, 
                                               hidden_layer_size = final_feat.get_shape().as_list()[-1], 
                                               out_layer_size = self.out_layer_size, 
                                               is_train = True, keep_prob = self.hidden_keep_prob)
            self.ques_embed.append(ques_embed)
            self.img_feat.append(img_feat)
            self.patch_loss.append(patch_loss)
            self.diff_loss.append(diff_loss)
            self.lr_img.append(lr_img)
            self.inp_patches_lst.append(inp_patches_lst)
            self.out_patches_lst.append(out_patches_lst)
            self.patches.append(patches)
            #self.final_img_feat.append(final_img_feat)
            #self.final_ques_feat.append(final_ques_feat)
            self.out_logit.append(out_logit)
            if idx == 0:
                self.out_proba_train = tf.nn.softmax(self.out_logit[0])
            
           
        with tf.name_scope('cross_entropy_loss'):
            ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.out_logit[idx], 
                                                              labels = self.ans[idx]))
            loss =  (self.ce_loss_lambda * ce_loss) + (self.patch_loss_lambda * patch_loss) + \
                (self.diff_loss_lambda * diff_loss)
            self.ce_loss.append(ce_loss)
            self.loss.append(loss)
       
            
        self.get_vars()
        print('img_feat', self.img[idx].get_shape().as_list(), 
              self.img[idx].dtype)
        print('ques', self.ques[idx].get_shape().as_list(), 
              self.ques[idx].dtype)
        print('answer', self.ans[idx].get_shape().as_list(), 
              self.ans[idx].dtype)
        print('ques_embed', self.ques_embed[idx].get_shape().as_list(), 
              self.ques_embed[idx].dtype)
        #print('ques_embed_W', self.ques_embed_W.get_shape().as_list(), 
              #self.ques_embed_W.dtype)
        #print('final_img_feat', self.final_img_feat[idx].get_shape().as_list(), 
              #self.final_img_feat[idx].dtype)
        #print('final_ques_feat', self.final_ques_feat[idx].get_shape().as_list(), 
             # self.final_ques_feat[idx].dtype)
        print('out_logit', self.out_logit[idx].get_shape().as_list(), self.out_logit[idx].dtype)
        print('loss', self.loss[idx].get_shape().as_list(), self.loss[idx].dtype)
        #sys.exit()   
    def get_vars(self):
        vars1 = tf.trainable_variables()
        self.ques_vars_dict = {}
        self.img_vars_dict = {}
        self.mm_vars_dict = {}
        for var in vars1:
            if 'ques_embed' in var.name:
                self.ques_vars_dict[var.name] = var
            elif 'img_embed' in var.name:
                self.img_vars_dict[var.name] = var
            elif 'multi_modal' in var.name:
                self.mm_vars_dict[var.name] = var
        self.ques_vars = self.ques_vars_dict.values()
        self.img_vars = self.img_vars_dict.values()
        self.mm_vars = self.mm_vars_dict.values()
        print('{} printing question embedding variables {}'.format('*'*20,'*'*20))
        for name in self.ques_vars_dict.keys():
            print(name)
        print()
        print('{} printing image embedding variables {}'.format('*'*20,'*'*20))
        for name in self.img_vars_dict.keys():
            print(name)
        print()
        print('{} printing molti modal variables {}'.format('*'*20,'*'*20))
        for name in self.mm_vars_dict.keys():
            print(name)
        print()
        self.vars = self.ques_vars + self.img_vars + self.mm_vars
    def merge_summaries(self):
        with tf.name_scope('summaries'):
            self.loss_summ = []
            for idx, loss in enumerate(self.loss):
                self.loss_summ.append(scalar_summary('device' + str(idx) + '/loss_summ', loss))
            self.ce_loss_summ = []
            for idx, loss in enumerate(self.ce_loss):
                self.ce_loss_summ.append(scalar_summary('device' + str(idx) + '/ce_loss_summ', loss)) 
            self.patch_loss_summ = []
            for idx, loss in enumerate(self.patch_loss):
                self.patch_loss_summ.append(scalar_summary('device' + str(idx) + '/patch_loss_summ', loss))
                
            self.diff_loss_summ = []
            for idx, loss in enumerate(self.diff_loss):
                self.diff_loss_summ.append(scalar_summary('device' + str(idx) + '/diff_loss_summ', loss))
            self.lr_img_summ = []
            for idx, img in enumerate(self.lr_img):
                self.lr_img_summ.append(tf.summary.image( 'device_{}/lr_img_summ'.format(idx), img))
            self.img_summ = []
            for idx, img in enumerate(self.img):
                self.img_summ.append(tf.summary.image( 'device_{}/img_summ'.format(idx), img))
            self.img1_summ = []
            for idx, img in enumerate(self.img1):
                self.img1_summ.append(tf.summary.image( 'device_{}/img1_summ'.format(idx), img))
            self.inp_patch_summ = []  
            for idx, patch_lst in enumerate(self.inp_patches_lst):
                for idx1, patch in enumerate(patch_lst):
                    self.inp_patch_summ.append(tf.summary.image( 'device_{}/inp_patch_summ_{}'.format(idx, idx1), patch))
            self.out_patch_summ = []  
            for idx, patch_lst in enumerate(self.out_patches_lst):
                for idx1, patch in enumerate(patch_lst):
                    self.out_patch_summ.append(tf.summary.image( 'device_{}/out_patch_summ_{}'.format(idx, idx1), patch))
            self.init_patch_summ = []
            for idx, img in enumerate(self.patches):
                self.init_patch_summ.append(tf.summary.image( 'device_{}/init_patch_summ'.format(idx), 
                                                             tf.squeeze(tf.slice(img, begin = [0, 36, 0, 0, 0], 
                                                                      size = [-1, 1, -1, -1, -1]), axis = 1)))
            
           
            if self.is_vars_summ:
                self.ques_vars_summ = []
                for var in self.ques_vars:
                    self.ques_vars_summ.append(histogram_summary(var.name.replace(':','_'), var))
                self.img_vars_summ = []
                for var in self.img_vars:
                    self.img_vars_summ.append(histogram_summary(var.name.replace(':','_'), var))
                self.mm_vars_summ = []
                for var in self.mm_vars:
                    self.mm_vars_summ.append(histogram_summary(var.name.replace(':','_'), var))
                    
            if self.is_grads_summ:
                self.grads_summ = []
                for grad, var in self.avg_grads:
                    self.grads_summ.append(histogram_summary(var.name.replace(':','_') + '/gradients', grad))
                
           
            self.val_accuracy_summ = scalar_summary('val_accuracy',  self.val_accuracy)
            self.val_inp_patch_summ = []
            for idx, patch in enumerate(self.inp_patches_lst_test):
                self.val_inp_patch_summ.append(tf.summary.image( 'val_inp_patch_summ'.format(idx), patch))
                
            self.val_out_patch_summ = []
            for idx, patch in enumerate(self.out_patches_lst_test):
                self.val_out_patch_summ.append(tf.summary.image( 'val_out_patch_summ'.format(idx), patch))
            self.val_diff_loss_summ = scalar_summary('val_diff_loss_summ', self.diff_loss_test)
            self.val_patch_loss_summ = scalar_summary('val_patch_loss_summ', self.patch_loss_test)
            #self.val_ce_loss_summ = scalar_summary('val_ce_loss_summ', self.ce_loss_test)
            self.val_lr_img_summ = tf.summary.image( 'val_lr_img_summ', self.lr_img_test)
            self.val_img_summ = tf.summary.image( 'val_img_summ', self.val_img)
            self.val_img1_summ = tf.summary.image( 'val_img1_summ', self.val_img1)
            self.val_ques_summ = tf.summary.text('val_ques_summ', self.val_ques_str)
            self.val_init_patch_summ = tf.summary.image('val_init_patch_summ', tf.squeeze(tf.slice(self.val_patches, 
                                                                                       begin = [0, 36, 0, 0, 0], 
                                                                                       size = [-1, 1, -1, -1, -1]), 
                                                                                       axis = 1))
            val_summ_lst = [self.val_accuracy_summ, self.val_patch_loss_summ, 
                            self.val_diff_loss_summ, self.val_lr_img_summ, 
                            self.val_img_summ, self.val_img1_summ, self.val_ques_summ] + \
                            self.val_inp_patch_summ + self.val_out_patch_summ
            self.val_summ = tf.summary.merge(val_summ_lst)
            summ_lst = self.ques_vars_summ + self.img_vars_summ + self.mm_vars_summ + \
            self.grads_summ + self.loss_summ + self.ce_loss_summ + \
            self.patch_loss_summ + self.diff_loss_summ + self.inp_patch_summ + \
            self.out_patch_summ + self.lr_img_summ + self.img_summ + self.img1_summ + [self.init_patch_summ]
            if self.combine_feature.is_bnorm:       
                self.pop_mean_summ = []
                self.pop_var_summ = []
                self.pop_mean = self.combine_feature.batch_norm.pop_mean
                self.pop_var = self.combine_feature.batch_norm.pop_var
            
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
    
        #opt = self.opt
        num_devices = len(devices)
        print('num_devices', num_devices)
        sess = self.sess
        init1 = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        sess.run([init1, init2])
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
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
        ce_losses = []
        patch_losses = []
        diff_losses = []
        loss =0.
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                #answers = sess.run(self.val_batch[3])
                #print(answers)
                #print('ashutosh')
                
                if batch_idx >= num_train_batches:
                        
                    mean_losses = np.mean(losses, 0)
                    fdict = {}
                    for idx in range(num_devices):
                        fdict[self.loss[idx]] = mean_losses[idx]
                     
                    _summ = sess.run(self.summ, feed_dict = fdict) 
                    losses = []
                    ce_losses = []
                    patch_losses = []
                    diff_losses = []
                    curr_epoch += 1
                    batch_idx = 0
                    self.save(self.save_path, self.global_step)
                    sess.run(self.increment_op)
                    self.train_writer.add_summary(_summ, sess.run(self.global_step))
                    self.train_writer.flush()
                    if (curr_epoch % 5) == 0:
                        val_accuracy_test = get_validation_score(self, curr_epoch)
                        #val_accuracy_train = self.get_validation_score(curr_epoch, True)
                        
                        val_summ_test = sess.run(self.val_summ, feed_dict = {self.val_accuracy:val_accuracy_test})
                        #val_summ_train = sess.run(self.val_accuracy_summ, feed_dict = {self.val_accuracy:val_accuracy_train})
                        self.val_writer.add_summary(val_summ_test, sess.run(self.global_step))
                        self.val_writer.flush()
                        #self.train_writer.add_summary(val_summ_train, sess.run(self.global_step))
                        self.train_writer.flush()
                    
                    
                else:
                    start = timeit.default_timer()
                    _, loss, ce_loss, patch_loss, diff_loss = sess.run([self.opt, self.loss, 
                                                                        self.ce_loss,
                                                                        self.patch_loss, 
                                                                        self.diff_loss])
                    losses.append(loss)
                    ce_losses.append(ce_loss)
                    patch_losses.append(patch_loss)
                    diff_losses.append(diff_loss)
                    batch_idx += num_devices
                    counter += num_devices
                
                    end = timeit.default_timer()
                    batch_timings.append(end - start)
                    print('{}/{} (epoch {}), loss = {:.5f}, '
                          'ce_loss = {:.5f}, '
                          'patch_loss = {:.5f}, '
                          'diff_loss = {:.5f}, '
                          'time/batch = {:.3f}, '
                          'mtime/batch = {:.3f}'.format(counter,
                                                        epoch * num_train_batches,
                                                        curr_epoch,
                                                        np.mean(loss),
                                                        np.mean(ce_loss), 
                                                        np.mean(patch_loss),
                                                        np.mean(diff_loss),
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
        
       
