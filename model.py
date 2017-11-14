import numpy as np
import tensorflow as tf
import timeit
import os
from ops import *
class Model(object):
    def __init__(self, name='BaseModel'):
        self.name = name
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False, dtype=tf.int32)
        self.increment_op = tf.assign(self.global_step, self.global_step + 1)

    def save(self, save_path, step):
        model_name= self.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1, max_to_keep = 10)
        self.saver.save(self.sess,
                        os.path.join(save_path, model_name),
                        global_step=step)

    def load(self, save_path, model_file=None):
        if not os.path.exists(save_path):
            print('[!] Checkpoints path does not exist...')
            return False
        print('[*] Reading checkpoints...')
        if model_file is None:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                return False
        else:
            ckpt_name = model_file
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        print(save_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(save_path, ckpt_name))
        print('[*] Read {}'.format(ckpt_name))
        return True
