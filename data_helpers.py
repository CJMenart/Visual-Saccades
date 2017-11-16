from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import h5py
import sys
import json
from itertools import izip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
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
                                        num_threads = 2,
                                        capacity=1000 + 3 * batch_size,
                                        min_after_dequeue = 1000,
                                        name='in_and_out')
        return output
    
def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

def get_vocab_size(filename):
    with open(filename, 'r') as f:
        vocab_list = f.read().decode('utf8').splitlines()
    return len(vocab_list)
        
def write_tfrecord_from_h5(out_filepath, input_img_h5, input_ques_h5, img_norm = True):
    train_data = {}
    print('loading image feature...')
    with h5py.File(input_img_h5,'r') as hf:
        # -----0~82459------
        tem = hf.get('images_train')
        img_feature = np.array(tem)
    # load h5 file
    print('loading h5 file...')
    with h5py.File(input_ques_h5,'r') as hf:
        # total number of training data is 215375
        # question is (26, )
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)-1
        # max length is 23
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        # total 82460 img
        tem = hf.get('img_pos_train')
        # convert into 0~82459
        train_data['img_list'] = np.array(tem)-1
        # answer is 1~1000
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('question aligning')
    train_data['question'] = right_align(train_data['question'], train_data['length_q'])
    ques_array = train_data['question'].astype(np.int32)
    img_id_array = train_data['img_list']
    ans_array = train_data['answers'].astype(np.int32)
    print(ans_array[0].shape)
    print(ques_array[0, :].shape)
    print(img_feature[0, :].shape)
    print('Normalizing image feature')
    img_feat_size = img_feature.shape[1]
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
        img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(img_feat_size, 1))))
        img_feature = img_feature.astype(np.float32)
        
    
    out_filepath = 'data.tfrecords'
    if os.path.exists(out_filepath):
        os.unlink(out_filepath)
    out_file = tf.python_io.TFRecordWriter(out_filepath) 
    n_train = ques_array.shape[0]
    
    for i in range(n_train):
        print('Writing data sample {}/{} in tfrecord.'.format(i+1, n_train), end='\r')
        sys.stdout.flush()
        write_tfrecords(out_file, [img_feature[img_id_array[i], :], 
                                   ques_array[i, :],
                                   ans_array[i]], ['img', 'ques', 'ans'])
        
    out_file.close()
                       
def get_val_data(data_path):
    data = np.load(data_path)
    img_array = data['img_array_val']
    ques_array = data['ques_array_val']
    ans_array = data['answers_val_all']
            
    return [img_array, ques_array, ans_array]


