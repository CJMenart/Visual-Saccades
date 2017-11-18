from __future__ import print_function
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
import tensorflow as tf
import os
import imageio
from skimage.transform import resize
import re
import timeit
from ops import *
from helpers import tokenize, selectFrequentAnswers, get_tokens, final_tokens, encode_questions
import sys

temp = '*'*10    
print(temp)
imdir='%s/COCO_%s_%012d.jpg'
questions_train = \
open('data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_train = \
open('data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = \
open('data/preprocessed/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = \
open('data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
images_train_path = \
open('data/preprocessed/images_train2014_path.txt', 'r').read().decode('utf8').splitlines()
answers_train_all = \
open('data/preprocessed/answers_train2014_all.txt', 'r').read().decode('utf8').splitlines()
#get_unique_images_train()

max_answers = 1000
questions_train, questions_lengths_train, answers_train, images_train, \
images_train_path, answers_train_all = selectFrequentAnswers(questions_train, 
                                                             questions_lengths_train, 
                                                             answers_train, 
                                                             images_train, 
                                                             images_train_path, 
                                                             answers_train_all, 
                                                             max_answers)
print ('ques_train, size = {}, sample = {}'.format(len(questions_train), questions_train[0]))
print ('ques_lengths, size = {}, sample = {}'.format(len(questions_lengths_train), questions_lengths_train[0]))
print ('ans_train, size = {}, sample = {}'.format(len(answers_train), answers_train[0]))
print ('imag_train, size = {}, sample = {}'.format(len(images_train), images_train[0]))
print ('imag_train_path, size = {}, sample = {}'.format(len(images_train_path), images_train_path[0]))
print ('ans_train_all, size = {}, sample = {}'.format(len(answers_train_all), answers_train_all[0]))
print(temp)

print(temp)
questions_val = \
open('data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_val = \
open('data/preprocessed/questions_lengths_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = \
open('data/preprocessed/answers_val2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_val = \
open('data/preprocessed/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val_path = \
open('data/preprocessed/images_val2014_path.txt', 'r').read().decode('utf8').splitlines()
answers_val_all = \
open('data/preprocessed/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()

print ('ques_val, size = {}, sampel = {}'.format(len(questions_val), questions_val[0]))
print ('ques_lengths_val, size = {}, sample = {}'.format(len(questions_lengths_val), questions_lengths_val[0]))
print ('ans_val, size = {}, sample = {}'.format(len(answers_val), answers_val[0]))
print ('imag_val, size = {}, sample = {}'.format(len(images_val), images_val[0]))
print ('imag_val_path, size = {}, sample = {}'.format(len(images_val_path), images_val_path[0]))
print ('ans_val_all, size = {}, sample = {}'.format(len(answers_val_all), answers_val_all[0]))
print(temp)

print(temp)
ques_tokens_train = get_tokens(questions_train)
ques_tokens_val = get_tokens(questions_val)
counts = {}
count_thr = 5
for i, tokens in enumerate(ques_tokens_train):#change to train
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
print('top words and their counts:')
print('\n'.join(map(str,cw[:20])))
# print some stats
total_words = sum(counts.itervalues())
print('total words:', total_words)
bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
vocab = [w for w,n in counts.iteritems() if n > count_thr]
bad_count = sum(counts[w] for w in bad_words)
print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
print('number of words in vocab would be %d' % (len(vocab), ))
print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
print('inserting the special UNK token')
vocab.append('UNK')

vocab_file = 'data/preprocessed/vocab_list.txt'.format(count_thr)
with open(vocab_file, 'w') as f:
    for word in vocab:
        f.write((word + '\n').encode('utf8'))
        
itow = {i+1:w for i,w in enumerate(vocab)} 
wtoi = {w:i+1 for i,w in enumerate(vocab)} 

ques_tokens_train_final = final_tokens(ques_tokens_train, counts, count_thr)
print('Sample train question tokens ==> {}'.format(ques_tokens_train[0]))
print('Total number of train questions ==> {}'.format(len(ques_tokens_train)))

ques_tokens_val_final = final_tokens(ques_tokens_val, counts, count_thr)
print('Sample validation question tokens ==> {}'.format(ques_tokens_val[0]))
print('Total number of validation questions ==> {}'.format(len(ques_tokens_val)))

ques_array_train = encode_questions(ques_tokens_train_final, wtoi, 25)
ques_array_val = encode_questions(ques_tokens_val_final, wtoi, 25)
print('Encoded train questions array shape ==> {}'.format(ques_array_train.shape))
print('Encoded validation questions array shape ==> {}'.format(ques_array_val.shape))
print(temp)

print('{} Creating labels for training answers {}'.format('*'*10, '*'*10))
print('Number of training answers ==> {}'.format(len(answers_train)))
print('A sample training answer ==> {}'.format(answers_train[5]))

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))

ans_array_train = labelencoder.transform(answers_train)
joblib.dump(labelencoder,'data/labelencoder.pkl')
print('{} Done creating labels for training answers {}'.format('*'*10, '*'*10))
print('')

img_shape = [256, 256, 3]
print('{} Writing tfrecord file for validation data {}'.format(temp, temp))
N = len(images_val_path)
out_filepath = 'data/val_data.tfrecords'
if os.path.exists(out_filepath):
    os.unlink(out_filepath)
out_file = tf.python_io.TFRecordWriter(out_filepath)
start = timeit.default_timer()
for i in range(N):
    img_path = os.path.join('/fs/project/PAS1315/VQA/Images/', images_val_path[i])
    img = imageio.imread(img_path) 
    if len(img.shape) == 2:
        print('Image {} is an RGB image. Converted to RGB. Image shape ==> {}'.format(i+1, img.shape))
        img = np.stack([img, img, img], axis = 2)
    img = resize(img, img_shape[:2], order = 3)
    img = img.astype(np.float32)
    ques = ques_array_val[i]
    ques = ques.astype(np.int32)
    ques_len = questions_lengths_val[i]
    ques_len = np.array(int(ques_len)).astype(np.int32)
    ans_all = answers_val_all[i].encode('utf8')
    ques_str = questions_val[i].encode('utf8')
    write_tfrecords_val(out_file, [img, ques, ques_len, ans_all, ques_str], 
                        ['img', 'ques', 'ques_len', 'ans_all', 'ques_str'])
    print('{}/{} written.'.format(i+1, N), end = '\r')
    sys.stdout.flush()
out_file.close()  
end = timeit.default_timer()
print('{} Done writing tfrecord file for validation data. Time = {:.2f} s. {}'.format(temp, end - start, temp))
print('')

print('{} Writing tfrecord file for training data {}'.format(temp, temp))
N = len(images_train_path)
out_filepath = 'data/train_data.tfrecords'
if os.path.exists(out_filepath):
    os.unlink(out_filepath)
out_file = tf.python_io.TFRecordWriter(out_filepath)
start = timeit.default_timer()
sum_stats = np.zeros(img_shape)
sum_2_stats = np.zeros(img_shape)
count = 0
for i in range(N):
    img_path = os.path.join('/fs/project/PAS1315/VQA/Images/', images_train_path[i])
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        print('Image {} is an RGB image. Converted to RGB. Image shape ==> {}'.format(i+1, img.shape))
        img = np.stack([img, img, img], axis = 2)
    img = resize(img, img_shape[:2], order = 3)
    img = img.astype(np.float32)
    sum_stats += img
    sum_2_stats += img*img
    count += 1
    ques = ques_array_train[i]
    ques = ques.astype(np.int32)
    ques_len = questions_lengths_train[i]
    ques_len = np.array(int(ques_len)).astype(np.int32)
    ans = ans_array_train[i]
    ans = np.array(ans).astype(np.int32)
    write_tfrecords(out_file, [img, ques, ques_len, ans], ['img', 'ques', 'ques_len', 'ans'])
    print('{}/{} written.'.format(i+1, N), end = '\r')
    sys.stdout.flush()
img_mean = sum_stats / float(count)
img_std = np.sqrt((sum_2_stats / float(count)) - (img_mean ** 2))
train_stats_path = 'data/train_stats.npz'
np.savez(train_stats_path, img_mean = img_mean, img_std = img_std)
out_file.close()    
end = timeit.default_timer()
print('{} Done writing tfrecord file for Training data. Time = {:.2f} s. {}'.format(temp, end - start, temp))
print('')
