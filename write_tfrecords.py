from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy.io
import sys
import operator
from sklearn.externals import joblib
from sklearn import preprocessing
from progressbar import Bar, ETA, Percentage, ProgressBar 
import tensorflow as tf
import os
import imageio
from skimage.transform import resize
import re
import timeit
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
    example = tf.train.Example(features = tf.train.Features(feature = dict1))
    out_file.write(example.SerializeToString())
'''
def write_tfrecords_val(out_file, var_list, name_list):
    dict1 = {}
    dict1[name_list[0]] = _bytes_feature(var_list[0].tostring())
    dict1[name_list[1]] = _bytes_feature(var_list[1].tostring())
    dict1[name_list[2]] = _bytes_feature(var_list[2].tostring())
    dict1[name_list[3]] = _bytes_feature(np.array(var_list[3]).tostring())
    example = tf.train.Example(features = tf.train.Features(feature = dict1))
    out_file.write(example.SerializeToString())
'''
def selectFrequentAnswers(questions_train, questions_lengths_train, answers_train, 
                          images_train, images_train_path,  
                          answers_train_all, max_answers):
    answer_fq= defaultdict(int)
    #build a dictionary of answers
    for answer in answers_train:
        answer_fq[answer] += 1

    sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
    top_answers, top_fq = zip(*sorted_fq)
    new_questions_train = []
    new_questions_lengths_train = []
    new_answers_train = []
    new_images_train = []
    new_images_train_path = []
    new_answers_train_all = []
    
    #only those answer which appear int he top 1K are used for training
    for question, question_length, answer, image, image_path, answer_all in zip(questions_train,
                                                                                     questions_lengths_train,
                                                                                     answers_train, 
                                                                                     images_train, 
                                                                                     images_train_path, 
                                                                                     answers_train_all):
        if answer in top_answers:
            new_questions_train.append(question)
            new_questions_lengths_train.append(question_length)
            new_answers_train.append(answer)
            new_images_train.append(image)
            new_images_train_path.append(image_path)
            new_answers_train_all.append(answer_all)

    return (new_questions_train, new_questions_lengths_train,
            new_answers_train, new_images_train, 
            new_images_train_path, new_answers_train_all)

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i, (N-lengths[i]):N]=seq[i, 0:lengths[i]]
    return v
def encode_questions(ques_tokens, wtoi, max_length):
    N = len(ques_tokens)
    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    for i, tokens in enumerate(ques_tokens):
        label_length[i] = min(max_length, len(tokens))
        for k, w in enumerate(tokens):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    #label_arrays = right_align(label_arrays, label_length)
    
    return label_arrays

def get_tokens(sent_list):
    ans = []
    for sent in sent_list:
        ans.append(tokenize(sent.lower()))
    return ans
def final_tokens(tokens_list, counts, count_thr):
    ans = []
    for tokens in tokens_list:
        final_tokens = [w if counts.get(w,0) > count_thr else 'UNK' for w in tokens]
        ans.append(final_tokens)
    return ans
def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def save_image_stats(images_train_path, img_shape = [256, 256, 3]):
    print('{} Computing mean and std of training images {}'.format(temp, temp))
    start = timeit.default_timer()
    N = len(images_train_path)
    sum_stats = np.zeros(img_shape)
    sum_2_stats = np.zeros(img_shape)
    count = 0
    for i in range(N):
        img_path = os.path.join('data', images_train_path[i])
        img = imageio.imread(img_path)
        if len(img.shape) != 3:
            print('Skipped for image number {}. Image shape ==> {}'.format(i+1, img.shape))
            continue
        img = resize(img, img_shape[:2], order=3)
        img = img.astype(np.float32)
        sum_stats += img
        sum_2_stats += img*img
        count += 1
        print('{}/{}'.format(i+1, N), end = '\r')
        sys.stdout.flush()
    img_mean = sum_stats / float(count)
    img_std = np.sqrt((sum_2_stats / float(count)) - (img_mean ** 2))
    train_stats_path = 'data/train_stats.npz'
    np.savez(train_stats_path, img_mean = img_mean, img_std = img_std)
    end = timeit.default_timer()
    print('{} Done computing mean and variance of training images. Time = {:.2f} s.{}'\
          .format(temp, end - start, temp))
    print('')
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

save_image_stats(images_train_path)
img_shape = [256, 256, 3]
print('{} Writing tfrecord file for validation data {}'.format(temp, temp))
N = len(images_val_path)
out_filepath = 'data/val_data.tfrecords'
if os.path.exists(out_filepath):
    os.unlink(out_filepath)
out_file = tf.python_io.TFRecordWriter(out_filepath)
start = timeit.default_timer()
for i in range(N):
    img_path = os.path.join('data', images_val_path[i])
    img = imageio.imread(img_path)
    if len(img.shape) < 3:
        print('Skipped for image number {}. Image shape ==> {}'.format(i+1, img.shape))
        continue
    img = resize(img, img_shape[:2], order = 3)
    img = img.astype(np.float32)
    ques = ques_array_val[i]
    ques = ques.astype(np.int32)
    ques_len = questions_lengths_val[i]
    ques_len = np.array(int(ques_len)).astype(np.int32)
    ans_all = str(answers_val_all[i])
    write_tfrecords_val(out_file, [img, ques, ques_len, ans_all], ['img', 'ques', 'ques_len', 'ans_all'])
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
for i in range(N):
    img_path = os.path.join('data', images_train_path[i])
    img = imageio.imread(img_path)
    if len(img.shape) < 3:
        print('Skipped for image number {}. Image shape ==> {}'.format(i+1, img.shape))
        continue
    img = resize(img, img_shape[:2], order = 3)
    img = img.astype(np.float32)
    ques = ques_array_train[i]
    ques = ques.astype(np.int32)
    ques_len = questions_lengths_train[i]
    ques_len = np.array(int(ques_len)).astype(np.int32)
    ans = ans_array_train[i]
    ans = np.array(ans).astype(np.int32)
    write_tfrecords(out_file, [img, ques, ques_len, ans], ['img', 'ques', 'ques_len', 'ans'])
    print('{}/{} written.'.format(i+1, N), end = '\r')
    sys.stdout.flush()
out_file.close()    
end = timeit.default_timer()
print('{} Done writing tfrecord file for Training data. Time = {:.2f} s. {}'.format(temp, end - start, temp))
print('')
