from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import h5py
import sys
import json
from itertools import izip_longest
from collections import defaultdict
import operator
import re
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

def getModalAnswer(answers):
    candidates = {}
    for i in xrange(10):
        ans = answers[i]['answer']
        candidates[ans] = 1

    for i in xrange(10):
        ans = answers[i]['answer']
        candidates[ans] = 1
    return max(candidates.iteritems(), key=operator.itemgetter(1))[0]

def getAllAnswer(answers):
    answer_list = []
    for i in xrange(10):
        ans = answers[i]['answer']
        ans = ans.replace('\"','')
        answer_list.append(ans)

    return '|'.join(answer_list)

def get_vocab_size(filename):
    with open(filename, 'r') as f:
        vocab_list = f.read().decode('utf8').splitlines()
    return len(vocab_list)