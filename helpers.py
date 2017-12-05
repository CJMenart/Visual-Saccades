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
import timeit
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

def loadGloveModel(gloveFile = 'data/glove.6B.300d.txt'):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def getQuesVecFromModel(ques_tokens_lst, model, ques_embed_dim = 300):
    ques_vec_lst = []
    for token_lst in ques_tokens_lst:
        ques_vec = np.zeros((ques_embed_dim, ))
        cnt = 0.
        for token in token_lst:
            if model.has_key(token):
                ques_vec += model[token]
                cnt += 1
            else:
                continue
        ques_vec = ques_vec / cnt
        ques_vec_lst.append(ques_vec)
    return ques_vec_lst

def get_validation_score(self, curr_epoch):
    temp = '*'*20
    print('{} Evaluating model on validation data {}'.format(temp, temp))
    if not os.path.exists(self.results_path):
        os.makedirs(self.results_path)

    start = timeit.default_timer()
    correct_val = 0.0
    total = 0.001

    other_count = 0
    binary_correct_val = 0.0
    binary_total = 0.001
    num_correct_val = 0.0
    num_total = 0.001
    other_correct_val = 0.0
    other_total = 0.001
    f1 = open('{}/predicted_answers_{}.txt'.format(self.results_path, self.name), 'w')
    for i in range(self.num_val_batches):

        y_proba,  val_answers, val_ques_str = self.sess.run([self.out_proba_test, 
                                                             self.val_answers, 
                                                             self.val_ques_str])
        y_predict = y_proba.argmax(axis = -1)
        y_predict_text = self.labelencoder.inverse_transform(y_predict)
        print('{}/{} (epoch = {})'.format(i+1, self.num_val_batches, curr_epoch))





        for prediction, truth, ques_str, in zip(y_predict_text, val_answers, val_ques_str):

            temp_count=0
            for _truth in truth.split('|'):
                if prediction == _truth:
                    temp_count+=1
            if temp_count>2:
                correct_val+=1
            else:
                correct_val+=float(temp_count)/3

            total += 1


            if prediction == 'yes' or prediction == 'no':
                binary_temp_count = 0
                for _truth in truth.split('|'):
                    if prediction == _truth:
                        binary_temp_count+=1
                if binary_temp_count>2:
                    binary_correct_val+=1
                else:
                    binary_correct_val+= float(binary_temp_count)/3
                binary_total+=1
            elif np.core.defchararray.isdigit(prediction):
                num_temp_count = 0
                for _truth in truth.split('|'):
                    if prediction == _truth:
                        num_temp_count+=1
                if num_temp_count>2:
                    num_correct_val+=1
                else:
                    num_correct_val+= float(num_temp_count)/3
                num_total+=1
            else:
                other_count = 0
                for _truth in truth.split('|'):
                    if prediction == _truth:
                        other_count += 1
                if other_count > 2:
                    other_correct_val += 1
                else:
                    other_correct_val += float(other_count) / 3
                other_total += 1

            f1.write('Question ==> {} \n'.format(ques_str).encode('utf-8'))
            f1.write('Prediction ==> {} \n'.format(prediction).encode('utf-8'))
            f1.write('Answer ==> {} \n'.format(truth))
            f1.write(('*'*100 + '\n').encode('utf8'))
    f1.close()
    f2 = open('{}/overall_results_{}.txt'.format(self.results_path, self.name), 'a')
    f2.write('Total Alccuracy: {:.4f} \n\n'.format(correct_val/float(total)))

    f2.write('Accuracy on Yes No questions: {:.4f} \n\n'.format(binary_correct_val / float(binary_total)))

    f2.write('Accuracy on Number type question: {:.4f} \n\n'.format(num_correct_val / float(num_total)))

    f2.write('Accuracy on Other type question: {:.4f} \n\n'.format(other_correct_val / float(other_total)))
    f2.write(('*'*100 + '\n').encode('utf8'))

    f2.close()
    accy = correct_val/float(total)
    print('')
    print('Final Accuracy is {:.4f}'.format(accy))
    print('')
    end = timeit.default_timer()
    print('{} Done evaluating model on validation data. Time = {:.2f} s. {}'.format(temp, end - start, temp))
    print('')
    return accy