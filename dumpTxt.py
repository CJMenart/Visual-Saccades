from __future__ import print_function
import operator
import argparse
import progressbar
import json
import spacy
import sys
import re
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', 
        help='Specify which part of the dataset you want to dump to text. Your options are: train, val, test, test-dev')
    parser.add_argument('--answers', type=str, default='modal',
        help='Specify if you want to dump just the most frequent answer for each questions (modal), or all the answers (all)')
    args = parser.parse_args()

    #nlp = English() #used for conting number of tokens

    if args.split == 'train':
        annFile = 'data/v2_mscoco_train2014_annotations.json'
        quesFile = 'data/v2_OpenEnded_mscoco_train2014_questions.json'
        questions_file = 'data/preprocessed/questions_train2014.txt'
        questions_id_file = 'data/preprocessed/questions_id_train2014.txt'
        questions_lengths_file = 'data/preprocessed/questions_lengths_train2014.txt'
        if args.answers == 'modal':
            answers_file = 'data/preprocessed/answers_train2014_modal.txt'
        elif args.answers == 'all':
            answers_file = 'data/preprocessed/answers_train2014_all.txt'
        coco_image_id = 'data/preprocessed/images_train2014.txt'
        coco_image_path =  'data/preprocessed/images_train2014_path.txt'
        data_split = 'training data'
        subtype = 'train2014'
    elif args.split == 'val':
        annFile = 'data/v2_mscoco_val2014_annotations.json'
        quesFile = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
        questions_file = 'data/preprocessed/questions_val2014.txt'
        questions_id_file = 'data/preprocessed/questions_id_val2014.txt'
        questions_lengths_file = 'data/preprocessed/questions_lengths_val2014.txt'
        if args.answers == 'modal':
            answers_file = 'data/preprocessed/answers_val2014_modal.txt'
        elif args.answers == 'all':
            answers_file = 'data/preprocessed/answers_val2014_all.txt'
        coco_image_id = 'data/preprocessed/images_val2014_all.txt'
        coco_image_path =  'data/preprocessed/images_val2014_path.txt'
        data_split = 'validation data'
        subtype = 'val2014'
    elif args.split == 'test-dev':
        quesFile = 'data/v2_OpenEnded_mscoco_test-dev2015_questions.json'
        questions_file = 'data/preprocessed/questions_test-dev2015.txt'
        questions_id_file = 'data/preprocessed/questions_id_test-dev2015.txt'
        questions_lengths_file = 'data/preprocessed/questions_lengths_test-dev2015.txt'
        coco_image_id = 'data/preprocessed/images_test-dev2015.txt'
        coco_image_path =  'data/preprocessed/images_test-dev2015_path.txt'
        data_split = 'test-dev data'
        subtype = 'test-dev2015'
    elif args.split == 'test':
        quesFile = 'data/v2_OpenEnded_mscoco_test2015_questions.json'
        questions_file = 'data/preprocessed/questions_test2015.txt'
        questions_id_file = 'data/preprocessed/questions_id_test2015.txt'
        questions_lengths_file = 'data/preprocessed/questions_lengths_test2015.txt'
        coco_image_id = 'data/preprocessed/images_test2015.txt'
        coco_image_path =  'data/preprocessed/images_test2015_path.txt'
        data_split = 'test data'
        subtype = 'test2015'
    else:
        raise RuntimeError('Incorrect split. Your choices are:\ntrain\nval\ntest-dev\ntest')

    #initialize VQA api for QA annotations
    #vqa=VQA(annFile, quesFile)
    questions = json.load(open(quesFile, 'r'))
    ques = questions['questions']
    if args.split == 'train' or args.split == 'val':
        qa = json.load(open(annFile, 'r'))
        qa = qa['annotations']

    #pbar = progressbar.ProgressBar()
    print('Dumping questions, answers, questionIDs, imageIDs, and questions lengths to text files...')
    imdir='%s/COCO_%s_%012d.jpg'
    N = len(ques)
    print('')
    print('{} Writing {} questions file {}'.format('*'*10, args.split, '*'*10))
    with open(questions_file, 'w') as f:
        for i, q in zip(range(N), ques):
            f.write((q['question'] + '\n').encode('utf8'))
            print('{}/{} written.'.format(i, N), end = '\r')
            sys.stdout.flush()
    print('{} Done writing {} questions file {}'.format('*'*10, args.split, '*'*10))
    print('')     
    print('{} Writing {} questions lengths file {}'.format('*'*10, args.split, '*'*10))
    with open(questions_lengths_file, 'w') as f:
        for i, q in zip(range(N), ques):
            f.write((str(len(tokenize(q['question'])))+ '\n').encode('utf8'))
            print('{}/{} written.'.format(i, N), end = '\r')
            sys.stdout.flush()
    print('{} Done writing {} questions length file {}'.format('*'*10, args.split, '*'*10))
    print('') 
    print('{} Writing {} questions id file {}'.format('*'*10, args.split, '*'*10))
    with open(questions_id_file, 'w') as f:
        for i, q in zip(range(N), ques):
            f.write((str(q['question_id']) + '\n').encode('utf8'))
            print('{}/{} written.'.format(i, N), end = '\r')
            sys.stdout.flush()
    print('{} Done writing {} questions id file {}'.format('*'*10, args.split, '*'*10))
    print('') 
    print('{} Writing {} coco_image id file {}'.format('*'*10, args.split, '*'*10))
    with open(coco_image_id, 'w') as f:
        for i, q in zip(range(N), ques):
            f.write((str(q['image_id']) + '\n').encode('utf8'))
            print('{}/{} written.'.format(i, N), end = '\r')
            sys.stdout.flush()
    print('{} Done writing {} coco_image id file {}'.format('*'*10, args.split, '*'*10))
    print('') 
    print('{} Writing {} coco_image_path file {}'.format('*'*10, args.split, '*'*10))
    with open(coco_image_path, 'w') as f:
        for i, q in zip(range(N), ques):
            image_path = imdir%(subtype, subtype, int(q['image_id']))
            f.write((image_path + '\n').encode('utf8'))
            print('{}/{} written.'.format(i, N), end = '\r')
            sys.stdout.flush()
    print('{} Done writing {} coco_image_path file {}'.format('*'*10, args.split, '*'*10))
    print('') 
    print('{} Writing {} answers file {}'.format('*'*10, args.split, '*'*10))
    with open(answers_file, 'w') as f:
        for i, q in zip(range(N), ques):
            if args.answers == 'modal':
                f.write(getModalAnswer(qa[i]['answers']).encode('utf8'))
            elif args.answers == 'all':
                f.write(getAllAnswer(qa[i]['answers']).encode('utf8'))
            f.write('\n'.encode('utf8'))
    print('{} Done writing {} answers file {}'.format('*'*10, args.split, '*'*10))
    print('') 

    print('completed dumping {}'.format(data_split))

if __name__ == "__main__":
    main()
