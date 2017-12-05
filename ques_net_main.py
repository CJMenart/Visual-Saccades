import os
from tensorflow.python.client import device_lib
import tensorflow as tf
import argparse
from QUES_NET import QUES_NET
import sys
LR = 0.0002
BATCH_SIZE = 256
EPOCH = 300
NUM_HIDDEN_LAYER = 5
HIDDEN_LAYER_SIZE = 300
OUT_LAYER_SIZE = 1000
WORD_EMBED_SIZE = 300
GPUS = '2'
IS_TRAIN = True
TRAIN_PATH = 'data/train_data_small.tfrecords'
VAL_PATH = 'data/val_data_small.tfrecords'
RESULT_PATH = 'Results'
SAVE_DIR = 'model'
IS_BNORM = True
LBL_ENC_FILE = 'data/labelencoder.pkl'
KEEP_PROB = 0.8
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    
    
    parser = argparse.ArgumentParser(description='Adversarial Training DNN')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate for training. Default: ' + str(LR) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='Number of Epoch to train. Default: ' + str(EPOCH) + '.')
  
    parser.add_argument('--num_hidden_layer', type=int, default=NUM_HIDDEN_LAYER, help='Number of hidden layers '
                        'in feed forward classifier. Default: ' + str(NUM_HIDDEN_LAYER) + '.')
    parser.add_argument('--hidden_layer_size', type=int, default=HIDDEN_LAYER_SIZE, help='Size of hidden layers '
                        'in feed forward classifier. Default: ' + str(HIDDEN_LAYER_SIZE) + '.')
    parser.add_argument('--out_layer_size', type=int, default=OUT_LAYER_SIZE, help='Number of classes in the '
                        'classifier. Default: ' + str(OUT_LAYER_SIZE) + '.')
    parser.add_argument('--word_embed_size', type=int, default=WORD_EMBED_SIZE, help='Size of word embedding'
                        ' vector. Default: ' + str(WORD_EMBED_SIZE) + '.')
    parser.add_argument('--gpus', type=str, default=GPUS, help='List of GPUS to use. Default: ' + str(GPUS) + '.')
 
    parser.add_argument('--is_train', type=_str_to_bool, default=IS_TRAIN, help='Whether to train. '
                        'Default: ' + str(IS_TRAIN) + ', Train the network')
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH, help='Path to train tfrecord file. '
                        'Default: ' + str(TRAIN_PATH) + '.')
    parser.add_argument('--val_path', type=str, default=VAL_PATH, help='Path to val tfrecord file. '
                        'Default: ' + str(VAL_PATH) + '.')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Directory path where to save model parameters. '
                        'Default: ' + str(SAVE_DIR) + '.')
    parser.add_argument('--is_bnorm', type=_str_to_bool, default=IS_BNORM, help='Whether to use batch norm ' 
                        'in fully connected layer or not. Default: ' + str(IS_BNORM) + ', Train the network')
    parser.add_argument('--lbl_enc_file', type=str, default=LBL_ENC_FILE, help='Path for labelencoder.pkl. '
                        'Default: ' + str(LBL_ENC_FILE) + '.')
    parser.add_argument('--result_path', type=str, default=RESULT_PATH, help='Path for labelencoder.pkl. '
                        'Default: ' + str(RESULT_PATH) + '.')
    parser.add_argument('--keep_prob', type=float, default=KEEP_PROB, help='Dropout keep '
                        'probability in hidden layers of feed forward net in '
                        'question. Default: ' + str(KEEP_PROB) + '.')
    args = parser.parse_args()
    return args

args = get_arguments()
print(args)
os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
devices = device_lib.list_local_devices()      
udevices = [] 
for device in devices:
    if len(devices) > 1 and 'cpu' in device.name.lower():
        # Use cpu only when we dont have gpus
        continue
    print('Using device: ', device.name)
    udevices.append(device.name)
    
print(udevices)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
tf.reset_default_graph()
with tf.Session(config=config) as sess:
    if args.is_train:
        model = QUES_NET(sess, udevices, args, infer = False)
        model.train()
    
      
