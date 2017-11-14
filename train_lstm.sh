#!/bin/bash
HIDDEN_KEEP_PROB=0.5
IS_BNORM=True
RESULT_PATH=result5_h2_outer_conv
GPUS='1'
SAVE_DIR=model5_h2_outer_conv
VAL_PATH='val_data_5.npz'
TFRECORDS_PATH='data_5.tfrecords'
LBL_ENC_FILE='labelencoder_5.pkl'
VOCAB_LIST='data/preprocessed/vocab_list_5.txt'
USE_PEEPHOLES=True
FEAT_JOIN='outer_conv'
NUM_HIDDEN_LAYER=2
BATCH_SIZE=64
python main_lstm_dnn.py --hidden_keep_prob=$HIDDEN_KEEP_PROB\
                        --is_bnorm=$IS_BNORM\
                        --result_path=$RESULT_PATH\
                        --gpus=$GPUS\
                        --save_dir=$SAVE_DIR\
                        --val_path=$VAL_PATH\
                        --tfrecords_path=$TFRECORDS_PATH\
                        --lbl_enc_file=$LBL_ENC_FILE\
                        --vocab_list=$VOCAB_LIST\
                        --use_peepholes=$USE_PEEPHOLES\
                        --feat_join=$FEAT_JOIN\
                        --num_hidden_layer=$NUM_HIDDEN_LAYER\
                        --batch_size=$BATCH_SIZE
