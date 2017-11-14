#!/bin/bash
cd data
./download.sh
cd ../
python dumpTxt.py
python get_unique_images.py
python extract_features.py
python write_tfrecords.py
./train_lstm.sh