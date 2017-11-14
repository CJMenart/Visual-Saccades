#!/bin/bash
cd data
./download.sh
cd ../
./txt_dump.sh
python write_tfrecords.py
./train_lstm.sh