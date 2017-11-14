#!/bin/bash
# Downloads the training and validation sets from visualqa.org. 

wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
#wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
#wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
unzip \*.zip
