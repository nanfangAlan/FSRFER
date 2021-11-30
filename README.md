## FSR-FER (TensorFlow)

This repository provides a TensorFlow implementation of the paper "Feature Super-Resolution Based Facial Expression Recognition for Multi-scale Low-Resolution Images"
More details will be added in the future.

## Dependencies

tensorflow, openCV, sklearn, numpy

The versions of my test environment :  
Python==3.6.8, tensorflow-gpu==1.12.0, openCV==4.1.0,  scikit-learn==0.20.3, numpy==1.16.2

## Models
You can download following models:

- The [CovPoolFER models](https://github.com/d-acharya/CovPoolFER) were used as the facial expression recognition model in our paper.
  You could download the pretrained models in this depository and unzip them in the project, make sure the dir has pretrained CovPoolFER model path "./models/model2/20170815-144407"
- The pretrained FSR-FER model will be added in the future.

## Datasets

The LR images were downsampled and resized to 100x100 with imresize() function in Matlab.  Append "x2-x8" to the filename and put LR images in the same dir. The HR and LR dataset path should be like this:

**+--RAFDB_train_lr**
|-- minitest
|   |-- Angry
|        |-- test_0027x2.jpg
|        |-- test_0027x3.jpg
|        |-- test_0027x4.jpg
|        |-- test_0027x5.jpg
|        |-- test_0027x6.jpg
|        |-- test_0027x7.jpg
|        |-- test_0027x8.jpg
|        |-- test_0037x2.jpg
|        \`-- ...
|   |-- Disgust
|   |-- Fear
|   |-- Happy
|   |-- Neutral
|   |-- Sad
|   \`-- Surprise

**+--RAFDB_train_hr**
|-- test_100
|   |-- Angry
|        |-- test_0027.jpg
|        |-- test_0037.jpg
|        |-- test_0047.jpg
|        \`-- ...
|   |-- Disgust
|   |-- Fear
|   |-- Happy
|   |-- Neutral
|   |-- Sad
|   \`-- Surprise



## Train and Test
Use train.sh to train the FSRFER model and load the trained model with classify.sh to train the classifier and classify features. 

|-- classify
|   |-- classify.sh                    
|   |-- classify_FER.py            (train and test the FER classifier)
|   |-- classify_FSRFER.py      (train and test the FSRFER classifier)
|   \`-- framework.py
|-- data
|-- lib
|   |-- covpoolnet.py
|   |-- network.py
|   |-- ops.py
|   |-- pretrain_generator.py
|   |-- train_module_full.py
|   \`-- utils.py
|-- nohup.out
|-- train.py                              (train the FSRFER feature extractor)
\`-- train.sh
