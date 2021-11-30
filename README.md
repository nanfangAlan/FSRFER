## FSR-FER (TensorFlow)

This repository provides a TensorFlow implementation of the paper "Feature Super-Resolution Based Facial Expression Recognition for Multi-scale Low-Resolution Images"
More details will be added in the future.

## Dependencies

tensorflow, openCV, sklearn, numpy

The versions of my test environment :  
Python==3.6.8, tensorflow-gpu==1.12.0, openCV==4.1.0,  scikit-learn==0.20.3, numpy==1.16.2

## Models
You can download following models:
The [CovPoolFER models](https://github.com/d-acharya/CovPoolFER) were used as the facial expression recognition model in our paper.
You could download the pretrained models in this depository and unzip them in the project, make sure the pretrained CovPoolFER model path is "./models/model2/20170815-144407"

## Datasets

The LR image should be downsample by (the default function used by "imresize()" function)with matlab, and the path should be like:
+--SFEW
|  +--
|  +--

The pretrained FSR-FER model will be added in the future.
