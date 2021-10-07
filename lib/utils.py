import logging
import os
import glob
import random
import cv2
import re
import numpy as np
import pickle
import tensorflow as tf
import importlib
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
import scipy.misc as misc


def log(logflag, message, level='info'):
    """logging to stdout and logfile if flag is true"""
    print(message, flush=True)

    if logflag:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'critical':
            logging.critical(message)


def create_dirs(target_dirs):
    """create necessary directories to save output files"""
    for dir_path in target_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def normalize_images(*arrays):
    """normalize input image arrays"""
    return [arr / 127.5 - 1 for arr in arrays]


def de_normalize_image(image):
    """de-normalize input image array"""
    return (image + 1) * 127.5


def save_image(FLAGS, images, phase, global_iter, save_max_num=5):
    """save images in specified directory"""
    if phase == 'train' or phase == 'pre-train':
        save_dir = FLAGS.train_result_dir
    elif phase == 'inference':
        save_dir = FLAGS.inference_result_dir
        save_max_num = len(images)
    else:
        print('specified phase is invalid')

    for i, img in enumerate(images):
        if i >= save_max_num:
            break

        cv2.imwrite(save_dir + '/{0}_HR_{1}_{2}.jpg'.format(phase, global_iter, i), de_normalize_image(img))


def crop(img, FLAGS):
    """crop patch from an image with specified size"""
    img_h, img_w, _ = img.shape

    rand_h = np.random.randint(img_h - FLAGS.crop_size)
    rand_w = np.random.randint(img_w - FLAGS.crop_size)

    return img[rand_h:rand_h + FLAGS.crop_size, rand_w:rand_w + FLAGS.crop_size, :]


def data_augmentation(LR_images, HR_images, aug_type='horizontal_flip'):
    """data augmentation. input arrays should be [N, H, W, C]"""

    if aug_type == 'horizontal_flip':
        return LR_images[:, :, ::-1, :], HR_images[:, :, ::-1, :]
    elif aug_type == 'rotation_90':
        return np.rot90(LR_images, k=1, axes=(1, 2)), np.rot90(HR_images, k=1, axes=(1, 2))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        need_load_name = ['DLP-CNN/Conv2d_1/weights:0', 'DLP-CNN/Conv2d_1/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2d_1/BatchNorm/moving_mean:0', 'DLP-CNN/Conv2d_1/BatchNorm/moving_variance:0', \
                          'DLP-CNN/Conv2d_2/weights:0', 'DLP-CNN/Conv2d_2/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2d_2/BatchNorm/moving_mean:0', 'DLP-CNN/Conv2d_2/BatchNorm/moving_variance:0', \
                          'DLP-CNN/Conv2d_3/weights:0', 'DLP-CNN/Conv2d_3/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2d_3/BatchNorm/moving_mean:0', 'DLP-CNN/Conv2d_3/BatchNorm/moving_variance:0', \
                          'DLP-CNN/Conv2d_4/weights:0', 'DLP-CNN/Conv2d_4/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2d_4/BatchNorm/moving_mean:0', 'DLP-CNN/Conv2d_4/BatchNorm/moving_variance:0', \
                          'DLP-CNN/Conv2d_5/weights:0', 'DLP-CNN/Conv2d_5/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2d_5/BatchNorm/moving_mean:0', 'DLP-CNN/Conv2d_5/BatchNorm/moving_variance:0', \
                          'DLP-CNN/Conv2dfo_6/weights:0', 'DLP-CNN/Conv2dfo_6/BatchNorm/beta:0', \
                          'DLP-CNN/Conv2dfo_6/BatchNorm/moving_mean:0',
                          'DLP-CNN/Conv2dfo_6/BatchNorm/moving_variance:0', \
                          'DLP-CNN/spdpooling1/orth_weight0:0', 'DLP-CNN/fc_1/weights:0','DLP-CNN/fc_1/biases:0',
                          'DLP-CNN/fc_1/BatchNorm/beta:0', \
                          'DLP-CNN/fc_1/BatchNorm/moving_mean:0', 'DLP-CNN/fc_1/BatchNorm/moving_variance:0', \
                          'DLP-CNN/fc_2/weights:0','DLP-CNN/fc_2/biases:0', 'DLP-CNN/fc_2/BatchNorm/beta:0',
                          'DLP-CNN/fc_2/BatchNorm/moving_mean:0', \
                          'DLP-CNN/fc_2/BatchNorm/moving_variance:0', 'DLP-CNN/Bottleneck/weights:0', \
                          'DLP-CNN/Bottleneck/BatchNorm/beta:0', 'DLP-CNN/Bottleneck/BatchNorm/moving_mean:0', \
                          'DLP-CNN/Bottleneck/BatchNorm/moving_variance:0','Logits/weights:0','Logits/biases:0']
        need_load = [n for n in tf.contrib.framework.get_variables_to_restore() if n.name in need_load_name]

        logits = [n for n in tf.contrib.framework.get_variables_to_restore() if 'biases' in n.name]
        print(logits)

        saver = tf.train.Saver(need_load)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        return need_load

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_and_save_data(FLAGS, logflag):
    """make HR and LR data. And save them as npz files"""
    assert os.path.isdir(FLAGS.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'
    assert os.path.isdir(FLAGS.lr_data_dir) is True, 'Directory specified by lr_data_dir does not exist or is not a directory'
    paths = FLAGS.data_dir
    lr_paths = FLAGS.lr_data_dir
    ret_HR_image = []
    ret_LR_image = []
    ret_label = []
    lr_ret_label = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                images.sort()
                images = [img for img in images if img.endswith('.jpg') or img.endswith('.png')]
                image_paths = [os.path.join(facedir, img) for img in images]
                for image_path in image_paths:
                    HR_image = misc.imread(image_path)
                    HR_image = prewhiten(HR_image)
                    ret_HR_image.append(HR_image)
                    ret_label.append(i)

    for path in lr_paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                images.sort()
                images = [img for img in images if img.endswith('.jpg') or img.endswith('.png')]
                image_paths = [os.path.join(facedir, img) for img in images]
                for image_path in image_paths:
                    LR_image = misc.imread(image_path)

                    LR_image = prewhiten(LR_image)
                    ret_LR_image.append(LR_image)
                    lr_ret_label.append(i)

    ret_HR_image = np.array(ret_HR_image)
    ret_LR_image = np.array(ret_LR_image)
    ret_label = np.array(ret_label)
    print(ret_HR_image.shape)
    print(ret_LR_image.shape)
    assert ret_HR_image.shape == ret_LR_image.shape , 'LR images number is not equal to HR images!'
    assert (lr_ret_label==ret_label).all() , 'labels are not equal!'

    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename, images=ret_HR_image)
    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename, images=ret_LR_image)
    np.savez(FLAGS.npz_data_dir + '/' + FLAGS.Label_npz_filename, labels=ret_label)

    return ret_HR_image, ret_LR_image, ret_label


def shuffle_examples(HR_images, LR_images, labels):
    shuffle_list = list(zip(HR_images, LR_images, labels))
    random.shuffle(shuffle_list)
    HR_images_shuff, LR_images_shuff,  labels_shuff = zip(*shuffle_list)
    return HR_images_shuff, LR_images_shuff, labels_shuff

def load_npz_data(FLAGS):
    """load array data from data_path"""
    return np.load(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename)['images'], \
           np.load(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename)['images'], \
           np.load(FLAGS.npz_data_dir + '/' + FLAGS.Label_npz_filename)['labels']

def load_inference_data(FLAGS):
    """load data from directory for inference"""
    assert os.path.isdir(FLAGS.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_LR_image = []
    ret_filename = []

    for file in all_file_path:
        img = cv2.imread(file)
        img = normalize_images(img)
        ret_LR_image.append(img[0][np.newaxis, ...])

        ret_filename.append(file.rsplit('/', 1)[-1])

    assert len(ret_LR_image) > 0, 'No available image is found in the directory'

    return ret_LR_image, ret_filename
