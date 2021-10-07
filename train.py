from datetime import datetime
import gc
import logging
import math
import os

import tensorflow as tf
from sklearn.utils import shuffle

from lib.ops import load_vgg19_weight
from lib.pretrain_generator import train_pretrain_generator
from lib.train_module_full import Network, Loss, Optimizer
from lib.utils import create_dirs, log, normalize_images, save_image, load_npz_data, load_and_save_data, load_model
from lib.covpoolnet import inference, inference_middle, last


def set_flags():
    Flags = tf.app.flags

    # About data
    Flags.DEFINE_string('data_dir', '/opt/FSRFER/data/RAFDB/RAFDB_100_bicubicup/RAFDB_100_lowx1/minitest', 'data directory')
    Flags.DEFINE_string('lr_data_dir', '/opt/FSRFER/data/RAFDB/RAFDB_100_bicubicup/RAFDB_100_lowx1/minitest', 'lr data directory')
    Flags.DEFINE_string('npz_data_dir', './data/npz', 'The npz data dir')
    Flags.DEFINE_string('HR_npz_filename', 'HR_image_multiscale_whiten.npz', 'the filename of HR image npz file')
    Flags.DEFINE_string('LR_npz_filename', 'LR_image_multiscale_whiten.npz', 'the filename of LR image npz file')
    Flags.DEFINE_string('Label_npz_filename', 'Label.npz', 'the filename of Label npz file')
    Flags.DEFINE_boolean('save_data', True, 'Whether to load and save data as npz file')
    Flags.DEFINE_string('train_result_dir', './', 'output directory during training')
    Flags.DEFINE_boolean('crop', False, 'Whether image cropping is enabled')
    Flags.DEFINE_integer('crop_size', 128, 'the size of crop of training HR images')
    Flags.DEFINE_integer('num_crop_per_image', 2, 'the number of random-cropped images per image')
    Flags.DEFINE_boolean('data_augmentation', True, 'whether to augment data')

    # About Network
    Flags.DEFINE_integer('scale_SR', 4, 'the scale of super-resolution')
    Flags.DEFINE_integer('num_repeat_RRDB', 6, 'The number of repeats of RRDB blocks')
    Flags.DEFINE_float('residual_scaling', 0.2, 'residual scaling parameter')
    Flags.DEFINE_integer('initialization_random_seed', 111, 'random seed of networks initialization')
    Flags.DEFINE_string('perceptual_loss', 'inference', 'the part of loss function. "VGG19" or "pixel-wise" or "inference"')
    Flags.DEFINE_string('gan_loss_type', 'WGAN_div', 'the type of GAN loss functions. "RaGAN or GAN"')

    # About training
    Flags.DEFINE_integer('num_iter', 1000000, 'The number of iterations')
    Flags.DEFINE_integer('pre_iter', 40000, 'The number of iterations')
    Flags.DEFINE_integer('batch_size', 4, 'Mini-batch size')
    Flags.DEFINE_integer('channel', 3, 'Number of input/output image channel')
    Flags.DEFINE_integer('feature_channel', 1, 'Number of feature channel')
    Flags.DEFINE_boolean('pretrain_generator', False, 'Whether to pretrain generator')
    Flags.DEFINE_float('pretrain_learning_rate', 2e-4, 'learning rate for pretrain')
    Flags.DEFINE_float('pretrain_lr_decay_step', 20000, 'decay by every n iteration')
    Flags.DEFINE_float('learning_rate', 2e-5, 'learning rate')
    Flags.DEFINE_float('weight_initialize_scale', 0.1, 'scale to multiply after MSRA initialization')
    Flags.DEFINE_integer('HR_image_size', 100,
                         'Image width and height of HR image. This flag is valid when crop flag is set to false.')
    Flags.DEFINE_integer('LR_image_size', 100,
                         'Image width and height of LR image. This size should be 1/4 of HR_image_size exactly. '
                         'This flag is valid when crop flag is set to false.')
    Flags.DEFINE_float('epsilon', 1e-12, 'used in loss function')
    Flags.DEFINE_float('gan_loss_coeff', 0.1, 'used in perceptual loss')
    Flags.DEFINE_float('class_loss_coeff', 1., 'used in perceptual loss')
    Flags.DEFINE_float('content_loss_coeff', 0.001, 'used in content loss')
    Flags.DEFINE_float('focal_loss_coeff', 0.01, 'used in content loss')
    Flags.DEFINE_float('perceptual_loss_coeff', 0.1, 'used in content loss')

    # About log
    Flags.DEFINE_boolean('logging', True, 'whether to record training log')
    Flags.DEFINE_integer('train_sample_save_freq', 100, 'save samples during training every n iteration')
    Flags.DEFINE_integer('train_ckpt_save_freq', 1000, 'save checkpoint during training every n iteration')
    Flags.DEFINE_integer('train_summary_save_freq', 100, 'save summary during training every n iteration')
    Flags.DEFINE_string('pre_train_checkpoint_dir', '/opt/FSRFER/checkpoint/WGAN_div/2020-05-20T15-57-02-RAFDB305k-img', 'pre-train checkpoint directory')
    Flags.DEFINE_string('checkpoint_dir', './checkpoint/WGAN_div', 'checkpoint directory')
    Flags.DEFINE_string('logdir', './log/WGAN_div', 'log directory')

    # About GPU setting
    Flags.DEFINE_string('gpu_dev_num', '1', 'Which GPU to use for multi-GPUs.')

    return Flags.FLAGS


def set_logger(FLAGS):
    """set logger for training recording"""
    if FLAGS.logging:
        logfile = '{0}/training_logfile_{1}.log'.format(FLAGS.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        formatter = '%(levelname)s:%(asctime)s:%(message)s'
        logging.basicConfig(level=logging.INFO, filename=logfile, format=formatter, datefmt='%Y-%m-%d %I:%M:%S')

        return True
    else:
        print('No logging is set')
        return False


def main():
    # set flag
    FLAGS = set_flags()

    # make dirs
    target_dirs = [FLAGS.pre_train_checkpoint_dir, FLAGS.checkpoint_dir, FLAGS.logdir]
    create_dirs(target_dirs)

    # set logger
    logflag = set_logger(FLAGS)
    log(logflag, 'Training script start', 'info')

    # load data
    if FLAGS.save_data:
        log(logflag, 'Data process : Data processing start', 'info')
        HR_train, LR_train, Label = load_and_save_data(FLAGS, logflag)
        log(logflag, 'Data process : Data loading and data processing are completed', 'info')
    else:
        log(logflag, 'Data process : Data loading start', 'info')
        HR_train, LR_train, Label = load_npz_data(FLAGS)
        log(logflag,
            'Data process : Loading existing data is completed. {} images are loaded'.format(len(HR_train)),
            'info')

    # pre-train generator with pixel-wise loss and save the trained model
    if FLAGS.pretrain_generator:
        train_pretrain_generator(FLAGS, LR_train, HR_train, logflag)
        tf.reset_default_graph()
        gc.collect()
    # else:
    log(logflag, 'Pre-train : Pre-train skips and an existing trained model will be used', 'info')
    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')
    Label_data = tf.placeholder(tf.int64,shape=[None], name='Label')
    HR_feature = inference(HR_data,batch_size = FLAGS.batch_size)
    LR_feature = inference(LR_data,batch_size = FLAGS.batch_size)
    network = Network(FLAGS, LR_feature, HR_feature)
    alpha = tf.random_uniform(shape=[FLAGS.batch_size,1,1,1],minval=0.,maxval=1.)

    gen_out = network.generator()


    dis_out_real = network.discriminator(HR_feature)
    dis_out_fake = network.discriminator(gen_out)
    difference = gen_out - HR_feature
    interpolates = HR_feature + alpha * difference
    gradients = tf.gradients(network.discriminator(interpolates),[interpolates])[0]
    loss = Loss()
    prelogits_fake,prelogits_fake_128 = inference_middle(gen_out, batch_size=FLAGS.batch_size)
    prelogits_real,prelogits_real_128 = inference_middle(HR_feature, batch_size=FLAGS.batch_size)
    prelogits_lr,prelogits_lr_128 = inference_middle(LR_feature,batch_size=FLAGS.batch_size)
    logits_fake = last(prelogits_fake,batch_size=FLAGS.batch_size)
    logits_real = last(prelogits_real, batch_size=FLAGS.batch_size)
    logits_lr = last(prelogits_lr, batch_size=FLAGS.batch_size)
    embeddings_fake = tf.nn.l2_normalize(prelogits_fake, 1, 1e-10, name='embeddings')
    embeddings_real = tf.nn.l2_normalize(prelogits_real, 1, 1e-10, name='embeddings_real')
    gen_loss, dis_loss = loss.gan_loss(FLAGS, HR_feature, gen_out, dis_out_real, dis_out_fake,gradients, Label_data,prelogits_fake,logits_fake,prelogits_real,logits_real,prelogits_lr,logits_lr,prelogits_fake_128,prelogits_real_128,prelogits_lr_128)
    # define optimizers
    global_iter = tf.Variable(0, trainable=False)
    dis_var, dis_optimizer, gen_var, gen_optimizer = Optimizer().gan_optimizer(FLAGS, global_iter, dis_loss, gen_loss)
    # build summary writer
    tr_summary = tf.summary.merge(loss.add_summary_writer())
    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.num_iter / num_batch_in_train))
    fetches = {'dis_optimizer': dis_optimizer, 'gen_optimizer': gen_optimizer,
               'dis_loss': dis_loss, 'gen_loss': gen_loss,
               'summary': tr_summary
               }

    gc.collect()

    config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    # Start Session
    with tf.Session(config=config) as sess:
        log(logflag, 'Training FSRFER starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(load_model('/opt/FSRFER/fer_model/RAFDB&SFEW/20170815-144407'))
        sess.run(global_iter.initializer)

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        writer = tf.summary.FileWriter(FLAGS.logdir + '/' + TIMESTAMP, graph=sess.graph)
        gen_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        dis_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        pre_saver = tf.train.Saver(var_list=gen_list + dis_list)
        pre_saver.restore(sess, tf.train.latest_checkpoint(FLAGS.pre_train_checkpoint_dir))

        if FLAGS.perceptual_loss == 'VGG19':
            sess.run(load_vgg19_weight(FLAGS))

        saver = tf.train.Saver(max_to_keep=100)

        for epoch in range(num_epoch):
            log(logflag, 'FSRFER Epoch: {0}'.format(epoch), 'info')
            HR_train, LR_train, Label = shuffle(HR_train, LR_train, Label, random_state=222)
            for iteration in range(num_batch_in_train):
                current_iter = tf.train.global_step(sess, global_iter)
                if current_iter > FLAGS.num_iter:
                    break

                feed_dict = {
                    HR_data: HR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    LR_data: LR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    Label_data: Label[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size]
                }

                # update weights of G/D
                result = sess.run(fetches=fetches, feed_dict=feed_dict)

                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    log(logflag,
                        'FSRFER iteration : {0}, gen_loss : {1}, dis_loss : {2}'.format(current_iter,
                                                                                        result['gen_loss'],
                                                                                        result['dis_loss']
                                                                                                           ),
                        'info')
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir+ '/' + TIMESTAMP, 'gen'), global_step=current_iter)

        writer.close()
        log(logflag, 'Training FSRFER end', 'info')
        log(logflag, 'Training script end', 'info')


if __name__ == '__main__':
    main()
