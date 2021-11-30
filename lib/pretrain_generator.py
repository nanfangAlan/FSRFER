import gc
import os
import math

from sklearn.utils import shuffle
import tensorflow as tf

from lib.ops import scale_initialization
from lib.train_module_full import Network, Loss, Optimizer
from lib.utils import log, normalize_images, save_image, load_model


from lib.covpoolnet import inference, inference_middle

def train_pretrain_generator(FLAGS, LR_train, HR_train, logflag, pre_gen_dir):
    """pre-train deep network as initialization weights of ESRGAN Generator"""
    log(logflag, 'Pre-train : Proce ss start', 'info')

    LR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.LR_image_size, FLAGS.LR_image_size, FLAGS.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel],
                             name='HR_input')

    LR_feature = inference(LR_data,batch_size = FLAGS.batch_size)
    HR_feature = inference(HR_data,batch_size = FLAGS.batch_size)
    network = Network(FLAGS, LR_feature, HR_feature)
    pre_gen_out = network.generator()

# build loss function
    loss = Loss()
    pre_gen_loss = loss.pretrain_loss(pre_gen_out, HR_feature)

    # build optimizer
    global_iter = tf.Variable(0, trainable=False)
    pre_gen_var, pre_gen_optimizer = Optimizer().pretrain_optimizer(FLAGS, global_iter, pre_gen_loss)

    # build summary writer
    pre_summary = tf.summary.merge(loss.add_summary_writer())

    num_train_data = len(HR_train)
    num_batch_in_train = int(math.floor(num_train_data / FLAGS.batch_size))
    num_epoch = int(math.ceil(FLAGS.pre_iter / num_batch_in_train))

    fetches = {'pre_gen_loss': pre_gen_loss, 'pre_gen_optimizer': pre_gen_optimizer, 'gen_HR': pre_gen_out,
               'summary': pre_summary}

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
        allow_growth=True,
        )
    )

    saver = tf.train.Saver(max_to_keep=10)

    # Start session
    with tf.Session(config=config) as sess:
        log(logflag, 'Pre-train : Training starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(load_model(FLAGS.fer_model_checkpoint_dir))
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, FLAGS))

        writer = tf.summary.FileWriter(FLAGS.logdir, graph=sess.graph, filename_suffix='pre-train')

        for epoch in range(num_epoch):
            log(logflag, 'Pre-train Epoch: {0}'.format(epoch), 'info')

            HR_train, LR_train = shuffle(HR_train, LR_train, random_state=222)

            for iteration in range(num_batch_in_train):
                current_iter = tf.train.global_step(sess, global_iter)

                if current_iter > FLAGS.num_iter:
                    break

                feed_dict = {
                    HR_data: HR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size],
                    LR_data: LR_train[iteration * FLAGS.batch_size:iteration * FLAGS.batch_size + FLAGS.batch_size]
                }

                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                # save summary every n iter
                if current_iter % FLAGS.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % FLAGS.train_sample_save_freq == 0:
                    log(logflag,
                        'Pre-train iteration : {0}, pixel-wise_loss : {1}'.format(current_iter, result['pre_gen_loss']),
                        'info')

                # save checkpoint
                if current_iter % FLAGS.train_ckpt_save_freq == 0:
                    if current_iter != 0:
                        saver.save(sess, pre_gen_dir, global_step=current_iter, write_meta_graph=False)
                    else:
                        saver.save(sess, pre_gen_dir, global_step=current_iter)

        writer.close()
        log(logflag, 'Pre-train : Process end', 'info')
