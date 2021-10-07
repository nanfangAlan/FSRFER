from collections import OrderedDict

import tensorflow as tf
import pickle
from sklearn.svm import SVC
from lib.network import Generator, Discriminator, Perceptual_VGG19, ACDiscriminator
from lib.covpoolnet import inference_middle
import os
class Network(object):
    """class to build networks"""
    def __init__(self, FLAGS, LR_data=None, HR_data=None):
        self.FLAGS = FLAGS
        self.LR_data = LR_data
        self.HR_data = HR_data


    def generator(self):

        with tf.name_scope('generator'):
            with tf.variable_scope('generator'):
                gen_out = Generator(self.FLAGS).build(self.LR_data)

        return gen_out

    def discriminator(self, gen_out):
        discriminator = Discriminator(self.FLAGS)

        with tf.name_scope('discriminator'):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                dis_out = discriminator.build(gen_out)

        return  dis_out


class Loss(object):
    """class to build loss functions"""
    def __init__(self):
        self.summary_target = OrderedDict()

    def center_loss(self,features, label, alfa=0.9, nrof_classes=7):
        """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
           (http://ydwen.github.io/papers/WenECCV16.pdf)
        """
        nrof_features = features.get_shape()[1]
        with tf.variable_scope('center_loss',reuse=tf.AUTO_REUSE):
            centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0), trainable=False)
            label = tf.reshape(label, [-1])
            centers_batch = tf.gather(centers, label)
            diff = (1 - alfa) * (centers_batch - features)
            centers = tf.scatter_sub(centers, label, diff)
            loss = tf.reduce_mean(tf.square(features - centers_batch))
        return loss, centers

    def pretrain_loss(self, pre_gen_out, HR_data):
        with tf.name_scope('loss_function'):
            with tf.variable_scope('feature-wise_loss'):

                pre_gen_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(pre_gen_out - HR_data), axis=[1, 2, 3]))


        self.summary_target['pre-train : feature-wise_loss'] = pre_gen_loss
        return pre_gen_loss

    def _perceptual_vgg19_loss(self, HR_data, gen_out):
        with tf.name_scope('perceptual_vgg19_HR'):
            with tf.variable_scope('perceptual_vgg19', reuse=False):
                vgg_out_hr = Perceptual_VGG19().build(HR_data)

        with tf.name_scope('perceptual_vgg19_Gen'):
            with tf.variable_scope('perceptual_vgg19', reuse=True):
                vgg_out_gen = Perceptual_VGG19().build(gen_out)

        return vgg_out_hr, vgg_out_gen

    def gan_loss(self, FLAGS, HR_data, gen_out, dis_out_real, dis_out_fake, gradients, Label_data, prelogits_fake=None, logits_fake=None,prelogits_real=None, logits_real=None,prelogits_lr=None,logits_lr=None,prelogits_fake_128=None, prelogits_real_128=None, pre_logits_lr_128=None):#这里加上了Label作为输入，需要用来定义损失了。

        with tf.name_scope('loss_function'):
            with tf.variable_scope('loss_generator'):
                #Take out the probability of correct classification in Logits
                prob = tf.nn.softmax(logits_fake)
                label_one_hot = tf.one_hot(Label_data, 7)
                masked_prob = tf.multiply(label_one_hot, prob)

                #re-weighting
                class_reweight = tf.reduce_max(masked_prob, axis=1)
                class_reweight = (1.5 - class_reweight)**2


                if FLAGS.gan_loss_type == 'RaGAN':
                    g_loss_p1 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                                                                labels=tf.zeros_like(dis_out_real)))

                    g_loss_p2 = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                                                                labels=tf.ones_like(dis_out_fake)))

                    gen_loss = FLAGS.gan_loss_coeff * (g_loss_p1 + g_loss_p2) / 2
                elif FLAGS.gan_loss_type == 'GAN':
                    gen_loss = FLAGS.gan_loss_coeff * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake, labels=tf.ones_like(dis_out_fake)))

                elif FLAGS.gan_loss_type == 'WGAN':
                    gen_loss =  -tf.reduce_mean(dis_out_fake)
                    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Label_data, logits=logits_fake, name='cross_entropy_per_example')
                    class_loss = tf.reduce_mean(class_loss, name='class_loss')
                    gen_loss = FLAGS.gan_loss_coeff*gen_loss

                elif FLAGS.gan_loss_type == 'WGAN_div':
                    gen_loss =  tf.reduce_mean(tf.multiply(class_reweight,dis_out_fake))
                    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Label_data, logits=logits_fake, name='cross_entropy_per_example')
                    class_loss = tf.reduce_mean(class_loss, name='class_loss')
                    gen_loss = gen_loss

                elif FLAGS.gan_loss_type == 'ACWGAN':
                    gen_loss = -tf.reduce_mean(dis_out_fake)

                else:
                    raise ValueError('Unknown GAN loss function type')



                class_loss_real = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Label_data, logits=logits_real,
                                                                                 name='cross_entropy_per_example_real')
                class_loss_real = tf.reduce_mean(class_loss_real, name='class_loss_real')
                class_loss_lr = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Label_data, logits=logits_lr,
                                                                               name='cross_entropy_per_example_lr')
                class_loss_lr = tf.reduce_mean(class_loss_lr, name='class_loss_lr')


                # content loss : L1 distance
                content_loss = FLAGS.content_loss_coeff * tf.reduce_mean(
                    tf.reduce_sum(tf.abs(gen_out - HR_data), axis=[1, 2, 3]))

                #feature L2 loss
                focal_loss = FLAGS.focal_loss_coeff*tf.reduce_mean(tf.multiply(class_reweight,tf.reduce_sum(tf.square(gen_out - HR_data), axis=[1, 2, 3])))
                gen_loss+= focal_loss


                # perceptual loss
                if FLAGS.perceptual_loss == 'pixel-wise':
                    perc_loss = tf.reduce_mean(tf.reduce_mean(tf.square(tf.square(gen_out - HR_data)), axis=[1,2,3]))
                    gen_loss += perc_loss
                elif FLAGS.perceptual_loss == 'VGG19':
                    vgg_out_gen, vgg_out_hr = self._perceptual_vgg19_loss(HR_data, gen_out)
                    perc_loss = tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                    gen_loss += perc_loss
                elif FLAGS.perceptual_loss == 'inference':
                    perc_loss = tf.reduce_mean(tf.multiply(class_reweight,tf.reduce_sum(tf.square(prelogits_real_128-prelogits_fake_128), axis=1)))
                    gen_loss += 10*perc_loss
                else:
                    raise ValueError('Unknown perceptual loss type')

            with tf.variable_scope('loss_discriminator'):
                if FLAGS.gan_loss_type == 'RaGAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real - tf.reduce_mean(dis_out_fake),
                                                                labels=tf.ones_like(dis_out_real))) / 2

                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake - tf.reduce_mean(dis_out_real),
                                                                labels=tf.zeros_like(dis_out_fake))) / 2

                    dis_loss = d_loss_real + d_loss_fake
                elif FLAGS.gan_loss_type == 'GAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_real, labels=tf.ones_like(dis_out_real)))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fake,
                                                                labels=tf.zeros_like(dis_out_fake)))

                    dis_loss = d_loss_real + d_loss_fake
                elif FLAGS.gan_loss_type == 'WGAN':
                    d_loss_real = tf.reduce_mean(dis_out_real)
                    d_loss_fake = tf.reduce_mean(dis_out_fake)
                    dis_loss =  d_loss_fake - d_loss_real
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    dis_loss +=10*gradient_penalty

                elif FLAGS.gan_loss_type == 'WGAN_div':

                    d_loss_real = tf.reduce_mean(tf.multiply(class_reweight,dis_out_real))
                    d_loss_fake = tf.reduce_mean(tf.multiply(class_reweight,dis_out_fake))
                    dis_loss = (-d_loss_fake + d_loss_real)
                    slopes = tf.pow(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]),3)
                    gradient_penalty = tf.reduce_mean(slopes)
                    dis_loss +=2*gradient_penalty


                elif FLAGS.gan_loss_type == 'ACWGAN':
                    cls_real_loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Label_data, logits=logits_real, name='cross_entropy_real'))
                    dis_loss = cls_real_loss
                else:
                    raise ValueError('Unknown GAN loss function type')

            self.summary_target['generator_loss'] = gen_loss
            self.summary_target['content_loss'] = content_loss
            self.summary_target['gradient_penalty'] = gradient_penalty
            self.summary_target['class_loss'] = class_loss
            self.summary_target['class_loss_real'] = class_loss_real
            self.summary_target['class_loss_lr'] = class_loss_lr
            self.summary_target['perceptual_loss'] = perc_loss
            self.summary_target['discriminator_loss'] = dis_loss
            self.summary_target['discriminator_real_loss'] = d_loss_real
            self.summary_target['discriminator_fake_loss'] = d_loss_fake
            self.summary_target['focal_loss'] = focal_loss

        return gen_loss, dis_loss

    def add_summary_writer(self):
        return [tf.summary.scalar(key, value) for key, value in self.summary_target.items()]


class Optimizer(object):
    """class to build optimizers"""
    @staticmethod
    def pretrain_optimizer(FLAGS, global_iter, pre_gen_loss):
        learning_rate = tf.train.exponential_decay(FLAGS.pretrain_learning_rate, global_iter,
                                                   FLAGS.pretrain_lr_decay_step, 0.5, staircase=True)

        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_generator'):
                pre_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                pre_gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=pre_gen_loss,
                                                                                                 global_step=global_iter,
                                                                                                 var_list=pre_gen_var)

        return pre_gen_var, pre_gen_optimizer

    @staticmethod
    def gan_optimizer(FLAGS, global_iter, dis_loss, gen_loss):


        #Reduce learning rate by stages in the middle
        boundaries = [10000, 20000, 50000, 100000,200000]
        values_g = [FLAGS.learning_rate, FLAGS.learning_rate, FLAGS.learning_rate * 0.5, FLAGS.learning_rate * 0.5 ,
                    FLAGS.learning_rate * 0.5 ** 2, FLAGS.learning_rate * 0.5 ** 3]
        learning_rate_g = tf.train.piecewise_constant(global_iter, boundaries, values_g)
        values_d = [FLAGS.learning_rate, FLAGS.learning_rate, FLAGS.learning_rate * 0.5, FLAGS.learning_rate * 0.5,
                    FLAGS.learning_rate * 0.5 ** 2, FLAGS.learning_rate * 0.5 ** 3]
        learning_rate_d = tf.train.piecewise_constant(global_iter, boundaries, values_d)


        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,beta1=0.).minimize(loss=dis_loss,
                                                                                             var_list=dis_var)

            with tf.variable_scope('optimizer_generator'):
                with tf.control_dependencies([dis_optimizer]):
                    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_g,beta1=0.).minimize(loss=gen_loss,
                                                                                                 global_step=global_iter,
                                                                                                 var_list=gen_var)


        return dis_var, dis_optimizer, gen_var, gen_optimizer
