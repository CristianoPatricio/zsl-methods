"""
NOTE:
1. This code was originally created by the author of the GitHub repository https://github.com/Hanzy1996/CE-GZSL
2. All changes were made by the author of the current repository in order to adapt the output to the standardized nomenclature of the rest of
the implemented methods.
"""

from __future__ import print_function
import os, os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

tf.get_logger().setLevel('ERROR')

import sys
import random
import argparse
import util
import classifier2

###################functions######################################################3
def generator(x, opt, name="generator", reuse=False, isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=opt.ngh, \
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02), \
                              activation=tf.nn.leaky_relu, name='gen_fc1', trainable=isTrainable, reuse=reuse)

        net = tf.layers.dense(inputs=net, units=opt.resSize, \
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                              activation=tf.nn.relu, name='gen_fc2', trainable=isTrainable, reuse=reuse)
        # the output is relu'd as the encoded representation is also the activations by relu

        return tf.reshape(net, [-1, opt.resSize])


def discriminator(x, opt, name="discriminator", reuse=False, isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=opt.ndh, \
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02), \
                              activation=tf.nn.leaky_relu, name='disc_fc1', trainable=isTrainable, reuse=reuse)

        real_fake = tf.layers.dense(inputs=net, units=1, \
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02), \
                                    activation=None, name='disc_rf', trainable=isTrainable, reuse=reuse)

        return tf.reshape(real_fake, [-1])


def classificationLayer(x, classes, name="classification", reuse=False, isTrainable=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        net = tf.layers.dense(inputs=x, units=classes, \
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02), \
                              activation=None, name='fc1', trainable=isTrainable, reuse=reuse)

        net = tf.reshape(net, [-1, classes])
    return net


def next_feed_dict(data, opt):
    batch_feature, batch_labels, batch_att = data.next_batch(opt.batch_size)
    batch_label = util.map_label(batch_labels, data.seenclasses)
    z_rand = np.random.normal(0, 1, [opt.batch_size, opt.nz]).astype(np.float32)

    return batch_feature, batch_att, batch_label, z_rand


"""
############## evaluation ################################################
if opt.gzsl:
    train_X = np.concatenate((data.train_feature, syn_res), axis=0)
    train_Y = np.concatenate((data.train_label, syn_label), axis=0)
    nclass = opt.nclass_all
    train_cls = classifier2.CLASSIFICATION2(train_X, train_Y, data, nclass, 'logs_gzsl_classifier',
                                            'models_gzsl_classifier', 0.001, 0.5, 25, opt.syn_num, True)
    print('unseen=%.4f, seen=%.4f, h=%.4f' % (train_cls.acc_unseen, train_cls.acc_seen, train_cls.H))

else:
    train_cls = classifier2.CLASSIFICATION2(syn_res, util.map_label(syn_label, data.unseenclasses), data,
                                            data.unseenclasses.shape[0], 'logs_zsl_classifier', 'models_zsl_classifier',
                                            0.001, 0.5, 25, opt.syn_num, False)
    acc = train_cls.acc
    print('unseen class accuracy= ', acc)
"""
