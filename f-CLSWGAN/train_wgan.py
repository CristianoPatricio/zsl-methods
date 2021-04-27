from __future__ import print_function
import os, os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import random
import argparse
import util
import classifier2
from clswgan import *

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA1', help='AWA1')
parser.add_argument('--dataroot', default='../datasets', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=84, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_modeldir', default='./models_classifier',
                    help='folder to get classifier model checkpoints')
parser.add_argument('--classifier_checkpoint', type=int, default=14,
                    help='tells which ckpt file of tensorflow model to load')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--logdir', default='./logs_AWA1', help='folder to output and hel#p print losses')
parser.add_argument('--modeldir', default='./models_AWA1', help='folder to output  model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')

opt = parser.parse_args()

if not os.path.exists(opt.logdir):
    os.makedirs(opt.logdir)
if not os.path.exists(opt.modeldir):
    os.makedirs(opt.modeldir)

random.seed(opt.manualSeed)
tf.set_random_seed(opt.manualSeed)

########################################################
#   LOAD DATA
########################################################
data = util.DATA_LOADER(opt)
####################################################################################

g1 = tf.Graph()
g2 = tf.Graph()

################### graoh1 definition ##########################################################

with g1.as_default():
    ########## placeholderS ############################
    input_res = tf.placeholder(tf.float32, [opt.batch_size, opt.resSize], name='input_features')
    input_att = tf.placeholder(tf.float32, [opt.batch_size, opt.attSize], name='input_attributes')
    noise_z = tf.placeholder(tf.float32, [opt.batch_size, opt.nz], name='noise')
    input_label = tf.placeholder(tf.int32, [opt.batch_size], name='input_label')

    ########## model definition ###########################
    train = True
    reuse = False

    noise = tf.concat([noise_z, input_att], axis=1)

    gen_res = generator(noise, opt, isTrainable=train, reuse=reuse)
    classificationLogits = classificationLayer(gen_res, data.seenclasses.shape[0], isTrainable=False, reuse=reuse)

    targetEmbd = tf.concat([input_res, input_att], axis=1)
    targetDisc = discriminator(targetEmbd, opt, isTrainable=train, reuse=reuse)
    genTargetEmbd = tf.concat([gen_res, input_att], axis=1)
    genTargetDisc = discriminator(genTargetEmbd, opt, isTrainable=train, reuse=True)

    ############ classification loss #########################

    classificationLoss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classificationLogits, labels=input_label))

    ############ discriminator loss ##########################

    genDiscMean = tf.reduce_mean(genTargetDisc)
    targetDiscMean = tf.reduce_mean(targetDisc)
    discriminatorLoss = tf.reduce_mean(genTargetDisc - targetDisc)
    alpha = tf.random_uniform(shape=[opt.batch_size, 1], minval=0., maxval=1.)

    # differences = genTargetEnc - targetEnc
    # interpolates = targetEnc + (alpha*differences)
    interpolates = alpha * input_res + ((1 - alpha) * gen_res)
    interpolate = tf.concat([interpolates, input_att], axis=1)
    gradients = tf.gradients(discriminator(interpolate, opt, reuse=True, isTrainable=train), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradientPenalty = tf.reduce_mean((slopes - 1.) ** 2)

    gradientPenalty = opt.lambda1 * gradientPenalty
    discriminatorLoss = discriminatorLoss + gradientPenalty

    # Wasserstein generator loss
    genLoss = -genDiscMean
    generatorLoss = genLoss + opt.cls_weight * classificationLoss

    #################### getting parameters to optimize ####################
    discParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    generatorParams = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    #for params in discParams:
      #  print(params.name)
      #  print('...................')

    #for params in generatorParams:
      #  print(params.name)

    discOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.999)
    genOptimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.999)

    discGradsVars = discOptimizer.compute_gradients(discriminatorLoss, var_list=discParams)
    genGradsVars = genOptimizer.compute_gradients(generatorLoss, var_list=generatorParams)

    discTrain = discOptimizer.apply_gradients(discGradsVars)
    generatorTrain = genOptimizer.apply_gradients(genGradsVars)

    #################### what all to visualize  ############################
    tf.summary.scalar("DiscriminatorLoss", discriminatorLoss)
    tf.summary.scalar("ClassificationLoss", classificationLoss)
    tf.summary.scalar("GeneratorLoss", generatorLoss)
    tf.summary.scalar("GradientPenaltyTerm", gradientPenalty)
    tf.summary.scalar("MeanOfGeneratedImages", genDiscMean)
    tf.summary.scalar("MeanOfTargetImages", targetDiscMean)

    for g, v in discGradsVars:
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + str('grad'), g)

    for g, v in genGradsVars:
        tf.summary.histogram(v.name, v)
        tf.summary.histogram(v.name + str('grad'), g)

    merged_all = tf.summary.merge_all()

############### training g1 graph ################################################
k = 1

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(opt.logdir, sess.graph)
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classification')

    saver = tf.train.Saver(var_list=params)

    #for var in params:
      #  print(var.name + "\t")

    string = opt.classifier_modeldir + '/models_' + str(opt.classifier_checkpoint) + '.ckpt'
  #  print(string)
    try:
        saver.restore(sess, string)
    except:
        print("Previous weights not found of classifier")
        sys.exit(0)

  #  print("Model loaded")
    saver = tf.train.Saver()

    print(f"[INFO] Training WGAN...")

    for epoch in range(opt.nepoch):
        for i in range(0, data.ntrain, opt.batch_size):
            for j in range(opt.critic_iter):
                batch_feature, batch_att, batch_label, z_rand = next_feed_dict(data, opt)
                _, discLoss, merged = sess.run([discTrain, discriminatorLoss, merged_all],
                                               feed_dict={input_res: batch_feature, input_att: batch_att,
                                                          input_label: batch_label, noise_z: z_rand})
                print("Discriminator loss is:" + str(discLoss))
                if j == 0:
                    summary_writer.add_summary(merged, k)

            batch_feature, batch_att, batch_label, z_rand = next_feed_dict(data, opt)
            _, genLoss, merged = sess.run([generatorTrain, generatorLoss, merged_all],
                                          feed_dict={input_res: batch_feature, input_att: batch_att,
                                                     input_label: batch_label, noise_z: z_rand})
            print("Generator loss is:" + str(genLoss))
            summary_writer.add_summary(merged, k)
            k = k + 1

        saver.save(sess, os.path.join(opt.modeldir, 'models_' + str(epoch) + '.ckpt'))
        print("Model saved")

    ##################### graph 2 definition ########################################################
with g2.as_default():
    ########## placeholderS ############################ data.unseenclasses, data.attribute,
    syn_att = tf.placeholder(tf.float32, [opt.syn_num, opt.attSize], name='input_attributes')
    noise_z1 = tf.placeholder(tf.float32, [opt.syn_num, opt.nz], name='noise')
    ########## model definition ##################################################

    noise1 = tf.concat([noise_z1, syn_att], axis=1)
    gen_res = generator(noise1, opt, isTrainable=False, reuse=False)

############ getting features from g2 graph ############################3
syn_res = np.empty((0, opt.resSize), np.float32)
syn_label = np.empty((0), np.float32)

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

    saver = tf.train.Saver(var_list=params)

    #for var in params:
      #  print(var.name + "\t")

    string = opt.modeldir + '/models_' + str(opt.nepoch - 1) + '.ckpt'
    ## print (string)

    try:
        saver.restore(sess, string)
    except:
        print("Previous weights not found of generator")
        sys.exit(0)

  #  print("Model loaded")

    saver = tf.train.Saver()

    for i in range(0, data.unseenclasses.shape[0]):
        iclass = data.unseenclasses[i]
        iclass_att = np.reshape(data.attribute[iclass], (1, opt.attSize))
        ## print (iclass_att.shape)
        ## print (iclass_att)
        batch_att = np.repeat(iclass_att, [opt.syn_num], axis=0)
        ## print (batch_att.shape)
        z_rand = np.random.normal(0, 1, [opt.syn_num, opt.nz]).astype(np.float32)

        syn_features = sess.run(gen_res, feed_dict={syn_att: batch_att, noise_z1: z_rand})
        syn_res = np.vstack((syn_res, syn_features))
        temp = np.repeat(iclass, [opt.syn_num], axis=0)
        ## print (temp.shape)
        syn_label = np.concatenate((syn_label, temp))

    ## print (syn_res.shape)
    ## print (syn_label.shape)
    np.savetxt('syn_res.txt', syn_res, delimiter=',')
    np.savetxt('syn_label.txt', syn_label, delimiter=',')
