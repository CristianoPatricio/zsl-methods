"""
NOTE:
1. This code was originally created by the author of the GitHub repository https://github.com/akku1506/Feature-Generating-Networks-for-ZSL.
2. All changes were made by the author of the current repository in order to adapt the output to the standardized nomenclature of the rest of
the implemented methods.
"""

import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import argparse
from classifier import CLASSIFIER
import util
import random
import sys

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA1', help='AWA1')
parser.add_argument('--dataroot', default='/home/cristianopatricio/Documents/Datasets/xlsa17/data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--train', default=True, help='enables training')
parser.add_argument('--test', default=True, help='enable testing mode')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--logdir', default='/home/cristianopatricio/PycharmProjects/zsl-methods/f-CLSWGAN/logs_classifier', help='folder to output and help print losses')
parser.add_argument('--modeldir', default='/home/cristianopatricio/PycharmProjects/zsl-methods/f-CLSWGAN/models_classifier', help='folder to output  model checkpoints')
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
parser.add_argument('--split_no', type=str, default='', help="Split number in case of LAD dataset.")

opt = parser.parse_args()

if not os.path.exists(opt.logdir):
    os.makedirs(opt.logdir)
if not os.path.exists(opt.modeldir):
    os.makedirs(opt.modeldir)

random.seed(opt.manualSeed)
tf.set_random_seed(opt.manualSeed)

if opt.train == False and opt.test == False:
    print("Program terminated as no train or test option is set true")
    sys.exit(0)

######################################################################
#   LOAD DATA
######################################################################
data = util.DATA_LOADER(opt)


######################################################################
#   TRAINING
######################################################################

train_cls = CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses),
                       data.seenclasses.shape[0], opt.resSize, opt.logdir, opt.modeldir, opt.lr, opt.beta1, opt.nepoch,
                       opt.batch_size, '')
if opt.train:
    print(f"[INFO]: Training classifier...")
    train_cls.train()

if opt.test:
    print(f"[INFO]: Testing results:")
    acc = train_cls.val(data.test_seen_feature, data.test_seen_label, data.seenclasses)
    print(f"Test Accuracy is: {str(acc)}")
    acc = train_cls.val(data.train_feature, data.train_label, data.seenclasses)
    print(f"Train Accuracy is: {str(acc)}")

