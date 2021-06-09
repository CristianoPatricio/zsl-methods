from __future__ import print_function
import os, os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import sys
import random
import argparse
import util
import re
import classifier2

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA1', help='AWA1')
parser.add_argument('--dataroot', default='/home/cristiano.patricio/datasets/xlsa17/data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--split_no', type=str, default='', help="Split number in case of LAD dataset.")

opt = parser.parse_args()

############################################################
#   LOAD DATA
############################################################
data = util.DATA_LOADER(opt)

# Syn features
syn_res = []
with open(os.path.join(os.getcwd(), "syn_res"+str(opt.split_no)+".txt"), "r") as f:
    lines = f.readlines()
    for l in lines:
        features = []
        values = re.split(",", l)
        for i in range(0, len(values)):
            value = values[i]
            features.append(float(value))

        syn_res.append(features)

syn_res = np.asarray(syn_res) # Shape (3000, 2048)

# Labels
syn_label = []
with open(os.path.join(os.getcwd(), "syn_label"+str(opt.split_no)+".txt"), "r") as f:
    lines = f.readlines()
    for l in lines:
        label = int(float(l.rstrip().lstrip()))
        syn_label.append(label)

syn_label = np.asarray(syn_label) # Shape (3000,) -> 300 syn feats per unseen class

######################################################################
#   TESTING
######################################################################

print(f"[INFO] Evaluating on {opt.dataset}...")

train_X = np.concatenate((data.train_feature, syn_res), axis=0)
train_Y = np.concatenate((data.train_label, syn_label), axis=0)
nclass = opt.nclass_all
train_gzsl = classifier2.CLASSIFICATION2(opt.dataset, train_X, train_Y, data, nclass, os.path.join(os.getcwd(), 'logs_gzsl_classifier'),
                                        os.path.join(os.getcwd(), 'models_gzsl_classifier'), opt.split_no, 0.001, 0.5, 25, opt.syn_num, True)

tf.reset_default_graph()

train_zsl = classifier2.CLASSIFICATION2(opt.dataset, syn_res, util.map_label(syn_label, data.unseenclasses), data,
                                        data.unseenclasses.shape[0], os.path.join(os.getcwd(), 'logs_zsl_classifier'), os.path.join(os.getcwd(), 'models_zsl_classifier'), opt.split_no,
                                        0.001, 0.5, 25, opt.syn_num, False)


print(f"[ZSL] Top-1 Accuracy (%): {train_zsl.acc:.2f} %")
print(f"[GZSL] Accuracy (%) - Seen: {train_gzsl.acc_seen:.2f} %, Unseen: {train_gzsl.acc_unseen:.2f} %, Harmonic={train_gzsl.H:.2f} %")

