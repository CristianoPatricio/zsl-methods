import re
import numpy as np
import argparse

from scipy.io import loadmat
from sklearn.metrics import accuracy_score

# Parser
parser = argparse.ArgumentParser(description="Evaluate LAD in ZSL setting")
parser.add_argument("--split", type=int, default=0, help="ZSL split to be evaluated.")
parser.add_argument("--att_split", type=str, default="att_split", help="Name of the att_split file.")
parser.add_argument("--preds_file", type=str, default="preds_att_split_0.txt", help="Name of the preds file.")

args = parser.parse_args()
ZSL_SPLIT = args.split
ATT_SPLIT = args.att_split
PREDS_FILE = args.preds_file

def encode_labels(Y):
    i = 0
    for labels in np.unique(Y):
        Y[Y == labels] = i
        i += 1

    return Y


# Get Classes Splits
zsl_splits = []
with open("/home/cristianopatricio/Documents/Datasets/LAD/LAD_annotations/split_zsl.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        line_preprocess = line[15:].replace(", ", ",")
        line = re.split(",", line_preprocess.strip())
        zsl_splits.append(line)

zsl_splits = np.asarray(zsl_splits)

# Labels Dict
labels_dict = dict()
with open("/home/cristianopatricio/Documents/Datasets/LAD/LAD_annotations/label_list.txt", "r") as file:
    lines = file.readlines()
    for idx, line in enumerate(lines):
        line = re.split(",", line)
        labels_dict[line[0].strip()] = idx + 1

unseen_classes = np.array([labels_dict[label] for label in zsl_splits[ZSL_SPLIT]])

animals = unseen_classes[:10]
fruits = unseen_classes[10:20]
vehicles = unseen_classes[20:30]
electronics = unseen_classes[30:40]
hairstyles = unseen_classes[40:]

preds = np.loadtxt(PREDS_FILE)

res101 = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/ResNet101.mat")
att_splits = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/"+ATT_SPLIT+".mat")

labels = res101['labels']
test_loc = 'test_unseen_loc'

Y_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)]

Y_test_unseen_animals = np.where(Y_test_unseen == animals)[0]
Y_test_unseen_fruits = np.where(Y_test_unseen == fruits)[0]
Y_test_unseen_vehicles = np.where(Y_test_unseen == vehicles)[0]
Y_test_unseen_electronics = np.where(Y_test_unseen == electronics)[0]
Y_test_unseen_hairstyles = np.where(Y_test_unseen == hairstyles)[0]

Y_test_unseen = encode_labels(Y_test_unseen)

print(f"################### RESULTS ###################")
# print(f"Preds animals: {preds[Y_test_unseen_animals]}")
# print(f"Ground truth animals: {np.squeeze(Y_test_unseen[Y_test_unseen_animals])}")
animals_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_animals]), preds[Y_test_unseen_animals])
print(
    f"[ANIMALS] \t\t ---> Accuracy score: {accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_animals]), preds[Y_test_unseen_animals]) * 100:.2f} %")

# print(f"Preds fruits: {preds[Y_test_unseen_fruits]}")
# print(f"Ground truth fruits: {np.squeeze(Y_test_unseen[Y_test_unseen_fruits])}")
fruits_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_fruits]), preds[Y_test_unseen_fruits])
print(
    f"[FRUITS] \t\t ---> Accuracy score: {accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_fruits]), preds[Y_test_unseen_fruits]) * 100:.2f} %")

# print(f"Preds vehicles: {preds[Y_test_unseen_vehicles]}")
# print(f"Ground truth vehicles: {np.squeeze(Y_test_unseen[Y_test_unseen_vehicles])}")
vehicles_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_vehicles]), preds[Y_test_unseen_vehicles])
print(
    f"[VEHICLES] \t\t ---> Accuracy score: {accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_vehicles]), preds[Y_test_unseen_vehicles]) * 100:.2f} %")

# print(f"Preds electronics: {preds[Y_test_unseen_electronics]}")
# print(f"Ground truth electronics: {np.squeeze(Y_test_unseen[Y_test_unseen_electronics])}")
electronics_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_electronics]), preds[Y_test_unseen_electronics])
print(
    f"[ELECTRONICS] \t ---> Accuracy score: {accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_electronics]), preds[Y_test_unseen_electronics]) * 100:.2f} %")

# print(f"Preds hairstyles: {preds[Y_test_unseen_hairstyles]}")
# print(f"Ground truth hairstyles: {np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles])}")
hairstyles_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles]), preds[Y_test_unseen_hairstyles])
print(
    f"[HAIRSTYLES] \t ---> Accuracy score: {accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles]), preds[Y_test_unseen_hairstyles]) * 100:.2f} %")

file = open("results_ZSL.txt", "a")
file.write(str(animals_acc) + "," + str(fruits_acc) + "," + str(vehicles_acc) + "," + str(electronics_acc) + "," + str(hairstyles_acc) + "\n")
file.close()