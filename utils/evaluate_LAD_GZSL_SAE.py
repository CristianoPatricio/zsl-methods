import re
import numpy as np
import argparse

from scipy.io import loadmat
from sklearn.metrics import accuracy_score

# Parser
parser = argparse.ArgumentParser(description="Evaluate LAD in ZSL setting")
parser.add_argument("--split", type=int, default=0, help="ZSL split to be evaluated.")
parser.add_argument("--att_split", type=str, default="att_splits", help="Name of the att_split file.")
parser.add_argument("--preds", type=str, default="preds_SAE_GZSL.txt", help="Name of the preds file.")

args = parser.parse_args()
ZSL_SPLIT = args.split
ATT_SPLIT = args.att_split
PREDS = args.preds

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

# For GZSL Evaluation
"""
num_to_label[1] = 'Label_A_01'
...
num_to_label[n] = 'Label_K_n'
"""
num_to_label = dict()
for key, value in labels_dict.items():
    num_to_label[value] = key

preds = np.loadtxt(PREDS) + 1

res101 = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/ResNet101.mat")
att_splits = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/"+ATT_SPLIT+".mat")

labels = res101['labels']
test_seen_loc = 'test_seen_loc'
test_loc = 'test_unseen_loc'

Y_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)]
Y_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)]
Y_gzsl = np.concatenate((Y_test_unseen, Y_test_seen), axis=0)

##############################################################################
#   SEEN CLASSES
##############################################################################

unique_seen_classes = np.unique(np.squeeze(Y_test_seen))
seen_classes = [num_to_label[i] for i in unique_seen_classes]

animals = unique_seen_classes[np.where(["A" in s for s in seen_classes])[0]]
fruits = unique_seen_classes[np.where(["F" in s for s in seen_classes])[0]]
vehicles = unique_seen_classes[np.where(["V" in s for s in seen_classes])[0]]
electronics = unique_seen_classes[np.where(["E" in s for s in seen_classes])[0]]
hairstyles = unique_seen_classes[np.where(["H" in s for s in seen_classes])[0]]

Y_test_seen_animals = np.where(Y_gzsl == animals)[0]
Y_test_seen_fruits = np.where(Y_gzsl == fruits)[0]
Y_test_seen_vehicles = np.where(Y_gzsl == vehicles)[0]
Y_test_seen_electronics = np.where(Y_gzsl == electronics)[0]
Y_test_seen_hairstyles = np.where(Y_gzsl == hairstyles)[0]

#print(accuracy_score(Y_gzsl[np.where([Y_gzsl == y for y in Y_test_seen])[1]], preds[np.where([Y_gzsl == y for y in Y_test_seen])[1]]))  # acc seen
#print(accuracy_score(Y_gzsl[np.where([Y_gzsl == y for y in Y_test_unseen])[1]], preds[np.where([Y_gzsl == y for y in Y_test_unseen])[1]]))  # acc unseen

print(f"///////////////////// SEEN ////////////////////////")

animals_acc = accuracy_score(Y_gzsl[Y_test_seen_animals], preds[Y_test_seen_animals])
print(f"[ANIMALS] \t ---> Accuracy score: {animals_acc * 100:.2f} %")
fruits_acc = accuracy_score(Y_gzsl[Y_test_seen_fruits], preds[Y_test_seen_fruits])
print(f"[FRUITS] \t ---> Accuracy score: {fruits_acc * 100:.2f} %")
vehicles_acc = accuracy_score(Y_gzsl[Y_test_seen_vehicles], preds[Y_test_seen_vehicles])
print(f"[VEHICLES] \t ---> Accuracy score: {vehicles_acc * 100:.2f} %")
electronics_acc = accuracy_score(Y_gzsl[Y_test_seen_electronics], preds[Y_test_seen_electronics])
print(f"[ELECTRONICS] \t ---> Accuracy score: {electronics_acc * 100:.2f} %")
hairstyles_acc = accuracy_score(Y_gzsl[Y_test_seen_hairstyles], preds[Y_test_seen_hairstyles])
print(f"[HAIRSTYLES] \t ---> Accuracy score: {hairstyles_acc * 100:.2f} %")

# Save to results_seen
file = open("results_seen.txt", "a")
file.write(str(animals_acc) + "," + str(fruits_acc) + "," + str(vehicles_acc) + "," + str(electronics_acc) + "," + str(hairstyles_acc) + "\n")
file.close()

##############################################################################
#   UNSEEN CLASSES
##############################################################################

unseen_classes = np.array([labels_dict[label] for label in zsl_splits[ZSL_SPLIT]])

animals = unseen_classes[:10]
fruits = unseen_classes[10:20]
vehicles = unseen_classes[20:30]
electronics = unseen_classes[30:40]
hairstyles = unseen_classes[40:]

Y_test_unseen_animals = np.where(Y_gzsl == animals)[0]
Y_test_unseen_fruits = np.where(Y_gzsl == fruits)[0]
Y_test_unseen_vehicles = np.where(Y_gzsl == vehicles)[0]
Y_test_unseen_electronics = np.where(Y_gzsl == electronics)[0]
Y_test_unseen_hairstyles = np.where(Y_gzsl == hairstyles)[0]

print(f"//////////////////// UNSEEN ///////////////////////")

animals_acc = accuracy_score(Y_gzsl[Y_test_unseen_animals], preds[Y_test_unseen_animals])
print(f"[ANIMALS] \t ---> Accuracy score: {animals_acc * 100:.2f} %")
fruits_acc = accuracy_score(Y_gzsl[Y_test_unseen_fruits], preds[Y_test_unseen_fruits])
print(f"[FRUITS] \t ---> Accuracy score: {fruits_acc * 100:.2f} %")
vehicles_acc = accuracy_score(Y_gzsl[Y_test_unseen_vehicles], preds[Y_test_unseen_vehicles])
print(f"[VEHICLES] \t ---> Accuracy score: {vehicles_acc * 100:.2f} %")
electronics_acc = accuracy_score(Y_gzsl[Y_test_unseen_electronics], preds[Y_test_unseen_electronics])
print(f"[ELECTRONICS] \t ---> Accuracy score: {electronics_acc * 100:.2f} %")
hairstyles_acc = accuracy_score(Y_gzsl[Y_test_unseen_hairstyles], preds[Y_test_unseen_hairstyles])
print(f"[HAIRSTYLES] \t ---> Accuracy score: {hairstyles_acc * 100:.2f} %")

# Save to results_seen
file = open("results_unseen.txt", "a")
file.write(str(animals_acc) + "," + str(fruits_acc) + "," + str(vehicles_acc) + "," + str(electronics_acc) + "," + str(hairstyles_acc) + "\n")
file.close()