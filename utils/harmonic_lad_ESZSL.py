import re
import numpy as np
import argparse

from scipy.io import loadmat
from sklearn.metrics import accuracy_score

# Parser
parser = argparse.ArgumentParser(description="Evaluate LAD in ZSL setting")
parser.add_argument("--split", type=int, default=0, help="ZSL split to be evaluated.")
parser.add_argument("--att_split", type=str, default="att_split", help="Name of the att_split file.")
parser.add_argument("--preds_seen_file", type=str, default="preds_seen_ESZSL_att_0.txt", help="Name of the preds file.")
parser.add_argument("--preds_unseen_file", type=str, default="preds_unseen_ESZSL_att_0.txt", help="Name of the preds file.")
parser.add_argument("--seen", action='store_true', default=False)

args = parser.parse_args()
ZSL_SPLIT = args.split
ATT_SPLIT = args.att_split
PREDS_SEEN_FILE = args.preds_seen_file
PREDS_UNSEEN_FILE = args.preds_unseen_file
SEEN = args.seen

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

# For GZSL Evaluation
"""
num_to_label[1] = 'Label_A_01'
...
num_to_label[n] = 'Label_K_n'
"""
num_to_label = dict()
for key, value in labels_dict.items():
    num_to_label[value] = key

# UNSEEN CLASSES SPLITS
if SEEN:

    preds = np.loadtxt(PREDS_SEEN_FILE)

    res101 = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/ResNet101.mat")
    att_splits = loadmat("/home/cristianopatricio/Documents/Datasets/xlsa17/data/LAD/"+ATT_SPLIT+".mat")

    labels = res101['labels']
    test_loc = 'test_seen_loc'

    Y_test_seen = labels[np.squeeze(att_splits[test_loc] - 1)]

    unique_seen_classes = np.unique(np.squeeze(Y_test_seen))
    seen_classes = [num_to_label[i] for i in unique_seen_classes]

    animals = unique_seen_classes[np.where(["A" in s for s in seen_classes])[0]]
    fruits = unique_seen_classes[np.where(["F" in s for s in seen_classes])[0]]
    vehicles = unique_seen_classes[np.where(["V" in s for s in seen_classes])[0]]
    electronics = unique_seen_classes[np.where(["E" in s for s in seen_classes])[0]]
    hairstyles = unique_seen_classes[np.where(["H" in s for s in seen_classes])[0]]

    Y_test_seen_animals = np.where(Y_test_seen == animals)[0]
    Y_test_seen_fruits = np.where(Y_test_seen == fruits)[0]
    Y_test_seen_vehicles = np.where(Y_test_seen == vehicles)[0]
    Y_test_seen_electronics = np.where(Y_test_seen == electronics)[0]
    Y_test_seen_hairstyles = np.where(Y_test_seen == hairstyles)[0]

    #Y_test_seen = encode_labels(Y_test_seen)


    print(f"################### RESULTS ###################")
    # print(f"Preds animals: {preds[Y_test_unseen_animals]}")
    # print(f"Ground truth animals: {np.squeeze(Y_test_unseen[Y_test_unseen_animals])}")
    animals_acc = accuracy_score(np.squeeze(Y_test_seen[Y_test_seen_animals]), preds[Y_test_seen_animals])
    print(
        f"[ANIMALS] \t\t ---> Accuracy score: {animals_acc * 100:.2f} %")

    # print(f"Preds fruits: {preds[Y_test_unseen_fruits]}")
    # print(f"Ground truth fruits: {np.squeeze(Y_test_unseen[Y_test_unseen_fruits])}")
    fruits_acc = accuracy_score(np.squeeze(Y_test_seen[Y_test_seen_fruits]), preds[Y_test_seen_fruits])
    print(
        f"[FRUITS] \t\t ---> Accuracy score: {fruits_acc * 100:.2f} %")

    # print(f"Preds vehicles: {preds[Y_test_unseen_vehicles]}")
    # print(f"Ground truth vehicles: {np.squeeze(Y_test_unseen[Y_test_unseen_vehicles])}")
    vehicles_acc = accuracy_score(np.squeeze(Y_test_seen[Y_test_seen_vehicles]), preds[Y_test_seen_vehicles])
    print(
        f"[VEHICLES] \t\t ---> Accuracy score: {vehicles_acc * 100:.2f} %")

    # print(f"Preds electronics: {preds[Y_test_unseen_electronics]}")
    # print(f"Ground truth electronics: {np.squeeze(Y_test_unseen[Y_test_unseen_electronics])}")
    electronics_acc = accuracy_score(np.squeeze(Y_test_seen[Y_test_seen_electronics]), preds[Y_test_seen_electronics])
    print(
        f"[ELECTRONICS] \t ---> Accuracy score: {electronics_acc * 100:.2f} %")

    # print(f"Preds hairstyles: {preds[Y_test_unseen_hairstyles]}")
    # print(f"Ground truth hairstyles: {np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles])}")
    hairstyles_acc = accuracy_score(np.squeeze(Y_test_seen[Y_test_seen_hairstyles]), preds[Y_test_seen_hairstyles])
    print(
        f"[HAIRSTYLES] \t ---> Accuracy score: {hairstyles_acc * 100:.2f} %")

    # Save to results_seen.txt
    file = open("results_seen.txt", "a")
    file.write(str(animals_acc) + "," + str(fruits_acc) + "," + str(vehicles_acc) + "," + str(electronics_acc) + "," + str(hairstyles_acc) + "\n")
    file.close()

else:
    unseen_classes = np.array([labels_dict[label] for label in zsl_splits[ZSL_SPLIT]])

    animals = unseen_classes[:10]
    fruits = unseen_classes[10:20]
    vehicles = unseen_classes[20:30]
    electronics = unseen_classes[30:40]
    hairstyles = unseen_classes[40:]

    preds = np.loadtxt(PREDS_UNSEEN_FILE)

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

    #Y_test_unseen = encode_labels(Y_test_unseen)

    print(f"################### RESULTS ###################")
    # print(f"Preds animals: {preds[Y_test_unseen_animals]}")
    # print(f"Ground truth animals: {np.squeeze(Y_test_unseen[Y_test_unseen_animals])}")
    animals_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_animals]), preds[Y_test_unseen_animals])
    print(
        f"[ANIMALS] \t\t ---> Accuracy score: {animals_acc * 100:.2f} %")

    # print(f"Preds fruits: {preds[Y_test_unseen_fruits]}")
    # print(f"Ground truth fruits: {np.squeeze(Y_test_unseen[Y_test_unseen_fruits])}")
    fruits_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_fruits]), preds[Y_test_unseen_fruits])
    print(
        f"[FRUITS] \t\t ---> Accuracy score: {fruits_acc * 100:.2f} %")

    # print(f"Preds vehicles: {preds[Y_test_unseen_vehicles]}")
    # print(f"Ground truth vehicles: {np.squeeze(Y_test_unseen[Y_test_unseen_vehicles])}")
    vehicles_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_vehicles]), preds[Y_test_unseen_vehicles])
    print(
        f"[VEHICLES] \t\t ---> Accuracy score: {vehicles_acc * 100:.2f} %")

    # print(f"Preds electronics: {preds[Y_test_unseen_electronics]}")
    # print(f"Ground truth electronics: {np.squeeze(Y_test_unseen[Y_test_unseen_electronics])}")
    electronics_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_electronics]), preds[Y_test_unseen_electronics])
    print(
        f"[ELECTRONICS] \t ---> Accuracy score: {electronics_acc * 100:.2f} %")

    # print(f"Preds hairstyles: {preds[Y_test_unseen_hairstyles]}")
    # print(f"Ground truth hairstyles: {np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles])}")
    hairstyles_acc = accuracy_score(np.squeeze(Y_test_unseen[Y_test_unseen_hairstyles]), preds[Y_test_unseen_hairstyles])
    print(
        f"[HAIRSTYLES] \t ---> Accuracy score: {hairstyles_acc * 100:.2f} %")

    # Save to results_unseen.txt
    file = open("results_unseen.txt", "a")
    file.write(str(animals_acc) + "," + str(fruits_acc) + "," + str(vehicles_acc) + "," + str(electronics_acc) + "," + str(hairstyles_acc) + "\n")
    file.close()

