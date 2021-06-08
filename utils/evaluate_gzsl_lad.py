import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename_seen", type=str, default="results_seen.txt")
parser.add_argument("--filename_unseen", type=str, default="results_unseen.txt")
args = parser.parse_args()

arr = []
with open(args.filename_seen, "r") as file:
    lines = file.readlines()
    for line in lines:
        values = re.split(",", line[:-1])   # until \n
        arr.append(values)

final_arr = np.array(arr).astype(np.float)
#print(f"Per class acc seen: {np.mean(final_arr, axis=0)}")
acc_seen = np.mean(np.mean(final_arr, axis=0))

arr = []
with open(args.filename_unseen, "r") as file:
    lines = file.readlines()
    for line in lines:
        values = re.split(",", line[:-1])   # until \n
        arr.append(values)

final_arr = np.array(arr).astype(np.float)
#print(f"Per class acc unseen: {np.mean(final_arr, axis=0)}")
acc_unseen = np.mean(np.mean(final_arr, axis=0))

H = 2*acc_seen*acc_unseen / (acc_seen + acc_unseen)

print(f"Unseen={acc_unseen*100:.2f}%, Seen={acc_seen*100:.2f}%, Harmonic={H*100:.2f}%")

