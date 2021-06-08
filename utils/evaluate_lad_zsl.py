import numpy as np
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="results_ZSL.txt", help="Name of the file containing the results.")
args = parser.parse_args()

arr = []
with open(args.filename, "r") as file:
    lines = file.readlines()
    for line in lines:
        values = re.split(",", line[:-1])   # until \n
        arr.append(values)

final_arr = np.array(arr).astype(np.float)
#print(np.mean(final_arr, axis=0))
final_avg = np.mean(np.mean(final_arr, axis=0))
print(f"ZSL Accuracy: {final_avg * 100:.2f}%")