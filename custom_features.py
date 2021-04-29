"""
@author Cristiano Patrício
@email cristiano.patricio@ubi.pt
"""

from scipy.io import loadmat, savemat
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Custom Features Maker")
parser.add_argument("--dataset", type=str, default="AWA2", help="{AWA1, AWA2, CUB, SUN, APY}.")
parser.add_argument("--dataroot", type=str, default="/home/cristianopatricio/Documents/Datasets/xlsa17/data/",
                    help="Path to dataset.")
parser.add_argument("--features", type=str, default="MobileNet-AWA2-features",
                    help="Name of the features file (without the extension).")
parser.add_argument("--features_path", type=str, default="/home/cristianopatricio/Documents/Datasets/Custom_Features/",
                    help="Path to features file.")


def create_mat_file(args):
    """
    Loads required data and creates .mat file.
    :param args: args.{dataset, dataroot, features, features_path}
    :return: 0
    """
    print(f"[INFO]: Loading data...")

    # Load default .mat file
    try:
        matcontent = loadmat(os.path.dirname(args.dataroot) + "/" + args.dataset + "/res101.mat")
    except Exception as e:
        sys.exit(f"[ERROR]: An error occurred when trying to open mat file: {e}.")

    image_files = matcontent['image_files']
    labels = matcontent['labels']

    # Load features file
    try:
        features = np.load(os.path.dirname(args.features_path) + "/" + args.dataset + "/" + args.features + ".npy").T
    except Exception as e:
        sys.exit(f"[ERROR]: An error occurred when trying to open features file: {e}.")

    new_dict = {
        "image_files": image_files,
        "features": features,
        "labels": labels
    }

    pos = args.features.find("-")
    filename = os.path.dirname(args.dataroot) + "/" + args.dataset + "/" + f"{args.features[:pos]}.mat"

    # Save new_dict into a .mat file
    try:
        savemat(filename, new_dict)
        print(f"[INFO]: Successfully created .mat file at {filename}")
    except Exception as e:
        sys.exit(f"[ERROR]: An error occurred when trying to save mat file: {e}.")


if __name__ == '__main__':
    args = parser.parse_args()
    create_mat_file(args)
