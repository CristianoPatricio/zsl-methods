import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing import image
import numpy as np
import glob
import time
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2

# Load pre-trained MobileNet V2 Model
print("[INFO]: Loading pre-trained model...")

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# List to save extracted features
mobilenet_feature_list = []

# Images path directory
img_path = '/home/cristiano.patricio/datasets/AWA2/images/*.jpg'

# Get directories under the directory 'images'
dirs = sorted(glob.glob(img_path))

tic = time.time()
for dir in dirs:

    img = image.load_img(dir, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    mnet_feature = model.predict(img_data)
    mnet_feature_np = np.array(mnet_feature)

    mobilenet_feature_list.append(mnet_feature_np.flatten())

mobilenet_feature_list = np.array(mobilenet_feature_list)

filename = "/home/cristiano.patricio/MobileNetV2-AWA2-features.npy"
with open(filename, "wb") as file:
    np.save(file, mobilenet_feature_list)
print(f"[INFO]: Extracted features saved at {filename}")
toc = time.time()

# Print elapsed time
print(f"[INFO]: Process completed in {((toc-tic)/60):.2f} minutes")
