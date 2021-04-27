import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing import image
import numpy as np
import glob
import time
import argparse

from keras.applications.mobilenet import preprocess_input as preprocess_input_mobilenet
from keras.applications.mobilenet import MobileNet

from keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenetv2
from keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient



parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--model', type=str, default='ResNet101',
                    help='Name of the Model')
parser.add_argument('--dataset_path', type=str, default='W:\ZSL\Animals_with_Attributes2\JPEGImages\*\*.jpg',
                    help='Path containing the images')

if __name__ == "__main__":


    args = parser.parse_args()

    print("[INFO]: Loading pre-trained model...")

    if args.model == 'ResNet101':
        preprocess_input = preprocess_input_resnet
        model = ResNet101(weights='imagenet', include_top=False, pooling='avg')
    elif args.model == 'MobileNet':
        preprocess_input = preprocess_input_mobilenet
        model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
    elif args.model == 'MobileNetV2':
        preprocess_input = preprocess_input_mobilenetv2
        model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    elif args.model == 'Xception':
        preprocess_input = preprocess_input_xception
        model = Xception(weights='imagenet', include_top=False, pooling='avg')
    elif args.model == 'EfficientNetB7':
        preprocess_input = preprocess_input_efficient
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')




# List to save extracted features
features = []

# Images path directory
img_path = 'W:\ZSL\Animals_with_Attributes2\JPEGImages\*\*.jpg'

# Get directories under the directory 'images'
dirs = sorted(glob.glob(img_path))

tic = time.time()
for dir in dirs:

    img = image.load_img(dir, target_size=(224, 224))
    img_data = image.img_to_array(img)
    #img_data = img_data[:, :, ::-1]
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    mnet_feature = model.predict(img_data)
    mnet_feature_np = np.array(mnet_feature)
    features.append(mnet_feature_np.flatten())

mobilenet_feature_list = np.array(features)

filename = args.model + "-AWA2-features.npy"
with open(filename, "wb") as file:
    np.save(file, mobilenet_feature_list)
print(f"[INFO]: Extracted features saved at {filename}")
toc = time.time()

# Print elapsed time
print(f"[INFO]: Process completed in {((toc-tic)/60):.2f} minutes")
