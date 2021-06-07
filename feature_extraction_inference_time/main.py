import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.applications import Xception, VGG16, VGG19, ResNet50, ResNet101, ResNet50V2, ResNet101V2, \
    InceptionV3, MobileNet, MobileNetV2, DenseNet201, NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
import time
import csv
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def write_to_csv(avg_time, std_time, dim, size):
    """
    Writes to a CSV file the content of the dictionary
    :param avg_time: dict, containing avg time
    :param std_time: dict, containing std time
    :param dim: dict, containing the dimension of the feature vector
    :param size: dict, containing the size of the model
    :return:
    """

    with open('results_models.csv', mode='w') as data_file:
        header = ['Model', f'Execution Time (AVG \u00B1 STD) (ms)', 'Features Dimension', 'Size (MB)']
        data_writer = csv.DictWriter(data_file, fieldnames=header)

        data_writer.writeheader()
        for name in dict(sorted(dim.items(), key=lambda item: item[1])).keys():
            data_writer.writerow({'Model': name, f'Execution Time (AVG \u00B1 STD) (ms)': f'{avg_time[name]:.2f} \u00B1 {std_time[name]:.2f}', 'Features Dimension': dim[name], 'Size (MB)': size[name]})


def get_feature_vector(img_path, model, name):
    """
    Extracts the features vector of the image for a given model
    :param img_path: string value, image directory path
    :param model: Model, keras model
    :return: features -- array of shape (None,), containing the feature vector of the image
             elapsed_time -- real value, elapsed time (in seconds) during feature extraction
    """

    # Load image
    if name == "NASNetLarge":
        img = image.load_img(img_path, target_size=(331, 331))
    else:
        img = image.load_img(img_path, target_size=(224, 224))


    # Model warmup phase
    for i in range(20):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = model.predict(x)

    tic = time.time()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract features
    features = model.predict(x)
    toc = time.time()

    # Compute elapsed time (ms)
    elapsed_time = (toc - tic)*1000

    return features, elapsed_time


if __name__ == '__main__':

    # Define models
    print(f"[INFO] Loading models... ", end='')
    models = [
        Xception(weights='imagenet', include_top=False, pooling='avg'),
        VGG16(weights='imagenet', include_top=False, pooling='avg'),
        VGG19(weights='imagenet', include_top=False, pooling='avg'),
        ResNet50(weights='imagenet', include_top=False, pooling='avg'),
        ResNet101(weights='imagenet', include_top=False, pooling='avg'),
        ResNet50V2(weights='imagenet', include_top=False, pooling='avg'),
        ResNet101V2(weights='imagenet', include_top=False, pooling='avg'),
        InceptionV3(weights='imagenet', include_top=False, pooling='avg'),
        MobileNet(weights='imagenet', include_top=False, pooling='avg'),
        MobileNetV2(weights='imagenet', include_top=False, pooling='avg'),
        DenseNet201(weights='imagenet', include_top=False, pooling='avg'),
        NASNetMobile(weights='imagenet', include_top=False, pooling='avg'),
        NASNetLarge(weights='imagenet', include_top=False, pooling='avg'),
        EfficientNetB0(weights='imagenet', include_top=False, pooling='avg'),
        EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
    ]
    print(f"SUCCESS!")

    # Select 5 random images
    images = [
        './images/img2.jpg',
        './images/img1.jpg',
        './images/img0.jpg',
        './images/img4.jpg',
        './images/img3.jpg']

    # Define Models Size
    model_size = {
        "Xception": 88,
        "VGG16": 528,
        "VGG19": 549,
        "ResNet50": 98,
        "ResNet101": 171,
        "ResNet50V2": 98,
        "ResNet101V2": 171,
        "InceptionV3": 92,
        "MobileNet": 16,
        "MobileNetV2": 14,
        "DenseNet201": 80,
        "NASNetMobile": 23,
        "NASNetLarge": 343,
        "EfficientNetB0": 29,
        "EfficientNetB7": 256
    }

    avg_time_dict = dict()
    std_time_dict = dict()
    dim_dict = dict()
    # Loop through models
    for model, name in zip(models, model_size.keys()):
        elap_times = []
        feat_dim = 0
        print(f"[INFO] Evaluating {name}... ", end='')
        # Loop through images to get metrics
        for img in images:
            feature_np, elap_time = get_feature_vector(img, model)
            elap_times.append(elap_time)
            feat_dim = feature_np.shape[1]

        # Compute SVG and STD
        avg_time_dict[name] = np.mean(elap_times[1:])
        std_time_dict[name] = np.std(elap_times[1:])
        dim_dict[name] = feat_dim

        print(f"SUCCESS!")

    # Write to CSV
    write_to_csv(avg_time_dict, std_time_dict, dim_dict, model_size)
    print(f"[INFO] Results saved at results_models.csv")
