"""
Python script to convert TF 2.x models to TensorRT
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input


#############################################################################
#   AUXILIARY FUNCTIONS
#############################################################################

def batch_input(batch_size=8):
    batched_input = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batched_input = tf.constant(batched_input)
    return batched_input


batched_input = batch_input(batch_size=32)


def load_tf_saved_model(input_saved_model_dir):
    print(f"Loading saved model {input_saved_model_dir}...")
    saved_model_loaded = tf.saved_model.load(input_saved_model_dir, tags=[tag_constants.SERVING])
    return saved_model_loaded


def predict_and_benchmark_throughput(batched_input, infer, N_warmup_run=50, N_run=1000):
    elapsed_time = []
    all_preds = []
    batch_size = batched_input.shape[0]

    for i in range(N_warmup_run):
        preds = infer(batched_input)

    for i in range(N_run):
        tic = time.time()
        preds = infer(batched_input)
        toc = time.time()

        elapsed_time = np.append(elapsed_time, toc - tic)

        all_preds.append(preds)

        if i % 50 == 0:
            print(f"Steps {i}-{i + 50} average: {(elapsed_time[-50:].mean()) * 1000:.1f} ms")

    print(f"Throughput: {N_run * batch_size / elapsed_time.sum():.0f} images/s")
    return all_preds


def show_predictions(model):
    # img_path = '/content/images/img0.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    x = np.random.randint(255, size=(224, 244, 3))  # image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = tf.constant(x, dtype=tf.float32)

    preds = model(x)
    print(f"Predictions shape: {preds['global_average_pooling2d'].numpy().shape}")

    return preds


def convert_to_trt_graph_and_save(precision_mode='float32',
                                  input_saved_model_dir='res101_saved_model',
                                  calibration_data=batched_input):
    if precision_mode == 'float32':
        precision_mode = trt.TrtPrecisionMode.FP32
        converter_save_suffix = '_TFTRT_FP32'

    if precision_mode == 'float16':
        precision_mode = trt.TrtPrecisionMode.FP16
        converter_save_suffix = '_TFTRT_FP16'

    if precision_mode == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
        converter_save_suffix = '_TFTRT_INT8'

    output_saved_model_dir = input_saved_model_dir + converter_save_suffix

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=4000000000  # ~ 4GB
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_saved_model_dir,
        conversion_params=conversion_params
    )

    print(f"Converting {input_saved_model_dir} to TF-TRT graph precision mode {precision_mode}...")

    if precision_mode == trt.TrtPrecisionMode.INT8:
        def calibration_input_fn():
            yield (calibration_data, )

        converter.convert(calibration_input_fn=calibration_input_fn)

    else:
        converter.convert()

    print(f"Saving converted model to {output_saved_model_dir}")
    converter.save(output_saved_model_dir)
    print("Complete!")


if __name__ == '__main__':
    # Load Keras model
    model = ResNet101(weights='imagenet', include_top=False, pooling='avg')

    # Save the entire model as a TensorFlow SavedModel.
    tf.saved_model.save(model, 'res101_saved_model')

    # Convert to TFTRT FP32
    convert_to_trt_graph_and_save(precision_mode='float32', input_saved_model_dir='res101_saved_model')

    # Load TFTRT model
    saved_model_loaded = load_tf_saved_model(input_saved_model_dir='res101_saved_model_TFTRT_FP32')

    # Make predictions and benchmark
    infer = saved_model_loaded.signatures['serving_default']
    show_predictions(infer)
    predict_and_benchmark_throughput(batched_input=batched_input, infer=infer, N_warmup_run=50, N_run=1000)
