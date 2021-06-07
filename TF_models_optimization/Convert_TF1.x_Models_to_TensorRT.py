"""
Python script to convert TF 1.x models to TensorRT
"""
import os
import numpy as np
import time

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.platform import gfile


###########################################################################
#   AUXILIARY FUNCTIONS
###########################################################################


def convert_to_frozen_graph(model_path, meta_graph_name, output_node_name):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + "/" + meta_graph_name + ".meta")
        saver.restore(sess, model_path)

        outputs = [str(output_node_name)]

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names=outputs
        )

    with gfile.FastGFile("frozen_" + meta_graph_name + ".pb", "wb") as f:
        f.write(frozen_graph.SerializeToString())
    print("Frozen model is successfully stored!")

    return frozen_graph, outputs


def convert_to_trt_graph(frozen_graph, outputs, frozen_model_name, precision_mode):
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=outputs,
        max_batch_size=2,
        max_workspace_size_bytes=2 * (10 ** 9),  # ~ 2GB
        precision_mode=precision_mode
    )

    with gfile.FastGFile(frozen_model_name + ".pb", "wb") as f:
        f.write(trt_graph.SerializeToString())
    print("TensorRT model is successfully stored!")

    all_nodes = len([1 for n in frozen_graph.node])
    print(f"No. of all_nodes in frozen graph: {all_nodes}")

    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print(f"No. of trt_engine_nodes in TensorRT graph: {trt_engine_nodes}")

    all_nodes = len([1 for n in trt_graph.node])
    print(f"No. of all_nodes in TensorRT graph: {all_nodes}")


def read_pb_graph(model):
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def predict_and_benchmark_throughput(trt_model, input_node_name, output_node_name, batch_size=1, N_warmup_run=50,
                                     N_run=1000):
    all_preds = []
    elapsed_time = []
    test_att = np.random.rand(1, 85)  # shape (1, 85)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            trt_graph = read_pb_graph(trt_model)

            tf.import_graph_def(trt_graph, name='')
            input = sess.graph.get_tensor_by_name(str(input_node_name) + ':0')
            output = sess.graph.get_tensor_by_name(str(output_node_name) + ':0')

            for i in range(N_warmup_run):
                preds = sess.run(output, feed_dict={input: test_att})

            for i in range(N_run):
                tic = time.time()
                preds = sess.run(output, feed_dict={input: test_att})
                toc = time.time()

                elapsed_time = np.append(elapsed_time, toc - tic)

                all_preds.append(preds)

                if i % 50 == 0:
                    print(f"Steps {i}-{i + 50} average: {(elapsed_time[-50:].mean()) * 1000:.1f} ms")

            print(f"Throughput: {N_run * batch_size / elapsed_time.sum():.0f} images/s")

    return all_preds


if __name__ == '__main__':
    # Convert saved model to frozen graph
    frozen_graph, outputs = convert_to_frozen_graph(model_path="model_DEM", meta_graph_name="model_dem",
                                                    output_node_name="op_to_restore")

    # Convert frozen model to TensorRT
    convert_to_trt_graph(frozen_graph, outputs, frozen_model_name="frozen_model_dem", precision_mode="FP32")

    # Make predictions and benchmark
    predict_and_benchmark_throughput(trt_model="TensorRT_model.pb", input_node_name="W1",
                                     output_node_name="op_to_restore", batch_size=1, N_warmup_run=50,
                                     N_run=1000)
    