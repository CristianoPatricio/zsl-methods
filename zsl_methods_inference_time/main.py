import os

"""
Level | Level for Humans | Level Description                  
 -------|------------------|------------------------------------ 
  0     | DEBUG            | [Default] Print all messages       
  1     | INFO             | Filter out INFO messages           
  2     | WARNING          | Filter out INFO & WARNING messages 
  3     | ERROR            | Filter out all messages   
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import argparse
from scipy.spatial import distance
import time
import sys
from keras.models import load_model
import csv
import tensorflow.compat.v1 as tf

tf.get_logger().setLevel('ERROR')

tf.disable_v2_behavior()


# parser = argparse.ArgumentParser(description='Eval')
# parser.add_argument('--feat_dim', type=str, default='2048',
#                    help='Features dimension')
# parser.add_argument('--model', type=str, default='SAE',
#                    help='Name of the model')

def write_to_csv(avg_time, std_time, methods, features_dim):
    """
    Writes to a CSV file the content of the dictionary
    :param avg_time: dict, containing avg time
    :param std_time: dict, containing std time
    :param dim: dict, containing the dimension of the feature vector
    :param size: dict, containing the size of the model
    :return:
    """

    with open('results_methods.csv', mode='w') as data_file:
        header = ['Method', '512', '1024', '1056', '1280', '1920', '2048', '2560', '4032']
        data_writer = csv.DictWriter(data_file, fieldnames=header)

        data_writer.writeheader()
        for i in range(6):
            # 512
            avg_512 = f"{'-'}" if avg_time[features_dim[0]][i] == '-' else f"{float(avg_time[features_dim[0]][i]):.3f}"
            std_512 = f"{'-'}" if std_time[features_dim[0]][i] == '-' else f"{float(std_time[features_dim[0]][i]):.3f}"
            # 1024
            avg_1024 = f"{'-'}" if avg_time[features_dim[1]][i] == '-' else f"{float(avg_time[features_dim[1]][i]):.3f}"
            std_1024 = f"{'-'}" if std_time[features_dim[1]][i] == '-' else f"{float(std_time[features_dim[1]][i]):.3f}"
            # 1056
            avg_1056 = f"{'-'}" if avg_time[features_dim[2]][i] == '-' else f"{float(avg_time[features_dim[2]][i]):.3f}"
            std_1056 = f"{'-'}" if std_time[features_dim[2]][i] == '-' else f"{float(std_time[features_dim[2]][i]):.3f}"
            # 1280
            avg_1280 = f"{'-'}" if avg_time[features_dim[3]][i] == '-' else f"{float(avg_time[features_dim[3]][i]):.3f}"
            std_1280 = f"{'-'}" if std_time[features_dim[3]][i] == '-' else f"{float(std_time[features_dim[3]][i]):.3f}"
            # 1920
            avg_1920 = f"{'-'}" if avg_time[features_dim[4]][i] == '-' else f"{float(avg_time[features_dim[4]][i]):.3f}"
            std_1920 = f"{'-'}" if std_time[features_dim[4]][i] == '-' else f"{float(std_time[features_dim[4]][i]):.3f}"
            # 2048
            avg_2048 = f"{'-'}" if avg_time[features_dim[5]][i] == '-' else f"{float(avg_time[features_dim[5]][i]):.3f}"
            std_2048 = f"{'-'}" if std_time[features_dim[5]][i] == '-' else f"{float(std_time[features_dim[5]][i]):.3f}"
            # 2560
            avg_2560 = f"{'-'}" if avg_time[features_dim[6]][i] == '-' else f"{float(avg_time[features_dim[6]][i]):.3f}"
            std_2560 = f"{'-'}" if std_time[features_dim[6]][i] == '-' else f"{float(std_time[features_dim[6]][i]):.3f}"
            # 4032
            avg_4032 = f"{'-'}" if avg_time[features_dim[7]][i] == '-' else f"{float(avg_time[features_dim[7]][i]):.3f}"
            std_4032 = f"{'-'}" if std_time[features_dim[7]][i] == '-' else f"{float(std_time[features_dim[7]][i]):.3f}"

            # Write line
            data_writer.writerow({'Method': methods[i],
                                  '512': f"{avg_512} \u00B1 {std_512}",
                                  '1024': f"{avg_1024} \u00B1 {std_1024}",
                                  '1056': f"{avg_1056} \u00B1 {std_1056}",
                                  '1280': f"{avg_1280} \u00B1 {std_1280}",
                                  '1920': f"{avg_1920} \u00B1 {std_1920}",
                                  '2048': f"{avg_2048} \u00B1 {std_2048}",
                                  '2560': f"{avg_2560} \u00B1 {std_2560}",
                                  '4032': f"{avg_4032} \u00B1 {std_4032}"})


class Eval():

    def __init__(self, args):
        self.feat_dim = int(args)

    def SAE(self):
        """
        Simulates SAE method at inference stage
        :return: elapsed time, in milliseconds
        """
        np.random.seed(10)  # Seed the generator

        # The dimension of the attributes is set to 85 and the number of classes is set to 1000
        W = np.random.rand(85, self.feat_dim)  # shape (85, n_dim_feat)
        img_feature_vector = np.random.rand(1, self.feat_dim)  # shape (1, n_dim_feat)
        x_test = np.random.rand(1000, 85)  # shape (1000, 85)
        x_labels = np.random.randint(0, 999, size=(1000, 1))  # shape (1000, 1)

        tic = time.time()
        # Calculate predictions
        s_pred = np.dot(img_feature_vector, W.T)  # shape (1, 85)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, x_test, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = x_labels[[i for i in idx_min]]
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time

    def ESZSL(self):
        """
        Simulates ESZSL method at inference stage
        :return: elapsed time, in milliseconds
        """
        np.random.seed(10)

        # The dimension of the attributes is set to 85 and the number of classes is set to 1000
        W = np.random.rand(self.feat_dim, 85)  # shape (n_dim_feat, 85)
        img_feature_vector = np.random.rand(1, self.feat_dim)  # shape (1, n_dim_feat)
        x_test = np.random.rand(85, 1000)  # shape (85, 1000)

        tic = time.time()
        # Compute predictions
        output = np.matmul(np.matmul(img_feature_vector, W), x_test)  # shape (1, 1000)
        # Get label
        preds = np.argmax(output, axis=1)
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time

    def DEM(self):
        """
        Simulates DEM method at inference stage
        :return: elapsed time, in milliseconds
        """
        tf.reset_default_graph()
        sess = tf.Session()

        np.random.seed(10)
        test_att = np.random.rand(1, 85)  # shape (1, 85)
        x_test = np.random.rand(1000, self.feat_dim)  # shape (1000, feat_dim)
        x_labels = np.random.randint(0, 999, size=(1000, 1))  # shape (1000, 1)

        # Dictionary containing hidden layer neurons regarding output dimension
        hidden_layer = {
            "512": 256,
            "1024": 512,
            "1056": 512,
            "1280": 512,
            "1920": 1000,
            "2048": 1600,
            "2560": 1600,
            "4032": 2048
        }

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # # Placeholder
        # define placeholder for inputs to network
        att_features = tf.placeholder(tf.float32, [None, 85])
        visual_features = tf.placeholder(tf.float32, [None, self.feat_dim])

        # # Network
        # AwA 85 1600 2048 ReLu, 1e-3 * regularisers, 64 batch, 0.0001 Adam
        W_left_a1 = weight_variable([85, hidden_layer[str(self.feat_dim)]])
        b_left_a1 = bias_variable([hidden_layer[str(self.feat_dim)]])
        left_a1 = tf.nn.relu(tf.matmul(att_features, W_left_a1) + b_left_a1)

        W_left_a2 = weight_variable([hidden_layer[str(self.feat_dim)], self.feat_dim])
        b_left_a2 = bias_variable([self.feat_dim])
        left_a2 = tf.nn.relu(tf.matmul(left_a1, W_left_a2) + b_left_a2)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Restore variables from disk.
        saver.restore(sess, "./model_dem_" + str(self.feat_dim) + "/model.ckpt")
        feat_pre = sess.run(left_a2, feed_dict={att_features: test_att})  # shape (1, 2048)
        tic = time.time()
        # Check the values of the variables
        feat_pre = sess.run(left_a2, feed_dict={att_features: test_att})  # shape (1, 2048)
        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(feat_pre, x_test, metric='cosine')
        # Get the index of min distances
        idx_min = np.argmin(dist, axis=1)
        # Get the labels of predictions
        preds = x_labels[[i for i in idx_min]]
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time

    def IAP(self):
        """
        Simulates IAP method at inference stage
        :return: elapsed time, in milliseconds
        """
        np.random.seed(10)

        M = np.random.rand(1000, 85)  # shape (1000, 85)
        P = np.random.rand(1, 85)  # shape (1, 85) estimated attribute
        prior = np.random.rand(85, )  # shape (85,) prior attribute is computed by mean of training attributes
        x_labels = np.random.randint(0, 999, size=(1000, 1))  # shape (1000, 1)

        tic = time.time()
        prob = []
        for p in P:
            prob.append(np.prod(M * p + (1 - M) * (1 - p), axis=1) / np.prod(M * prior + (1 - M) * (1 - prior), axis=1))

        MCpred = np.argmax(prob, axis=1)
        pred = x_labels[[i for i in MCpred]]
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time

    def DAP(self):
        """
        Simulates DAP method at inference stage
        :return: elapsed time, in milliseconds
        """
        np.random.seed(10)

        M = np.random.rand(1000, 85)  # shape (1000, 85)
        P = np.random.rand(1, 85)  # shape (1, 85) estimated attribute
        prior = np.random.rand(85, )  # shape (85,) prior attribute is computed by mean of training attributes
        x_labels = np.random.randint(0, 999, size=(1000, 1))  # shape (1000, 1)

        tic = time.time()
        prob = []
        for p in P:
            prob.append(np.prod(M * p + (1 - M) * (1 - p), axis=1) / np.prod(M * prior + (1 - M) * (1 - prior), axis=1))

        MCpred = np.argmax(prob, axis=1)
        pred = x_labels[[i for i in MCpred]]
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time

    def fCLSWGAN(self):
        """
        Simulates fCLSWGAN method at inference stage
        :return: elapsed time, in milliseconds
        """
        img_feature_vector = np.random.rand(1, self.feat_dim)  # shape (1, n_dim_feat)

        # Load keras model
        model = load_model("./models_f-CLSWGAN/f-CLSWGAN_" + str(self.feat_dim) + ".h5")
        model.summary()

        dummy = model.predict(img_feature_vector)

        tic = time.time()
        # Predict class label
        preds = model.predict(img_feature_vector)
        toc = time.time()

        elapsed_time = (toc - tic) * 1000

        return elapsed_time


if __name__ == '__main__':
    # args = parser.parse_args()
    # model = str(args.model)
    # eval = Eval(args=args)

    features_dim = [512, 1024, 1056, 1280, 1920, 2048, 2560, 4032]

    avg_time_dict = dict()
    std_time_dict = dict()
    # Loop through feature dimensions
    for feat_dim in features_dim:

        print(f"[INFO] Feature Dimension: {str(feat_dim)}")

        eval = Eval(args=feat_dim)

        """
        methods = {"DAP": eval.DAP(),
                   "IAP": eval.IAP(),
                   "SAE": eval.SAE(),
                   "ESZSL": eval.ESZSL(),
                   "DEM": eval.DEM(),
                   "fCLSWGAN": eval.fCLSWGAN()}
        """

        # Loop through methods
        avgs = []
        stds = []
        # for method, name in zip(methods.values(), methods.keys()):

        # DAP
        times = []
        print(f"[INFO] Evaluating DAP... ", end='')
        for i in range(10):
            elapsed_time = eval.DAP()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(f"SUCCESS!")

        # IAP
        times = []
        print(f"[INFO] Evaluating IAP... ", end='')
        for i in range(10):
            elapsed_time = eval.IAP()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(f"SUCCESS!")

        # SAE
        times = []
        print(f"[INFO] Evaluating SAE... ", end='')
        for i in range(10):
            elapsed_time = eval.SAE()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(f"SUCCESS!")

        # ESZSL
        times = []
        print(f"[INFO] Evaluating ESZSL... ", end='')
        for i in range(10):
            elapsed_time = eval.ESZSL()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(f"SUCCESS!")

        # DEM
        times = []
        print(f"[INFO] Evaluating DEM... ", end='')
        for i in range(10):
            elapsed_time = eval.DEM()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(f"SUCCESS!")

        # f-CLSWGAN
        times = []
        print(f"[INFO] Evaluating f-CLSWGAN... ", end='')
        for i in range(10):
            elapsed_time = eval.fCLSWGAN()
            times.append(elapsed_time)

        avgs.append(np.mean(times[1:]))
        stds.append(np.std(times[1:]))
        print(times[1:])
        print(f"SUCCESS!")

        # Save data to dicts
        avg_time_dict[feat_dim] = np.array(avgs)
        std_time_dict[feat_dim] = np.array(stds)

    # print(avg_time_dict)
    # print(std_time_dict)

    methods = ["DAP", "IAP", "SAE", "ESZSL", "DEM", "f-CLSWGAN"]
    write_to_csv(avg_time_dict, std_time_dict, methods, features_dim)
    print(f"[INFO] Results saved at results_methods.csv")
