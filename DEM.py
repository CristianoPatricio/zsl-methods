import numpy as np
import scipy.io as sio
import argparse
import pickle
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
import tensorflow.compat.v1 as tf
from sklearn.metrics import accuracy_score

tf.disable_v2_behavior()

parser = argparse.ArgumentParser(description='DEM')
parser.add_argument('--dataset', type=str, default='AWA2',
                    help='Name of the dataset.')
parser.add_argument('--dataset_path', type=str, default='./datasets/',
                    help='Directory path containing the datasets.')
parser.add_argument('--filename', type=str, default='res101.mat',
                    help='Name of the dataset features file.')
parser.add_argument('--lamb', type=float, default=1e-3,
                    help='Regularization loss hyperparameter.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch-size.')
parser.add_argument('--hidden_dim', type=int, default=1600,
                    help='Dimension of hidden layer')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_gzsl_accuracy(sess, att_features, A2, S, X, Y):
    # Compute feature predictions
    feat_preds = sess.run(A2, feed_dict={att_features: S})
    # Calculate distance between the estimated representation and the projected prototypes
    dist = distance.cdist(X, feat_preds, metric='euclidean')
    # Get the labels of predictions
    preds = np.array([np.argmin(y) for y in dist])

    # Compute accuracy
    unique_labels = np.unique(Y)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(Y == l)[0]
        acc += accuracy_score(Y[idx], preds[idx])
    acc = acc / unique_labels.shape[0]

    return acc


def compute_zsl_accuracy(sess, att_features, A2, S, X, Y):
    feat_preds = sess.run(A2, feed_dict={att_features: S})
    # Calculate distance between the estimated representation and the projected prototypes
    dist = distance.cdist(X, feat_preds, metric='euclidean')
    # Get the labels of predictions
    preds = np.array([np.argmin(y) for y in dist])

    cmat = confusion_matrix(Y, preds)
    per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

    acc = np.mean(per_class_acc)

    return acc


class DEM:

    def __init__(self, args):
        print(f"Evaluating on {args.dataset}...")

        point_position = args.filename.find('.')
        extension = args.filename[point_position:]

        if "mat" in extension:  # dataset file is a .mat file
            matcontent = sio.loadmat(args.dataset_path + args.dataset + '/' + args.filename)
        else:  # dataset file is a .pickle file
            with open(args.dataset_path + args.dataset + '/' + args.filename, "rb") as f:
                matcontent = pickle.load(f)

        att_splits = sio.loadmat(args.dataset_path + args.dataset + '/att_splits.mat')

        train_loc = 'train_loc'
        val_loc = 'val_loc'
        trainval_loc = 'trainval_loc'
        test_loc = 'test_unseen_loc'
        test_seen_loc = 'test_seen_loc'

        features = matcontent['features']  # shape (feat_dim, n_samples)
        labels = matcontent['labels']  # shape (n_samples,)
        attributes = att_splits['original_att']  # shape (attribute_dim, n_classes)

        # Train
        self.X_train = features[:, np.squeeze(att_splits[train_loc] - 1)].T  # shape (n_samples_train, feat_dim)
        self.Y_train = labels[np.squeeze(att_splits[train_loc] - 1)].squeeze()  # shape (n_samples_train,)
        self.Y_train_unique = np.unique(self.Y_train)  # shape list [n_classes_train]
        self.S_train = attributes[:, self.Y_train - 1].squeeze().T  # shape (n_samples_train, attribute_dim)
        self.S_train_unique = np.unique(self.S_train, axis=0)  # shape (n_attributes_train, attribute_dim)

        # Validation
        self.X_val = features[:, np.squeeze(att_splits[val_loc] - 1)].T  # shape (n_samples_val, feat_dim)
        self.Y_val = labels[np.squeeze(att_splits[val_loc] - 1)].squeeze()  # shape (n_samples_val,)
        self.Y_val_unique = np.unique(self.Y_val)  # shape list [n_classes_val]
        self.S_val = attributes[:, self.Y_val - 1].squeeze().T  # shape (n_samples_val, attribute_dim)
        self.S_val_unique = np.unique(self.S_val, axis=0)  # shape (n_attributes_val, attribute_dim)

        # TrainVal
        self.X_trainval = features[:, np.squeeze(att_splits[trainval_loc] - 1)].T  # shape (n_samples_train, feat_dim)
        self.Y_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)].squeeze()  # shape (n_samples_train,)
        self.Y_trainval_unique = np.unique(self.Y_trainval)  # shape list [n_classes_trainval]
        self.S_trainval = attributes[:, self.Y_trainval - 1].squeeze().T  # shape (n_samples_train, attribute_dim)
        self.S_trainval_unique = np.unique(self.S_trainval, axis=0)  # shape (n_attributes_trainval, attribute_dim)

        # Test Unseen
        self.X_test_unseen = features[:, np.squeeze(att_splits[test_loc] - 1)].T  # shape (n_samples_test_unseen, feat_dim)
        self.Y_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)].squeeze()  # shape (n_samples_test_unseen,)
        self.Y_test_unseen_unique = np.unique(self.Y_test_unseen)   # shape list [n_classes_test_unseen]
        self.Y_test_unseen_orig = np.copy(self.Y_test_unseen)
        self.S_test_unseen = attributes[:, np.unique(self.Y_test_unseen) - 1].squeeze().T  # shape (n_test_unseen_classes, attribute_dim)
        self.S_test_unseen_unique = np.unique(self.S_test_unseen, axis=0)   # shape (n_attributes_test_unseen, attribute_dim)

        # Test Seen
        self.X_test_seen = features[:, np.squeeze(att_splits[test_seen_loc] - 1)].T  # shape (n_samples_test_unseen, feat_dim)
        self.Y_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)].squeeze()  # shape (n_samples_test_seen,)
        self.Y_test_seen_unique = np.unique(self.Y_test_seen)   # shape list [n_classes_test_seen]
        self.S_test_seen = attributes[:, np.unique(self.Y_test_seen) - 1].squeeze().T  # shape (n_test_seen_classes, attribute_dim)
        self.S_test_seen_unique = np.unique(self.S_test_seen, axis=0)   # shape (n_attributes_test_seen, attribute_dim)

        # Additional
        self.S_all = attributes.T   # shape (n_attributes_unique, attribute_dim)

        # Converts Y_test_unseen labels into a range between 0 and len(Y_test_unseen) - 1
        l = 0
        for labels in self.Y_test_unseen_unique:
            self.Y_test_unseen[self.Y_test_unseen == labels] = l
            l = l + 1

        # Arguments
        self.hidden_dim = args.hidden_dim
        self.lamb = args.lamb
        self.lr = args.lr

    def data_iterator(self, batch_size):
        batch_idx = 0
        while True:
            # shuffle labels and features
            idxs = np.arange(0, len(self.X_trainval))
            np.random.shuffle(idxs)
            shuf_visual = self.X_trainval[idxs]
            shuf_att = self.S_trainval[idxs]
            batch_size = batch_size
            for batch_idx in range(0, len(self.X_trainval), batch_size):
                visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
                visual_batch = visual_batch.astype("float32")
                att_batch = shuf_att[batch_idx:batch_idx + batch_size]
                yield att_batch, visual_batch

    def test(self, sess, att_features, A2):
        zsl_acc = compute_zsl_accuracy(sess, att_features, A2, self.S_test_unseen, self.X_test_unseen, self.Y_test_unseen)
        gzsl_seen_acc = compute_gzsl_accuracy(sess, att_features, A2, self.S_all, self.X_test_seen, self.Y_test_seen - 1)
        gzsl_unseen_acc = compute_gzsl_accuracy(sess, att_features, A2, self.S_all, self.X_test_unseen, self.Y_test_unseen_orig - 1)
        gzsl_harmonic_mean = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

        print(f"[INFO]: ZSL Accuracy (%) - {zsl_acc:.5f}")
        print(f"[INFO]: GZSL - seen={gzsl_seen_acc:.5f} unseen={gzsl_unseen_acc:.5f} Harmonic={gzsl_harmonic_mean:.5f}")

    def train(self, batch_size):
        # Define the placeholders for the inputs of the network
        att_features = tf.placeholder(tf.float32, [None, self.S_trainval.shape[1]])
        visual_features = tf.placeholder(tf.float32, [None, self.X_trainval.shape[1]])

        # Network
        W1 = weight_variable([self.S_trainval.shape[1], self.hidden_dim])
        b1 = bias_variable([self.hidden_dim])
        A1 = tf.nn.relu(tf.matmul(att_features, W1) + b1)

        W2 = weight_variable([self.hidden_dim, self.X_trainval.shape[1]])
        b2 = bias_variable([self.X_trainval.shape[1]])
        A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)

        # # loss
        loss = tf.reduce_mean(tf.square(A2 - visual_features))

        # L2 regularisation for the fully connected parameters.
        regularisers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
                        + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2))

        loss += self.lamb * regularisers

        train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        # Run
        accuracy = []
        iter_ = self.data_iterator(batch_size=batch_size)
        for i in range(1000000):
            att_batch_val, visual_batch_val = next(iter_)
            sess.run(train_step, feed_dict={att_features: att_batch_val, visual_features: visual_batch_val})
            if i % 1000 == 0:
                self.test(sess, att_features, A2)


if __name__ == '__main__':
    np.random.seed(10)
    args = parser.parse_args()
    dem = DEM(args)
    dem.train(batch_size=args.batch_size)
