import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
import argparse
from scipy.linalg import solve_sylvester
from scipy.spatial import distance
import pickle

parser = argparse.ArgumentParser(description='SAE')
parser.add_argument('--dataset', type=str, default='AWA2',
                    help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='./datasets/',
                    help='Name of the dataset')
parser.add_argument('--filename', type=str, default='res101.mat',
                    help='Name of the dataset features file')
parser.add_argument('--lamb_ZSL', type=float, default=2,
                    help='value of hyper-parameter')
parser.add_argument('--lamb_GZSL', type=float, default=2,
                    help='value of hyper-parameter')
parser.add_argument('--setting', type=str, default='V2S',
                    help='Type of evaluation {V2S, S2V}.')


def encode_labels(Y):
    i = 0
    for labels in np.unique(Y):
        Y[Y == labels] = i
        i += 1

    return Y


def normalizeFeature(x):
    # x = d x N dims (d: feature dimension, N: the number of features)
    x = x + 1e-10  # for avoid RuntimeWarning: invalid value encountered in divide
    feature_norm = np.sum(x ** 2, axis=1) ** 0.5  # l2-norm
    feat = x / feature_norm[:, np.newaxis]
    return feat


def compute_W(X, S, lamb):
    """
    SAE - Semantic Autoencoder
    :param X: d x N data matrix
    :param S: k x N semantic matrix
    :param lamb: regularization parameter
    :return: W -> k x d projection function
    """

    A = np.dot(S, S.T)
    B = lamb * np.dot(X, X.T)
    C = (1 + lamb) * np.dot(S, X.T)
    W = solve_sylvester(A, B, C)

    return W


class SAE:
    """docstring for ClassName"""

    def __init__(self, args):
        print(f"Evaluating on {args.dataset}...")

        point_position = args.filename.find('.')
        extension = args.filename[point_position:]

        if "mat" in extension:  # dataset file is a .mat file
            res101 = scipy.io.loadmat(args.dataset_path + args.dataset + '/' + args.filename)
        else:  # dataset file is a .pickle file
            with open(args.dataset_path + args.dataset + '/' + args.filename, "rb") as f:
                res101 = pickle.load(f)

        att_splits = scipy.io.loadmat(args.dataset_path + args.dataset + '/att_splits.mat')

        trainval_loc = 'trainval_loc'
        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'
        test_seen_loc = 'test_seen_loc'

        labels = res101['labels']
        features = res101['features']
        attributes = att_splits['att']

        # Train
        self.X_train = features[:, np.squeeze(att_splits[train_loc] - 1)]   # shape (features_dim, n_samples)
        self.X_train = normalizeFeature(self.X_train.T).T   # shape (features_dim, n_samples)
        self.Y_train = labels[np.squeeze(att_splits[train_loc] - 1)]    # shape (n_samples, 1)
        self.Y_train_unique = np.unique(self.Y_train)   # shape (n_samples,)
        self.Y_train_orig = self.Y_train.copy()     # shape (n_samples, 1)
        self.S_train = attributes.T[self.Y_train_orig - 1].squeeze().transpose()    # shape (n_samples_train, attribute_dim)
        self.S_train_unique = attributes[:, self.Y_train_unique - 1]    # shape (attributes_dim, n_train_classes)

        # Validation
        self.X_val = features[:, np.squeeze(att_splits[val_loc] - 1)]
        self.Y_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.Y_val_unique = np.unique(self.Y_val)
        self.S_val = attributes.T[self.Y_val - 1].squeeze().transpose()
        self.S_val_unique = attributes[:, self.Y_val_unique - 1]

        # TrainVal
        self.X_trainval = features[:, np.squeeze(att_splits[trainval_loc] - 1)]
        self.X_trainval = normalizeFeature(self.X_trainval.T).T
        self.Y_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)]
        self.Y_trainval_unique = np.unique(self.Y_trainval)
        self.Y_trainval_orig = self.Y_trainval.copy()
        self.S_trainval = attributes.T[self.Y_trainval_orig - 1].squeeze().transpose()
        self.S_trainval_unique = attributes[:, self.Y_trainval_unique - 1]

        # Test Unseen
        self.X_test_unseen = features[:, np.squeeze(att_splits[test_loc] - 1)]
        self.Y_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)]
        self.Y_test_unseen_unique = np.unique(self.Y_test_unseen)
        self.Y_test_unseen_orig = self.Y_test_unseen.copy()
        self.S_test_unseen = attributes.T[self.Y_test_unseen_orig - 1].squeeze().transpose()
        self.S_test_unseen_unique = attributes[:, self.Y_test_unseen_unique - 1]

        # Test Seen
        self.X_test_seen = features[:, np.squeeze(att_splits[test_seen_loc] - 1)]
        self.Y_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)]
        self.Y_test_seen_orig = self.Y_test_seen.copy()
        self.S_test_seen = attributes.T[self.Y_test_seen - 1].squeeze().transpose()
        self.S_test_seen_unique = attributes[:, self.Y_test_unseen_unique - 1]

        # GZSL
        self.X_gzsl = np.concatenate((self.X_test_unseen, self.X_test_seen), axis=1)
        self.X_gzsl = normalizeFeature(self.X_gzsl.T).T
        self.Y_gzsl = np.concatenate((self.Y_test_unseen, self.Y_test_seen), axis=0)
        self.Y_gzsl_unique = np.unique(self.Y_gzsl)
        self.Y_gzsl_orig = self.Y_gzsl.copy()
        self.S_gzsl = attributes.T[self.Y_gzsl_orig - 1].squeeze().transpose()
        self.S_gzsl_unique = attributes[:, self.Y_gzsl_unique - 1]

        # Additional
        self.Y_train = encode_labels(self.Y_train)
        self.Y_val = encode_labels(self.Y_val)
        self.Y_trainval = encode_labels(self.Y_trainval)
        self.Y_test_unseen = encode_labels(self.Y_test_unseen)
        self.Y_gzsl = encode_labels(self.Y_gzsl)

    def train_zsl(self, lamb):
        W = compute_W(self.X_train, self.S_train, lamb)
        return W

    def train_gzsl(self, lamb):
        W = compute_W(self.X_trainval, self.S_trainval, lamb)
        return W

    def gzsl_accuracy_semantic(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """

        # Test [V >>> S]
        s_pred = np.dot(self.X_gzsl.T, normalizeFeature(weights).T)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.S_gzsl_unique.T, metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.Y_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.Y_gzsl[
            np.where([self.Y_gzsl_orig == i for i in self.Y_test_seen_orig])[1]]
        unseen_classes_encoded = self.Y_gzsl[
            np.where([self.Y_gzsl_orig == i for i in self.Y_test_unseen_orig])[1]]

        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

        return acc_seen, acc_unseen, harmonic_mean

    def gzsl_accuracy_feature(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """
        # Test [S>>>V]
        x_pred = np.dot(self.S_gzsl_unique.T, weights)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(self.X_gzsl.T, normalizeFeature(x_pred), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.Y_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.Y_gzsl[
            np.where([self.Y_gzsl_orig == i for i in self.Y_test_seen_orig])[1]]
        unseen_classes_encoded = self.Y_gzsl[
            np.where([self.Y_gzsl_orig == i for i in self.Y_test_unseen_orig])[1]]

        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

        return acc_seen, acc_unseen, harmonic_mean

    def zsl_accuracy_semantic(self, weights):
        # Test [V >>> S]
        s_pred = np.dot(self.X_test_unseen.T, normalizeFeature(weights).T)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.S_test_unseen_unique.transpose(), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.Y_test_unseen, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def zsl_accuracy_feature(self, weights):
        # Test [S >>> V]
        x_pred = np.dot(self.S_test_unseen_unique.T, normalizeFeature(weights))

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(self.X_test_unseen.transpose(), normalizeFeature(x_pred), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.Y_test_unseen, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def test(self, weights_zsl, weights_gzsl, mode):
        if mode == 'V2S':
            zsl_acc = self.zsl_accuracy_semantic(weights_zsl)
            gzsl_acc_seen, gzsl_acc_unseen, gzsl_harmonic_mean = model.gzsl_accuracy_semantic(weights_gzsl)
        else:
            zsl_acc = self.zsl_accuracy_feature(weights_zsl)
            gzsl_acc_seen, gzsl_acc_unseen, gzsl_harmonic_mean = model.gzsl_accuracy_feature(weights_gzsl)

        return zsl_acc, gzsl_acc_seen, gzsl_acc_unseen, gzsl_harmonic_mean


if __name__ == "__main__":
    args = parser.parse_args()
    lamb_ZSL = args.lamb_ZSL
    lamb_GZSL = args.lamb_GZSL
    model = SAE(args=args)
    # Train
    weights_zsl = model.train_zsl(lamb_ZSL)
    weights_gzsl = model.train_gzsl(lamb_GZSL)
    # Test
    [zsl_acc, gzsl_acc_seen, gzsl_acc_unseen, gzsl_harmonic_mean] = model.test(weights_zsl, weights_gzsl, mode=args.setting)
    print(f"Mode: {args.setting}")
    print(f"[ZSL] Top-1 Accuracy (%): {(zsl_acc * 100):.2f}")
    print(f"[GZSL] Accuracy (%) - Seen: {(gzsl_acc_seen * 100):.2f}, Unseen: {(gzsl_acc_unseen * 100):.2f}, Harmonic: {(gzsl_harmonic_mean * 100):.2f}")
