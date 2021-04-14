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


class SAE():
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
        self.labels_train = labels[np.squeeze(att_splits[train_loc] - 1)]
        self.original_labels_train = self.labels_train.copy()
        self.labels_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)]
        self.original_labels_trainval = self.labels_trainval.copy()
        self.labels_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)]
        self.original_labels_test = self.labels_test_unseen.copy()
        self.labels_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)]
        self.labels_test_gzsl = np.concatenate((self.labels_test_unseen, self.labels_test_seen), axis=0)
        self.original_labels_test_gzsl = self.labels_test_gzsl.copy()

        # For GZSL evaluation
        self.decoded_seen_classes = self.labels_test_seen.copy()
        self.decoded_unseen_classes = self.labels_test_unseen.copy()
        self.decoded_y_true = self.labels_test_gzsl.copy()

        self.train_labels_seen = np.unique(self.labels_train)
        self.val_labels_unseen = np.unique(self.labels_val)
        self.trainval_labels_seen = np.unique(self.labels_trainval)
        self.test_labels_unseen = np.unique(self.labels_test_unseen)
        self.test_labels_gzsl = np.unique(self.labels_test_gzsl)

        # print("Number of overlapping classes between train and val:",
        #      len(set(self.train_labels_seen).intersection(set(self.val_labels_unseen))))
        # print("Number of overlapping classes between trainval and test:",
        #      len(set(self.trainval_labels_seen).intersection(set(self.test_labels_unseen))))

        i = 0
        for labels in self.train_labels_seen:
            self.labels_train[self.labels_train == labels] = i
            i = i + 1
        j = 0
        for labels in self.val_labels_unseen:
            self.labels_val[self.labels_val == labels] = j
            j = j + 1
        k = 0
        for labels in self.trainval_labels_seen:
            self.labels_trainval[self.labels_trainval == labels] = k
            k = k + 1
        l = 0
        for labels in self.test_labels_unseen:
            self.labels_test_unseen[self.labels_test_unseen == labels] = l
            l = l + 1
        m = 0
        for labels in self.test_labels_gzsl:
            self.labels_test_gzsl[self.labels_test_gzsl == labels] = m
            m = m + 1

        X_features = res101['features']
        self.train_vec = X_features[:, np.squeeze(att_splits[train_loc] - 1)]
        self.train_vec = self.normalizeFeature(self.train_vec.T).T
        self.val_vec = X_features[:, np.squeeze(att_splits[val_loc] - 1)]
        self.trainval_vec = X_features[:, np.squeeze(att_splits[trainval_loc] - 1)]
        self.trainval_vec = self.normalizeFeature(self.trainval_vec.T).T
        self.test_vec = X_features[:, np.squeeze(att_splits[test_loc] - 1)]
        self.test_vec_seen = X_features[:, np.squeeze(att_splits[test_seen_loc] - 1)]
        self.test_gzsl = np.concatenate((self.test_vec, self.test_vec_seen), axis=1)
        self.test_gzsl = self.normalizeFeature(self.test_gzsl.T).T

        # Signature matrix
        signature = att_splits['att']
        self.train_sig = signature[:, (self.train_labels_seen) - 1]
        self.train_all_sig = signature.T[(self.original_labels_train) - 1].squeeze().transpose()
        self.val_sig = signature[:, (self.val_labels_unseen) - 1]
        self.trainval_sig = signature[:, (self.trainval_labels_seen) - 1]
        self.trainval_all_sig = signature.T[(self.original_labels_trainval) - 1].squeeze().transpose()
        self.test_sig = signature[:, (self.test_labels_unseen) - 1]
        self.test_all_sig = signature.T[(self.original_labels_test) - 1].squeeze().transpose()
        self.test_gzsl_sig = signature[:, (self.test_labels_gzsl) - 1]
        self.test_all_gzsl_sig = signature.T[(self.original_labels_test_gzsl) - 1].squeeze().transpose()

    def normalizeFeature(self, x):
        # x = d x N dims (d: feature dimension, N: the number of features)
        x = x + 1e-10  # for avoid RuntimeWarning: invalid value encountered in divide
        feature_norm = np.sum(x ** 2, axis=1) ** 0.5  # l2-norm
        feat = x / feature_norm[:, np.newaxis]
        return feat

    def SAE(self, X, S, lamb):
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

    def train_zsl(self, lamb):
        W = self.SAE(self.train_vec, self.train_all_sig, lamb)
        return W

    def train_gzsl(self, lamb):
        W = self.SAE(self.trainval_vec, self.trainval_all_sig, lamb)
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
        s_pred = np.dot(self.test_gzsl.T, self.normalizeFeature(weights).T)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.test_gzsl_sig.T, metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.labels_test_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.labels_test_gzsl[
            np.where([self.decoded_y_true == i for i in self.decoded_seen_classes])[1]]
        unseen_classes_encoded = self.labels_test_gzsl[
            np.where([self.decoded_y_true == i for i in self.decoded_unseen_classes])[1]]

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
        x_pred = np.dot(self.test_gzsl_sig.T, weights)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(self.test_gzsl.T, self.normalizeFeature(x_pred), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.labels_test_gzsl, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.labels_test_gzsl[
            np.where([self.decoded_y_true == i for i in self.decoded_seen_classes])[1]]
        unseen_classes_encoded = self.labels_test_gzsl[
            np.where([self.decoded_y_true == i for i in self.decoded_unseen_classes])[1]]

        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

        return acc_seen, acc_unseen, harmonic_mean

    def zsl_accuracy_semantic(self, weights):
        # Test [V >>> S]
        s_pred = np.dot(self.test_vec.T, self.normalizeFeature(weights).T)

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(s_pred, self.test_sig.transpose(), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.labels_test_unseen, preds)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def zsl_accuracy_feature(self, weights):
        # Test [S >>> V]
        x_pred = np.dot(self.test_sig.T, self.normalizeFeature(weights))

        # Calculate distance between the estimated representation and the projected prototypes
        dist = distance.cdist(self.test_vec.transpose(), self.normalizeFeature(x_pred), metric='cosine')
        # Get the labels of predictions
        preds = np.array([np.argmin(y) for y in dist])

        cmat = confusion_matrix(self.labels_test_unseen, preds)
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
