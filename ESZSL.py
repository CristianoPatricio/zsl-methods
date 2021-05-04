import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import pickle

parser = argparse.ArgumentParser(description='ESZSL')
parser.add_argument('--dataset', type=str, default='AWA2',
                    help='Name of the dataset')
parser.add_argument('--dataset_path', type=str, default='./datasets/',
                    help='Name of the dataset')
parser.add_argument('--filename', type=str, default='res101.mat',
                    help='Name of the dataset file')
parser.add_argument('--alpha', type=int, default=2,
                    help='value of hyper-parameter')
parser.add_argument('--gamma', type=int, default=2,
                    help='value of hyper-parameter')


def encode_labels(Y):
    i = 0
    for labels in np.unique(Y):
        Y[Y == labels] = i
        i += 1

    return Y


def compute_gzsl_accuracy(X, Y, S, weights):

    outputs = np.matmul(np.matmul(X.transpose(), weights), S)
    preds = np.argmax(outputs, axis=1) + 1

    # Compute accuracy
    unique_labels = np.unique(Y)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(Y == l)[0]
        acc += accuracy_score(Y[idx], preds[idx])
    acc = acc / unique_labels.shape[0]

    return acc


class ESZSL:

    def __init__(self, args):

        point_position = args.filename.find('.')
        extension = args.filename[point_position:]

        if "mat" in extension:  # dataset file is a .mat file
            res101 = scipy.io.loadmat(args.dataset_path + args.dataset + '/' + args.filename)
        else:   # dataset file is a .pickle file
            with open(args.dataset_path + args.dataset + '/' + args.filename, "rb") as f:
                res101 = pickle.load(f)

        att_splits = scipy.io.loadmat(args.dataset_path + args.dataset + '/att_splits.mat')

        trainval_loc = 'trainval_loc'
        train_loc = 'train_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'
        test_seen_loc = 'test_seen_loc'

        features = res101['features']
        labels = res101['labels']
        attributes = att_splits['att']

        # Train
        self.X_train = features[:, np.squeeze(att_splits[train_loc] - 1)]   # shape (features_dim, n_samples)
        self.Y_train = labels[np.squeeze(att_splits[train_loc] - 1)]    # shape (n_samples, 1)
        self.Y_train_unique = np.unique(self.Y_train)   # shape (n_train_classes,)
        self.S_train = attributes[:, self.Y_train_unique - 1]    # shape (attributes_dim, n_train_classes)

        # Validation
        self.X_val = features[:, np.squeeze(att_splits[val_loc] - 1)]
        self.Y_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.Y_val_unique = np.unique(self.Y_val)
        self.S_val = attributes[:, self.Y_val_unique - 1]

        # TrainVal
        self.X_trainval = features[:, np.squeeze(att_splits[trainval_loc] - 1)]
        self.Y_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)]
        self.Y_trainval_unique = np.unique(self.Y_trainval)
        self.S_trainval = attributes[:, self.Y_trainval_unique - 1]

        # Test Unseen
        self.X_test_unseen = features[:, np.squeeze(att_splits[test_loc] - 1)]
        self.Y_test_unseen = labels[np.squeeze(att_splits[test_loc] - 1)]
        self.Y_test_unseen_unique = np.unique(self.Y_test_unseen)
        self.Y_test_unseen_orig = self.Y_test_unseen.copy()
        self.S_test_unseen = attributes[:, self.Y_test_unseen_unique - 1]

        # Test Seen
        self.X_test_seen = features[:, np.squeeze(att_splits[test_seen_loc] - 1)]
        self.Y_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)]
        self.Y_test_seen_unique = np.unique(self.Y_test_seen)
        self.Y_test_seen_orig = self.Y_test_seen.copy()
        self.S_test_seen = attributes[:, self.Y_test_seen_unique - 1]

        # GZSL
        self.X_gzsl = np.concatenate((self.X_test_unseen, self.X_test_seen), axis=1)
        self.Y_gzsl = np.concatenate((self.Y_test_unseen, self.Y_test_seen), axis=0)
        self.Y_gzsl_unique = np.unique(self.Y_gzsl)
        self.Y_gzsl_orig = self.Y_gzsl.copy()
        self.S_gszl = attributes[:, self.Y_gzsl_unique - 1]

        print("Number of overlapping classes between train and val:",
              len(set(self.Y_train_unique).intersection(set(self.Y_val_unique))))
        print("Number of overlapping classes between trainval and test:",
              len(set(self.Y_trainval_unique).intersection(set(self.Y_test_unseen_unique))))

        # Additional
        self.Y_train = encode_labels(self.Y_train)
        self.Y_val = encode_labels(self.Y_val)
        self.Y_trainval = encode_labels(self.Y_trainval)
        self.Y_test_unseen = encode_labels(self.Y_test_unseen)
        self.Y_gzsl = encode_labels(self.Y_gzsl)

        # params for train and val set
        m_train = self.Y_train.shape[0]
        z_train = len(self.Y_train_unique)

        # params for trainval and test set
        m_trainval = self.Y_trainval.shape[0]
        z_trainval = len(self.Y_trainval_unique)

        # ground truth for train and val set
        self.gt_train = 0 * np.ones((m_train, z_train))
        self.gt_train[np.arange(m_train), np.squeeze(self.Y_train)] = 1

        # grountruth for trainval and test set
        self.gt_trainval = 0 * np.ones((m_trainval, z_trainval))
        self.gt_trainval[np.arange(m_trainval), np.squeeze(self.Y_trainval)] = 1

    def find_hyperparams(self):
        # train set
        d_train = self.X_train.shape[0]
        a_train = self.S_train.shape[0]

        accu = 0.10
        alph1 = 4
        gamm1 = 1

        # Weights
        V = np.zeros((d_train, a_train))
        for alpha in range(-3, 4):
            for gamma in range(-3, 4):
                # One line solution
                part_1 = np.linalg.pinv(
                    np.matmul(self.X_train, self.X_train.transpose()) + (10 ** alpha) * np.eye(d_train))
                part_0 = np.matmul(np.matmul(self.X_train, self.gt_train), self.S_train.transpose())
                part_2 = np.linalg.pinv(
                    np.matmul(self.S_train, self.S_train.transpose()) + (10 ** gamma) * np.eye(a_train))

                V = np.matmul(np.matmul(part_1, part_0), part_2)
                # print(V)

                # predictions
                outputs = np.matmul(np.matmul(self.X_val.transpose(), V), self.S_val)
                preds = np.array([np.argmax(output) for output in outputs])

                # print(accuracy_score(labels_val,preds))
                cm = confusion_matrix(self.Y_val, preds)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                avg = sum(cm.diagonal()) / len(self.Y_val_unique)

                if avg > accu:
                    accu = avg
                    alph1 = alpha
                    gamm1 = gamma
                    print(alph1, gamm1, avg)
        print("Alpha and gamma:", alph1, gamm1)

        return alph1, gamm1

    def train(self, alpha, gamma):
        print("Trainval shape", self.X_trainval.shape)
        print("GT trainval shape", self.gt_trainval.shape)
        print("Sig trainval shape", self.S_trainval.shape)
        # trainval set
        d_trainval = self.X_trainval.shape[0]
        a_trainval = self.S_trainval.shape[0]
        W = np.zeros((d_trainval, a_trainval))
        part_1_test = np.linalg.pinv(
            np.matmul(self.X_trainval, self.X_trainval.transpose()) + (10 ** alpha) * np.eye(d_trainval))
        print(part_1_test.shape)
        part_0_test = np.matmul(np.matmul(self.X_trainval, self.gt_trainval), self.S_trainval.transpose())
        print(part_0_test.shape)
        part_2_test = np.linalg.pinv(
            np.matmul(self.S_trainval, self.S_trainval.transpose()) + (10 ** gamma) * np.eye(a_trainval))
        print(part_2_test.shape)
        W = np.matmul(np.matmul(part_1_test, part_0_test), part_2_test)

        return W

    def zsl_accuracy(self, weights):

        outputs_1 = np.matmul(np.matmul(self.X_test_unseen.transpose(), weights), self.S_test_unseen)
        preds_1 = np.argmax(outputs_1, axis=1)
        cmat = confusion_matrix(self.Y_test_unseen, preds_1)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def test(self, weights):
        """

        :param weights:
        :return: ZSL Accuracy, GZSL Seen Accuracy, GZSL Unseen Accuracy, GZSL Harmonic Mean
        """

        print("x_test: ", self.X_test_unseen.shape)
        print("s_test: ", self.S_test_unseen.shape)
        print(f"Weights shape: {weights.shape}")

        # ZSL
        zsl_acc = self.zsl_accuracy(weights)

        # GZSL
        acc_seen = compute_gzsl_accuracy(self.X_test_seen, self.Y_test_seen_orig, self.S_gszl, weights)
        acc_unseen = compute_gzsl_accuracy(self.X_test_unseen, self.Y_test_unseen_orig, self.S_gszl, weights)
        harmonic_mean = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)

        print(f"The top-1 accuracy is: {(zsl_acc * 100):.2f} %")
        print(f"[GZSL]: Seen: {(acc_seen * 100):.2f} % | Unseen: {(acc_unseen * 100):.2f} % | H: {(harmonic_mean * 100):.2f} %")

        return zsl_acc, acc_seen, acc_unseen, harmonic_mean


if __name__ == '__main__':
    args = parser.parse_args()
    alpha = args.alpha
    gamma = args.gamma
    model = ESZSL(args=args)
    if not args.alpha and args.gamma:
        alpha, gamma = model.find_hyperparams()
    weights = model.train(alpha, gamma)
    zsl_acc, acc_seen, acc_unseen, harmonic_mean = model.test(weights)
