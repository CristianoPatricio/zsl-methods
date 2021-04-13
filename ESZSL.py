import numpy as np
import scipy.io
from sklearn.metrics import confusion_matrix
import argparse
import pickle

"""
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
"""

class ESZSL():

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

        labels = res101['labels']
        self.labels_train = labels[np.squeeze(att_splits[train_loc] - 1)]
        self.labels_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc] - 1)]
        self.labels_test = labels[np.squeeze(att_splits[test_loc] - 1)]
        self.labels_test_seen = labels[np.squeeze(att_splits[test_seen_loc] - 1)]
        self.labels_test_gzsl = np.concatenate((self.labels_test, self.labels_test_seen), axis=0)

        self.decoded_seen_classes = self.labels_test_seen.copy()
        self.decoded_unseen_classes = self.labels_test.copy()
        self.decoded_y_true = self.labels_test_gzsl.copy()

        self.train_labels_seen = np.unique(self.labels_train)
        self.val_labels_unseen = np.unique(self.labels_val)
        self.trainval_labels_seen = np.unique(self.labels_trainval)
        self.test_labels_unseen = np.unique(self.labels_test)
        self.test_labels_gzsl = np.unique(self.labels_test_gzsl)

        print("Number of overlapping classes between train and val:",
              len(set(self.train_labels_seen).intersection(set(self.val_labels_unseen))))
        print("Number of overlapping classes between trainval and test:",
              len(set(self.trainval_labels_seen).intersection(set(self.test_labels_unseen))))

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
            self.labels_test[self.labels_test == labels] = l
            l = l + 1
        m = 0
        for labels in self.test_labels_gzsl:
            self.labels_test_gzsl[self.labels_test_gzsl == labels] = m
            m = m + 1

        X_features = res101['features']
        self.train_vec = X_features[:, np.squeeze(att_splits[train_loc] - 1)]
        self.val_vec = X_features[:, np.squeeze(att_splits[val_loc] - 1)]
        self.trainval_vec = X_features[:, np.squeeze(att_splits[trainval_loc] - 1)]
        self.test_vec = X_features[:, np.squeeze(att_splits[test_loc] - 1)]
        self.test_vec_seen = X_features[:, np.squeeze(att_splits[test_seen_loc] - 1)]
        self.test_gzsl = np.concatenate((self.test_vec, self.test_vec_seen), axis=1)

        def normalization(vec, mean, std):
            sol = vec - mean
            sol1 = sol / std
            return sol1

        # Signature matrix
        signature = att_splits['att']
        self.train_sig = signature[:, (self.train_labels_seen) - 1]
        self.val_sig = signature[:, (self.val_labels_unseen) - 1]
        self.trainval_sig = signature[:, (self.trainval_labels_seen) - 1]
        self.test_sig = signature[:, (self.test_labels_unseen) - 1]
        self.test_gzsl_sig = signature[:, (self.test_labels_gzsl) - 1]

        # params for train and val set
        m_train = self.labels_train.shape[0]
        z_train = len(self.train_labels_seen)

        # params for trainval and test set
        m_trainval = self.labels_trainval.shape[0]
        z_trainval = len(self.trainval_labels_seen)

        # ground truth for train and val set
        self.gt_train = 0 * np.ones((m_train, z_train))
        self.gt_train[np.arange(m_train), np.squeeze(self.labels_train)] = 1

        # grountruth for trainval and test set
        self.gt_trainval = 0 * np.ones((m_trainval, z_trainval))
        self.gt_trainval[np.arange(m_trainval), np.squeeze(self.labels_trainval)] = 1

    def find_hyperparams(self):
        # train set
        d_train = self.train_vec.shape[0]
        a_train = self.train_sig.shape[0]

        accu = 0.10
        alph1 = 4
        gamm1 = 1

        # Weights
        V = np.zeros((d_train, a_train))
        for alpha in range(-3, 4):
            for gamma in range(-3, 4):
                # One line solution
                part_1 = np.linalg.pinv(
                    np.matmul(self.train_vec, self.train_vec.transpose()) + (10 ** alpha) * np.eye(d_train))
                part_0 = np.matmul(np.matmul(self.train_vec, self.gt_train), self.train_sig.transpose())
                part_2 = np.linalg.pinv(
                    np.matmul(self.train_sig, self.train_sig.transpose()) + (10 ** gamma) * np.eye(a_train))

                V = np.matmul(np.matmul(part_1, part_0), part_2)
                # print(V)

                # predictions
                outputs = np.matmul(np.matmul(self.val_vec.transpose(), V), self.val_sig)
                preds = np.array([np.argmax(output) for output in outputs])

                # print(accuracy_score(labels_val,preds))
                cm = confusion_matrix(self.labels_val, preds)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                avg = sum(cm.diagonal()) / len(self.val_labels_unseen)

                if avg > accu:
                    accu = avg
                    alph1 = alpha
                    gamm1 = gamma
                    print(alph1, gamm1, avg)
        print("Alpha and gamma:", alph1, gamm1)

        return alpha, gamma

    def train(self, alpha, gamma):
        print("Trainval shape", self.trainval_vec.shape)
        print("GT trainval shape", self.gt_trainval.shape)
        print("Sig trainval shape", self.trainval_sig.shape)
        # trainval set
        d_trainval = self.trainval_vec.shape[0]
        a_trainval = self.trainval_sig.shape[0]
        W = np.zeros((d_trainval, a_trainval))
        part_1_test = np.linalg.pinv(
            np.matmul(self.trainval_vec, self.trainval_vec.transpose()) + (10 ** alpha) * np.eye(d_trainval))
        print(part_1_test.shape)
        part_0_test = np.matmul(np.matmul(self.trainval_vec, self.gt_trainval), self.trainval_sig.transpose())
        print(part_0_test.shape)
        part_2_test = np.linalg.pinv(
            np.matmul(self.trainval_sig, self.trainval_sig.transpose()) + (10 ** gamma) * np.eye(a_trainval))
        print(part_2_test.shape)
        W = np.matmul(np.matmul(part_1_test, part_0_test), part_2_test)

        return W

    def gzsl_accuracy(self, weights):
        """
        Calculate harmonic mean
        :param y_true: ground truth labels
        :param y_preds: estimated labels
        :param seen_classes: array of seen classes
        :param unseen_classes: array of unseen classes
        :return: harmonic mean
        """

        outputs_1 = np.matmul(np.matmul(self.test_gzsl.transpose(), weights), self.test_gzsl_sig)
        preds_1 = np.argmax(outputs_1, axis=1)
        cmat = confusion_matrix(self.labels_test_gzsl, preds_1)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        seen_classes_encoded = self.labels_test_gzsl[np.where([self.decoded_y_true == i for i in self.decoded_seen_classes])[1]]
        unseen_classes_encoded = self.labels_test_gzsl[np.where([self.decoded_y_true == i for i in self.decoded_unseen_classes])[1]]

        acc_seen = np.mean(per_class_acc[seen_classes_encoded])
        acc_unseen = np.mean(per_class_acc[unseen_classes_encoded])
        harmonic_mean = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

        return acc_seen, acc_unseen, harmonic_mean

    def zsl_accuracy(self, weights):

        outputs_1 = np.matmul(np.matmul(self.test_vec.transpose(), weights), self.test_sig)
        preds_1 = np.argmax(outputs_1, axis=1)
        cmat = confusion_matrix(self.labels_test, preds_1)
        per_class_acc = cmat.diagonal() / cmat.sum(axis=1)

        acc = np.mean(per_class_acc)

        return acc

    def test(self, weights):
        """

        :param weights:
        :return: ZSL Accuracy, GZSL Seen Accuracy, GZSL Unseen Accuracy, GZSL Harmonic Mean
        """

        print("x_test: ", self.test_vec.shape)
        print("s_test: ", self.test_sig.shape)
        print(f"Weights shape: {weights.shape}")

        # ZSL
        zsl_acc = self.zsl_accuracy(weights)

        # GZSL
        acc_seen, acc_unseen, harmonic_mean = self.gzsl_accuracy(weights)

        #print(f"The top-1 accuracy is: {(zsl_acc * 100):.2f} %")
        #print(f"[GZSL]: Seen: {(acc_seen * 100):.2f} % | Unseen: {(acc_unseen * 100):.2f} % | H: {(harmonic_mean * 100):.2f} %")

        return zsl_acc, acc_seen, acc_unseen, harmonic_mean

"""
if __name__ == '__main__':
    args = parser.parse_args()
    alpha = args.alpha
    gamma = args.gamma
    model = ESZSL(args=args)
    if not args.alpha and args.gamma:
        alpha, gamma = model.find_hyperparams()
    weights = model.train(alpha, gamma)
    zsl_acc, acc_seen, acc_unseen, harmonic_mean = model.test(weights)
"""