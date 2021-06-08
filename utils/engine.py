#!/usr/bin/env python3
import os


def SAE_helper(dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split, default_lamb_ZSL, default_lamb_GZSL,
               default_att_split):
    dataset_path = dataset_path if dataset_path is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                                 "/xlsa17/data/"
    filename = filename if filename is not None else "res101"
    lamb_ZSL = lamb_ZSL if lamb_ZSL is not None else default_lamb_ZSL
    lamb_GZSL = lamb_GZSL if lamb_GZSL is not None else default_lamb_GZSL
    setting = setting if setting is not None else "V2S"
    att_split = att_split if att_split is not None else default_att_split

    return dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split


def SAE(dataset=None, dataset_path=None, filename=None, lamb_ZSL=None, lamb_GZSL=None, setting=None,
        att_split=None):
    """
    SAE - Semantic Autoencoder
    :param dataset: {AWA1, AWA2, CUB, SUN, aPY, LAD}
    :param dataset_path: {'./datasets/}
    :param filename: {name}
    :param lamb_ZSL: float value, default=2
    :param lamb_GZSL: float value, default=2
    :param setting: Type of evaluation {V2S, S2V}
    :param att_split: In case of LAD, specify the split to be evaluated (e.g. _0 .. 4)
    :return:
    """

    if dataset == "AWA2":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL, setting,
                                                                                           att_split,
                                                                                           default_lamb_ZSL=0.6,
                                                                                           default_lamb_GZSL=0.8,
                                                                                           default_att_split="")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL, setting,
                                                                                           att_split,
                                                                                           default_lamb_ZSL=0.2,
                                                                                           default_lamb_GZSL=0.2,
                                                                                           default_att_split="")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(setting) + "'")

    elif dataset == "AWA1":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=3.0,
                                                                                           default_lamb_GZSL=3.2,
                                                                                           default_att_split="")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=0.8,
                                                                                           default_lamb_GZSL=0.8,
                                                                                           default_att_split="")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(setting) + "'")

    elif dataset == "CUB":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=100,
                                                                                           default_lamb_GZSL=80,
                                                                                           default_att_split="")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=0.2,
                                                                                           default_lamb_GZSL=0.2,
                                                                                           default_att_split="")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(setting) + "'")

    elif dataset == "SUN":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=0.32,
                                                                                           default_lamb_GZSL=0.32,
                                                                                           default_att_split="")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=0.16,
                                                                                           default_lamb_GZSL=0.08,
                                                                                           default_att_split="")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(setting) + "'")

    elif dataset == "APY":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=2.0,
                                                                                           default_lamb_GZSL=2.0,
                                                                                           default_att_split="")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=4.0,
                                                                                           default_lamb_GZSL=2.56,
                                                                                           default_att_split="")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(setting) + "'")

    elif dataset == "LAD":

        if setting == "V2S":
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=51.2,
                                                                                           default_lamb_GZSL=51.2,
                                                                                           default_att_split="_0")
        else:
            # Get parameters
            [dataset_path, filename, lamb_ZSL, lamb_GZSL, setting, att_split] = SAE_helper(dataset_path, filename,
                                                                                           lamb_ZSL,
                                                                                           lamb_GZSL,
                                                                                           setting, att_split,
                                                                                           default_lamb_ZSL=6.4,
                                                                                           default_lamb_GZSL=6.4,
                                                                                           default_att_split="_0")

        # Run script
        os.system(
            "python SAE.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb_ZSL=" + str(lamb_ZSL) + " --lamb_GZSL=" + str(lamb_GZSL) + " --setting='" + str(
                setting) + "' --att_split='" + str(att_split) + "'")

    else:
        raise TypeError("Sorry, the dataset you provided does not exist.")


def ESZSL_helper(dataset_path, filename, alpha, gamma, att_split, default_alpha, default_gamma, default_att_split):
    dataset_path = dataset_path if dataset_path is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                                 "/xlsa17/data/"
    filename = filename if filename is not None else "res101.mat"
    alpha = alpha if alpha is not None else default_alpha
    gamma = gamma if gamma is not None else default_gamma
    att_split = att_split if att_split is not None else default_att_split

    return dataset_path, filename, alpha, gamma, att_split


def ESZSL(dataset=None, dataset_path=None, filename=None, alpha=None, gamma=None, att_split=None):
    """
    ESZSL
    :param dataset: dataset: {AWA1, AWA2, CUB, SUN, aPY, LAD}
    :param dataset_path: {'./datasets/}
    :param filename: {name.mat}
    :param alpha: float value, default=0
    :param gamma: float value, default=0
    :param att_split: In case of LAD, specify the split to be evaluated (e.g. att_splits_0 .. 4)
    :return:
    """

    if dataset == "AWA2":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=3,
                                                                         default_gamma=0, default_att_split="")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    elif dataset == "AWA1":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=3,
                                                                         default_gamma=0, default_att_split="")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    elif dataset == "CUB":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=2,
                                                                         default_gamma=0, default_att_split="")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    elif dataset == "SUN":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=2,
                                                                         default_gamma=2, default_att_split="")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    elif dataset == "APY":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=3,
                                                                         default_gamma=-1, default_att_split="")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    elif dataset == "LAD":

        # Get parameters
        [dataset_path, filename, alpha, gamma, att_split] = ESZSL_helper(dataset_path, filename, alpha, gamma,
                                                                         att_split, default_alpha=3,
                                                                         default_gamma=1, default_att_split="_0")

        # Run script
        os.system(
            "python ESZSL.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --alpha=" + str(alpha) + " --gamma=" + str(gamma) + " --att_split='" + str(att_split) + "'")

    else:
        raise TypeError("Sorry, the dataset you provided does not exist.")


def DEM_helper(dataset_path, filename, lamb, lr, batch_size, hidden_dim, default_lamb, default_lr, default_batch_size,
               default_hidden_dim):
    dataset_path = dataset_path if dataset_path is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                                 "/xlsa17/data/"
    filename = filename if filename is not None else "res101.mat"
    lamb = lamb if lamb is not None else default_lamb
    lr = lr if lr is not None else default_lr
    batch_size = batch_size if batch_size is not None else default_batch_size
    hidden_dim = hidden_dim if hidden_dim is not None else default_hidden_dim

    return dataset_path, filename, lamb, lr, batch_size, hidden_dim


def DEM(dataset=None, dataset_path=None, filename=None, lamb=None, lr=None, batch_size=None, hidden_dim=None,
        att_split=None):
    """
    Deep Embedding Model For ZSL
    :param dataset: {AWA1, AWA2, CUB, SUN, aPY, LAD}
    :param dataset_path: {'./datasets/}
    :param filename: {name.mat}
    :param lamb: float value, default=1e-3
    :param lr: float value, default=1e-4
    :param batch_size: batch_size, default=64
    :param hidden_dim: Dimension of the hidden layer, default=1600
    :param att_split: In case of LAD, specify the split to be evaluated (e.g. att_splits_0 .. 4)
    :return:
    """

    if dataset == "AWA2":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-3,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + "")

    elif dataset == "AWA1":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-3,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + "")

    elif dataset == "CUB":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-2,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + "")

    elif dataset == "SUN":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-5,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + "")

    elif dataset == "APY":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-4,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + "")

    elif dataset == "LAD":

        # Get parameters
        [dataset_path, filename, lamb, lr, batch_size, hidden_dim] = DEM_helper(dataset_path, filename, lamb, lr,
                                                                                batch_size, hidden_dim,
                                                                                default_lamb=1e-3,
                                                                                default_lr=1e-4, default_batch_size=64,
                                                                                default_hidden_dim=1600)
        att_split = att_split if att_split is not None else "_0"

        # Run script
        os.system(
            "python DEM.py --dataset='" + str(dataset) + "' --dataset_path='" + str(
                dataset_path) + "' --filename='" + str(filename) + \
            "' --lamb=" + str(lamb) + " --lr=" + str(lr) + " --batch_size=" + str(batch_size) + " --hidden_dim=" + \
            str(hidden_dim) + " --att_split=" + str(att_split) + "")

    else:
        raise TypeError("Sorry, the dataset you provided does not exist.")


#####################################################################################
# F-CLSWGAN
#####################################################################################

def f_CLSWGAN_Helper(dataroot, image_embedding, class_embedding, batch_size, resSize, attSize, nepoch, lr, beta1,
                     split_no,
                     cls_weight, syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz,
                     default_resSize, default_attSize, default_batch_size, default_nepoch, default_lr, default_beta1,
                     default_split_no, default_cls_weight, default_syn_num, default_ngh, default_ndh, default_lambda1,
                     default_classifier_checkpoint, default_syn_att, default_nz):
    dataroot = dataroot if dataroot is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                     "/xlsa17/data/"
    image_embedding = image_embedding if image_embedding is not None else "res101"
    class_embedding = class_embedding if class_embedding is not None else "att"
    resSize = resSize if resSize is not None else default_resSize
    attSize = attSize if attSize is not None else default_attSize
    batch_size = batch_size if batch_size is not None else default_batch_size
    nepoch = nepoch if nepoch is not None else default_nepoch
    lr = lr if lr is not None else default_lr
    beta1 = beta1 if beta1 is not None else default_beta1
    split_no = split_no if split_no is not None else default_split_no
    cls_weight = cls_weight if cls_weight is not None else default_cls_weight
    syn_num = syn_num if syn_num is not None else default_syn_num
    ngh = ngh if ngh is not None else default_ngh
    ndh = ndh if ndh is not None else default_ndh
    lambda1 = lambda1 if lambda1 is not None else default_lambda1
    classifier_checkpoint = classifier_checkpoint if classifier_checkpoint is not None else default_classifier_checkpoint
    syn_att = syn_att if syn_att is not None else default_syn_att
    nz = nz if nz is not None else default_nz

    return dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no, cls_weight, \
           syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz


def f_CLSWGAN(dataset, dataroot=None, image_embedding=None, class_embedding=None, split_no=None, attSize=None,
              resSize=None, nepoch=None, modeldir=None,
              lr=None, beta1=None, batch_size=None, cls_weight=None, syn_num=None, ngh=None, ndh=None, lambda1=None,
              classifier_checkpoint=None, syn_att=None, nz=None):
    if dataset == "AWA2" or dataset == "AWA1":

        [dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no,
         cls_weight, \
         syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz] = f_CLSWGAN_Helper(dataroot, image_embedding,
                                                                                            class_embedding, \
                                                                                            batch_size, resSize,
                                                                                            attSize, nepoch, lr, beta1,
                                                                                            split_no,
                                                                                            cls_weight, syn_num, ngh,
                                                                                            ndh, lambda1,
                                                                                            classifier_checkpoint,
                                                                                            syn_att, nz,
                                                                                            default_resSize=2048,
                                                                                            default_attSize=85,
                                                                                            default_batch_size=100,
                                                                                            default_nepoch=50,
                                                                                            default_lr=0.0001,
                                                                                            default_beta1=0.5,
                                                                                            default_split_no='',
                                                                                            default_cls_weight=0.01,
                                                                                            default_syn_num=300,
                                                                                            default_ngh=4096,
                                                                                            default_ndh=4096,
                                                                                            default_lambda1=10,
                                                                                            default_classifier_checkpoint=49,
                                                                                            default_syn_att='_',
                                                                                            default_nz=85)

        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_classifier_" + str(dataset))

        # Train classifier
        os.system("python f-CLSWGAN/train_cls.py --manualSeed 9182 --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + \
                  " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --split_no '" + str(split_no) + \
                  "' --attSize " + str(attSize) + " --resSize " + str(resSize) + " --nepoch " + str(
            nepoch) + " --modeldir '" + str(modeldir) + \
                  "' --beta1 " + str(beta1) + " --logdir '" + str(logdir) + "'")

        classifier_modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_" + str(dataset))

        # Train WGAN
        os.system("python f-CLSWGAN/train_wgan.py --manualSeed 9182 --dataset " + str(dataset) + " --split_no '" + str(
            split_no) + \
                  "' --syn_att '" + str(syn_att) + "' --cls_weight " + str(
            cls_weight) + " --preprocessing --val_every 1 " + \
                  "--lr " + str(lr) + " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + \
                  " --netG_name MLP_G --netD_name MLP_CRITIC --nepoch " + str(nepoch) + " --syn_num " + str(
            syn_num) + " --ngh " + str(ngh) + \
                  " --ndh " + str(ndh) + " --lambda1 " + str(lambda1) + " --critic_iter 5 --batch_size " + str(
            batch_size) + " --nz " + str(nz) + \
                  " --attSize " + str(attSize) + " --resSize " + str(resSize) + " --modeldir '" + str(
            modeldir) + "' --logdir '" + str(logdir) + \
                  "' --classifier_modeldir '" + str(classifier_modeldir) + "' --classifier_checkpoint " + str(
            classifier_checkpoint) + \
                  " --dataroot " + str(dataroot) + "")

        nclass_all = 50

        # Evaluate
        os.system("python f-CLSWGAN/evaluate.py --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + " --image_embedding " + \
                  str(image_embedding) + " --split_no '" + str(split_no) + "' --syn_att '" + str(
            syn_att) + "' --nclass_all " + str(nclass_all))

    elif dataset == "CUB":

        [dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no,
         cls_weight, \
         syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz] = f_CLSWGAN_Helper(dataroot, image_embedding,
                                                                                            class_embedding, \
                                                                                            batch_size, resSize,
                                                                                            attSize, nepoch, lr, beta1,
                                                                                            split_no,
                                                                                            cls_weight, syn_num, ngh,
                                                                                            ndh, lambda1,
                                                                                            classifier_checkpoint,
                                                                                            syn_att, nz,
                                                                                            default_resSize=2048,
                                                                                            default_attSize=312,
                                                                                            default_batch_size=100,
                                                                                            default_nepoch=50,
                                                                                            default_lr=0.0001,
                                                                                            default_beta1=0.5,
                                                                                            default_split_no='',
                                                                                            default_cls_weight=0.01,
                                                                                            default_syn_num=300,
                                                                                            default_ngh=4096,
                                                                                            default_ndh=4096,
                                                                                            default_lambda1=10,
                                                                                            default_classifier_checkpoint=49,
                                                                                            default_syn_att='_',
                                                                                            default_nz=312)

        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_classifier_" + str(dataset))

        # Train classifier
        os.system("python f-CLSWGAN/train_cls.py --manualSeed 9182 --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + \
                  " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --split_no '" + str(split_no) + \
                  "' --attSize " + str(attSize) + " --resSize " + str(resSize) + " --nepoch " + str(
            nepoch) + " --modeldir '" + str(modeldir) + \
                  "' --beta1 " + str(beta1) + " --logdir '" + str(logdir) + "'")

        classifier_modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_" + str(dataset))

        # Train WGAN
        os.system("python f-CLSWGAN/train_wgan.py --manualSeed 9182 --dataset " + str(dataset) + " --split_no '" + str(
            split_no) + \
                  "' --syn_att '" + str(syn_att) + "' --cls_weight " + str(
            cls_weight) + " --preprocessing --val_every 1 " + \
                  "--lr " + str(lr) + " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + \
                  " --netG_name MLP_G --netD_name MLP_CRITIC --nepoch " + str(nepoch) + " --syn_num " + str(
            syn_num) + " --ngh " + str(ngh) + \
                  " --ndh " + str(ndh) + " --lambda1 " + str(lambda1) + " --critic_iter 5 --batch_size " + str(
            batch_size) + " --nz " + str(nz) + \
                  " --attSize " + str(attSize) + " --resSize " + str(resSize) + " --modeldir '" + str(
            modeldir) + "' --logdir '" + str(logdir) + \
                  "' --classifier_modeldir '" + str(classifier_modeldir) + "' --classifier_checkpoint " + str(
            classifier_checkpoint) + \
                  " --dataroot " + str(dataroot) + "")

        nclass_all = 200

        # Evaluate
        os.system("python f-CLSWGAN/evaluate.py --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + " --image_embedding " + \
                  str(image_embedding) + " --split_no '" + str(split_no) + "' --syn_att '" + str(
            syn_att) + "' --nclass_all " + str(nclass_all))

    elif dataset == "SUN":

        [dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no,
         cls_weight, \
         syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz] = f_CLSWGAN_Helper(dataroot, image_embedding,
                                                                                            class_embedding, \
                                                                                            batch_size, resSize,
                                                                                            attSize, nepoch, lr, beta1,
                                                                                            split_no,
                                                                                            cls_weight, syn_num, ngh,
                                                                                            ndh, lambda1,
                                                                                            classifier_checkpoint,
                                                                                            syn_att, nz,
                                                                                            default_resSize=2048,
                                                                                            default_attSize=102,
                                                                                            default_batch_size=100,
                                                                                            default_nepoch=50,
                                                                                            default_lr=0.0001,
                                                                                            default_beta1=0.5,
                                                                                            default_split_no='',
                                                                                            default_cls_weight=0.01,
                                                                                            default_syn_num=300,
                                                                                            default_ngh=4096,
                                                                                            default_ndh=4096,
                                                                                            default_lambda1=10,
                                                                                            default_classifier_checkpoint=49,
                                                                                            default_syn_att='_',
                                                                                            default_nz=102)

        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_classifier_" + str(dataset))

        # Train classifier
        os.system("python f-CLSWGAN/train_cls.py --manualSeed 9182 --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + \
                  " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --split_no '" + str(split_no) + \
                  "' --attSize " + str(attSize) + " --resSize " + str(resSize) + " --nepoch " + str(
            nepoch) + " --modeldir '" + str(modeldir) + \
                  "' --beta1 " + str(beta1) + " --logdir '" + str(logdir) + "'")

        classifier_modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_" + str(dataset))

        # Train WGAN
        os.system("python f-CLSWGAN/train_wgan.py --manualSeed 9182 --dataset " + str(dataset) + " --split_no '" + str(
            split_no) + \
                  "' --syn_att '" + str(syn_att) + "' --cls_weight " + str(
            cls_weight) + " --preprocessing --val_every 1 " + \
                  "--lr " + str(lr) + " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + \
                  " --netG_name MLP_G --netD_name MLP_CRITIC --nepoch " + str(nepoch) + " --syn_num " + str(
            syn_num) + " --ngh " + str(ngh) + \
                  " --ndh " + str(ndh) + " --lambda1 " + str(lambda1) + " --critic_iter 5 --batch_size " + str(
            batch_size) + " --nz " + str(nz) + \
                  " --attSize " + str(attSize) + " --resSize " + str(resSize) + " --modeldir '" + str(
            modeldir) + "' --logdir '" + str(logdir) + \
                  "' --classifier_modeldir '" + str(classifier_modeldir) + "' --classifier_checkpoint " + str(
            classifier_checkpoint) + \
                  " --dataroot " + str(dataroot) + "")

        nclass_all = 717

        # Evaluate
        os.system("python f-CLSWGAN/evaluate.py --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + " --image_embedding " + \
                  str(image_embedding) + " --split_no '" + str(split_no) + "' --syn_att '" + str(
            syn_att) + "' --nclass_all " + str(nclass_all))

    elif dataset == "APY":

        [dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no,
         cls_weight, \
         syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz] = f_CLSWGAN_Helper(dataroot, image_embedding,
                                                                                            class_embedding, \
                                                                                            batch_size, resSize,
                                                                                            attSize, nepoch, lr, beta1,
                                                                                            split_no,
                                                                                            cls_weight, syn_num, ngh,
                                                                                            ndh, lambda1,
                                                                                            classifier_checkpoint,
                                                                                            syn_att, nz,
                                                                                            default_resSize=2048,
                                                                                            default_attSize=64,
                                                                                            default_batch_size=100,
                                                                                            default_nepoch=50,
                                                                                            default_lr=0.0001,
                                                                                            default_beta1=0.5,
                                                                                            default_split_no='',
                                                                                            default_cls_weight=0.01,
                                                                                            default_syn_num=300,
                                                                                            default_ngh=4096,
                                                                                            default_ndh=4096,
                                                                                            default_lambda1=10,
                                                                                            default_classifier_checkpoint=49,
                                                                                            default_syn_att='_',
                                                                                            default_nz=64)

        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_classifier_" + str(dataset))

        # Train classifier
        os.system("python f-CLSWGAN/train_cls.py --manualSeed 9182 --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + \
                  " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --split_no '" + str(split_no) + \
                  "' --attSize " + str(attSize) + " --resSize " + str(resSize) + " --nepoch " + str(
            nepoch) + " --modeldir '" + str(modeldir) + \
                  "' --beta1 " + str(beta1) + " --logdir '" + str(logdir) + "'")

        classifier_modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset))
        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_" + str(dataset))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_" + str(dataset))

        # Train WGAN
        os.system("python f-CLSWGAN/train_wgan.py --manualSeed 9182 --dataset " + str(dataset) + " --split_no '" + str(
            split_no) + \
                  "' --syn_att '" + str(syn_att) + "' --cls_weight " + str(
            cls_weight) + " --preprocessing --val_every 1 " + \
                  "--lr " + str(lr) + " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + \
                  " --netG_name MLP_G --netD_name MLP_CRITIC --nepoch " + str(nepoch) + " --syn_num " + str(
            syn_num) + " --ngh " + str(ngh) + \
                  " --ndh " + str(ndh) + " --lambda1 " + str(lambda1) + " --critic_iter 5 --batch_size " + str(
            batch_size) + " --nz " + str(nz) + \
                  " --attSize " + str(attSize) + " --resSize " + str(resSize) + " --modeldir '" + str(
            modeldir) + "' --logdir '" + str(logdir) + \
                  "' --classifier_modeldir '" + str(classifier_modeldir) + "' --classifier_checkpoint " + str(
            classifier_checkpoint) + \
                  " --dataroot " + str(dataroot) + "")

        nclass_all = 32

        # Evaluate
        os.system("python f-CLSWGAN/evaluate.py --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + " --image_embedding " + \
                  str(image_embedding) + " --split_no '" + str(split_no) + "' --syn_att '" + str(
            syn_att) + "' --nclass_all " + str(nclass_all))

    elif dataset == "LAD":

        [dataroot, image_embedding, class_embedding, resSize, attSize, batch_size, nepoch, lr, beta1, split_no,
         cls_weight, \
         syn_num, ngh, ndh, lambda1, classifier_checkpoint, syn_att, nz] = f_CLSWGAN_Helper(dataroot, image_embedding,
                                                                                            class_embedding, \
                                                                                            batch_size, resSize,
                                                                                            attSize, nepoch, lr, beta1,
                                                                                            split_no,
                                                                                            cls_weight, syn_num, ngh,
                                                                                            ndh, lambda1,
                                                                                            classifier_checkpoint,
                                                                                            syn_att, nz,
                                                                                            default_resSize=2048,
                                                                                            default_attSize=359,
                                                                                            default_batch_size=100,
                                                                                            default_nepoch=50,
                                                                                            default_lr=0.0001,
                                                                                            default_beta1=0.5,
                                                                                            default_split_no='_0',
                                                                                            default_cls_weight=0.01,
                                                                                            default_syn_num=300,
                                                                                            default_ngh=4096,
                                                                                            default_ndh=4096,
                                                                                            default_lambda1=10,
                                                                                            default_classifier_checkpoint=49,
                                                                                            default_syn_att='_',
                                                                                            default_nz=359)

        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset) + str(split_no))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_classifier_" + str(dataset) + str(split_no))

        # Train classifier
        os.system("python f-CLSWGAN/train_cls.py --manualSeed 9182 --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + \
                  " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --split_no '" + str(split_no) + \
                  "' --attSize " + str(attSize) + " --resSize " + str(resSize) + " --nepoch " + str(
            nepoch) + " --modeldir '" + str(modeldir) + \
                  "' --beta1 " + str(beta1) + " --logdir '" + str(logdir) + "'")

        classifier_modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_classifier_" + str(dataset) + str(split_no))
        modeldir = os.path.join(os.getcwd(), "f-CLSWGAN/models_" + str(dataset) + str(split_no))
        logdir = os.path.join(os.getcwd(), "f-CLSWGAN/logs_" + str(dataset) + str(split_no))

        # Train WGAN
        os.system("python f-CLSWGAN/train_wgan.py --manualSeed 9182 --dataset " + str(dataset) + " --split_no '" + str(
            split_no) + \
                  "' --syn_att '" + str(syn_att) + "' --cls_weight " + str(
            cls_weight) + " --preprocessing --val_every 1 " + \
                  "--lr " + str(lr) + " --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + \
                  " --netG_name MLP_G --netD_name MLP_CRITIC --nepoch " + str(nepoch) + " --syn_num " + str(
            syn_num) + " --ngh " + str(ngh) + \
                  " --ndh " + str(ndh) + " --lambda1 " + str(lambda1) + " --critic_iter 5 --batch_size " + str(
            batch_size) + " --nz " + str(nz) + \
                  " --attSize " + str(attSize) + " --resSize " + str(resSize) + " --modeldir '" + str(
            modeldir) + "' --logdir '" + str(logdir) + \
                  "' --classifier_modeldir '" + str(classifier_modeldir) + "' --classifier_checkpoint " + str(
            classifier_checkpoint) + \
                  " --dataroot " + str(dataroot) + "")

        nclass_all = 230

        # Evaluate
        os.system("python f-CLSWGAN/evaluate.py --dataset " + str(dataset) + " --dataroot " + str(
            dataroot) + " --image_embedding " + \
                  str(image_embedding) + " --split_no '" + str(split_no) + "' --syn_att '" + str(
            syn_att) + "' --nclass_all " + str(nclass_all))


######################################################################################
#   TF-VAEGAN
######################################################################################

def TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                     dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight,
                     feed_lr, dec_lr, feedback_loop, a1, a2, default_resSize, default_attSize, default_batch_size,
                     default_nepoch,
                     default_lr, default_syn_num, default_ngh, default_ndh, default_lambda1, default_nz, default_gammaD,
                     default_gammaG,
                     default_nclass_all, default_split, default_latent_size, default_classifier_lr,
                     default_recons_weight, default_feed_lr, default_dec_lr, default_feedback_loop, default_a1,
                     default_a2):
    dataroot = dataroot if dataroot is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                     "/xlsa17/data"
    image_embedding = image_embedding if image_embedding is not None else "res101"
    class_embedding = class_embedding if class_embedding is not None else "att"
    resSize = resSize if resSize is not None else default_resSize
    attSize = attSize if attSize is not None else default_attSize
    batch_size = batch_size if batch_size is not None else default_batch_size
    nepoch = nepoch if nepoch is not None else default_nepoch
    lr = lr if lr is not None else default_lr
    syn_num = syn_num if syn_num is not None else default_syn_num
    ngh = ngh if ngh is not None else default_ngh
    ndh = ndh if ndh is not None else default_ndh
    lambda1 = lambda1 if lambda1 is not None else default_lambda1
    nz = nz if nz is not None else default_nz
    gammaD = gammaD if gammaD is not None else default_gammaD
    gammaG = gammaG if gammaG is not None else default_gammaG
    nclass_all = nclass_all if nclass_all is not None else default_nclass_all
    split = split if split is not None else default_split
    latent_size = latent_size if latent_size is not None else default_latent_size
    classifier_lr = classifier_lr if classifier_lr is not None else default_classifier_lr
    recons_weight = recons_weight if recons_weight is not None else default_recons_weight
    feed_lr = feed_lr if feed_lr is not None else default_feed_lr
    dec_lr = dec_lr if dec_lr is not None else default_dec_lr
    feedback_loop = feedback_loop if feedback_loop is not None else default_feedback_loop
    a1 = a1 if a1 is not None else default_a1
    a2 = a2 if a2 is not None else default_a2

    return gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
           dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
           feed_lr, dec_lr, feedback_loop, a1, a2


def TF_VAEGAN(dataset=None, gammaD=None, gammaG=None, image_embedding=None, class_embedding=None, nepoch=None,
              syn_num=None, ngh=None, ndh=None, lambda1=None, nclass_all=None, \
              dataroot=None, split=None, batch_size=None, nz=None, latent_size=None, attSize=None, resSize=None,
              lr=None, classifier_lr=None, recons_weight=None, \
              feed_lr=None, dec_lr=None, feedback_loop=None, a1=None, a2=None):
    if dataset == "AWA2":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=85,
                                                                    default_batch_size=64, default_nepoch=120,
                                                                    default_lr=0.00001, default_syn_num=2400,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=10, default_nz=85,
                                                                    default_gammaD=10, default_gammaG=10,
                                                                    default_nclass_all=50, default_split="",
                                                                    default_latent_size=85, default_classifier_lr=0.001,
                                                                    default_recons_weight=0.1, default_feed_lr=0.0001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=0.01, default_a2=0.01)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")

    if dataset == "AWA1":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=85,
                                                                    default_batch_size=64, default_nepoch=120,
                                                                    default_lr=0.00001, default_syn_num=1800,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=10, default_nz=85,
                                                                    default_gammaD=10, default_gammaG=10,
                                                                    default_nclass_all=50, default_split="",
                                                                    default_latent_size=85, default_classifier_lr=0.001,
                                                                    default_recons_weight=0.1, default_feed_lr=0.0001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=0.01, default_a2=0.01)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")

    if dataset == "SUN":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=102,
                                                                    default_batch_size=64, default_nepoch=400,
                                                                    default_lr=0.00001, default_syn_num=400,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=1, default_nz=102,
                                                                    default_gammaD=1, default_gammaG=10,
                                                                    default_nclass_all=717, default_split="",
                                                                    default_latent_size=102,
                                                                    default_classifier_lr=0.0005,
                                                                    default_recons_weight=0.01, default_feed_lr=0.0001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=0.1, default_a2=0.01)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")

    if dataset == "CUB":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=312,
                                                                    default_batch_size=64, default_nepoch=300,
                                                                    default_lr=0.0001, default_syn_num=2400,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=10, default_nz=312,
                                                                    default_gammaD=10, default_gammaG=10,
                                                                    default_nclass_all=200, default_split="",
                                                                    default_latent_size=312,
                                                                    default_classifier_lr=0.001,
                                                                    default_recons_weight=0.01, default_feed_lr=0.00001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=1, default_a2=1)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")

    if dataset == "APY":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=64,
                                                                    default_batch_size=64, default_nepoch=200,
                                                                    default_lr=0.00001, default_syn_num=300,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=10, default_nz=64,
                                                                    default_gammaD=10, default_gammaG=10,
                                                                    default_nclass_all=32, default_split="",
                                                                    default_latent_size=64, default_classifier_lr=0.001,
                                                                    default_recons_weight=0.1, default_feed_lr=0.0001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=0.01, default_a2=0.01)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")

    if dataset == "LAD":
        [gammaD, gammaG, image_embedding, class_embedding, nepoch, syn_num, ngh, ndh, lambda1, nclass_all, \
         dataroot, split, batch_size, nz, latent_size, attSize, resSize, lr, classifier_lr, recons_weight, \
         feed_lr, dec_lr, feedback_loop, a1, a2] = TF_VAEGAN_helper(gammaD, gammaG, image_embedding, class_embedding,
                                                                    nepoch, syn_num, ngh, ndh, lambda1, nclass_all,
                                                                    dataroot, split, batch_size, nz, latent_size,
                                                                    attSize, resSize, lr, classifier_lr, recons_weight,
                                                                    feed_lr, dec_lr, feedback_loop, a1, a2,
                                                                    default_resSize=2048, default_attSize=359,
                                                                    default_batch_size=64, default_nepoch=100,
                                                                    default_lr=0.00001, default_syn_num=1800,
                                                                    default_ngh=4096, default_ndh=4096,
                                                                    default_lambda1=10, default_nz=359,
                                                                    default_gammaD=10, default_gammaG=10,
                                                                    default_nclass_all=230, default_split="_0",
                                                                    default_latent_size=359,
                                                                    default_classifier_lr=0.001,
                                                                    default_recons_weight=0.1, default_feed_lr=0.0001,
                                                                    default_dec_lr=0.0001, default_feedback_loop=2,
                                                                    default_a1=0.01, default_a2=0.01)

        os.system("python TF-VAEGAN/tfvaegan-adapted/train_images.py --gammaD " + str(gammaD) + " --gammaG " + str(
            gammaG) + " --gzsl --encoded_noise --manualSeed 9182 " + \
                  "--preprocessing --image_embedding " + str(image_embedding) + " --class_embedding " + str(
            class_embedding) + " --nepoch " + str(nepoch) + \
                  " --syn_num " + str(syn_num) + " --ngh " + str(ngh) + " --ndh " + str(ndh) + " --lambda1 " + str(
            lambda1) + " --critic_iter 5 --nclass_all " + str(nclass_all) + \
                  " --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + " --split '" + str(
            split) + "' --batch_size " + str(batch_size) + \
                  " --nz " + str(nz) + " --latent_size " + str(latent_size) + " --attSize " + str(
            attSize) + " --resSize " + str(resSize) + \
                  " --lr " + str(lr) + " --classifier_lr " + str(classifier_lr) + " --recons_weight " + str(
            recons_weight) + \
                  " --freeze_dec --feed_lr " + str(feed_lr) + " --dec_lr " + str(dec_lr) + " --feedback_loop " + str(
            feedback_loop) + " --a1 " + str(a1) + " --a2 " + str(a2) + "")


#############################################################################################
# CE_GZSL
#############################################################################################

def CE_GZSL_Helper(dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
                   embedSize, syn_num, outzSize,
                   nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_split, lr,
                   default_lr, default_batch_size, default_nepoch, default_attSize, default_resSize, default_nz,
                   default_embedSize,
                   default_syn_num, default_outzSize, default_nhF, default_ins_weight, default_cls_weight,
                   default_ins_temp, default_cls_temp, default_nclass_all, default_nclass_seen):
    dataroot = dataroot if dataroot is not None else "/home/cristianopatricio/Documents/Datasets" \
                                                     "/xlsa17/data"
    image_embedding = image_embedding if image_embedding is not None else "res101"
    class_embedding = class_embedding if class_embedding is not None else "att"
    split = split if split is not None else default_split
    lr = lr if lr is not None else default_lr
    batch_size = batch_size if batch_size is not None else default_batch_size
    nepoch = nepoch if nepoch is not None else default_nepoch
    attSize = attSize if attSize is not None else default_attSize
    resSize = resSize if resSize is not None else default_resSize
    nz = nz if nz is not None else default_nz
    embedSize = embedSize if embedSize is not None else default_embedSize
    syn_num = syn_num if syn_num is not None else default_syn_num
    outzSize = outzSize if outzSize is not None else default_outzSize
    nhF = nhF if nhF is not None else default_nhF
    ins_weight = ins_weight if ins_weight is not None else default_ins_weight
    cls_weight = cls_weight if cls_weight is not None else default_cls_weight
    ins_temp = ins_temp if ins_temp is not None else default_ins_temp
    cls_temp = cls_temp if cls_temp is not None else default_cls_temp
    nclass_all = nclass_all if nclass_all is not None else default_nclass_all
    nclass_seen = nclass_seen if nclass_seen is not None else default_nclass_seen

    return dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize, \
           syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen


def CE_GZSL(dataset=None, dataroot=None, image_embedding=None, class_embedding=None, split=None, batch_size=None,
            nepoch=None, attSize=None, resSize=None, nz=None, lr=None,
            embedSize=None, syn_num=None, outzSize=None,
            nhF=None, ins_weight=None, cls_weight=None, ins_temp=None, cls_temp=None, nclass_all=None,
            nclass_seen=None):
    """
    --dataset: datasets, e.g: CUB.
    --class_embedding: the semantic descriptors to use, e.g: sent or att.
    --syn_num: number synthetic features for each class.
    --batch_size: the number of the instances in a mini-batch.
    --attSize: size of semantic features.
    --nz: size of the Gaussian noise.
    --embedSize: size of embedding h.
    --outzSize: size of non-liner projection z.
    --nhF: size of the hidden units comparator network F.
    --ins_weight: weight of the classification loss when learning G.
    --cls_weight: weight of the score function when learning G.
    --ins_temp: temperature in instance-level supervision.
    --cls_temp: temperature in class-level supervision
    --manualSeed: manual seed.
    --nclass_all: number of all classes.
    --nclass_seen: number of seen classes
    :return:
    """

    if dataset == "AWA2":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="",
            default_batch_size=4096, default_nepoch=100, default_attSize=85, default_resSize=2048, default_nz=85,
            default_embedSize=2048,
            default_syn_num=2400, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=10.0, default_cls_temp=1.0, default_nclass_all=50, default_nclass_seen=40)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")

    if dataset == "AWA1":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="",
            default_batch_size=4096, default_nepoch=100, default_attSize=85, default_resSize=2048, default_nz=85,
            default_embedSize=2048,
            default_syn_num=1800, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=0.1, default_cls_temp=0.1, default_nclass_all=50, default_nclass_seen=40)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")

    if dataset == "SUN":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="",
            default_batch_size=1024, default_nepoch=100, default_attSize=102, default_resSize=2048, default_nz=102,
            default_embedSize=2048,
            default_syn_num=100, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=0.1, default_cls_temp=0.1, default_nclass_all=717, default_nclass_seen=635)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")

    if dataset == "CUB":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="",
            default_batch_size=2048, default_nepoch=100, default_attSize=312, default_resSize=2048, default_nz=312,
            default_embedSize=2048,
            default_syn_num=300, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=0.1, default_cls_temp=0.1, default_nclass_all=200, default_nclass_seen=150)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")

    if dataset == "APY":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="",
            default_batch_size=1024, default_nepoch=100, default_attSize=64, default_resSize=2048, default_nz=64,
            default_embedSize=2048,
            default_syn_num=1800, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=0.1, default_cls_temp=0.1, default_nclass_all=32, default_nclass_seen=20)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")

    if dataset == "LAD":
        [dataroot, image_embedding, class_embedding, split, lr, batch_size, nepoch, attSize, resSize, nz, embedSize,
         syn_num, outzSize, nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen] = CE_GZSL_Helper(
            dataroot, image_embedding, class_embedding, split, batch_size, nepoch, attSize, resSize, nz,
            embedSize, syn_num, outzSize,
            nhF, ins_weight, cls_weight, ins_temp, cls_temp, nclass_all, nclass_seen, default_lr=0.0001,
            default_split="_0",
            default_batch_size=1024, default_nepoch=100, default_attSize=359, default_resSize=2048, default_nz=359,
            default_embedSize=2048,
            default_syn_num=1800, default_outzSize=512, default_nhF=2048, default_ins_weight=0.001,
            default_cls_weight=0.001,
            default_ins_temp=0.1, default_cls_temp=0.1, default_nclass_all=230, default_nclass_seen=181)

        os.system(
            "python CE-GZSL/CE-GZSL-adapted/CE_GZSL.py --dataroot '" + str(dataroot) + "' --dataset " + str(dataset) + \
            " --image_embedding " + str(image_embedding) + " --class_embedding " + str(class_embedding) + \
            " --split '" + str(split) + "' --batch_size " + str(batch_size) + " --nepoch " + str(
                nepoch) + " --attSize " + str(attSize) + " --lr " + str(lr) + \
            " --resSize " + str(resSize) + " --nz " + str(nz) + " --embedSize " + str(embedSize) + " --syn_num " + str(
                syn_num) + \
            " --outzSize " + str(outzSize) + " --nhF " + str(nhF) + " --ins_weight " + str(
                ins_weight) + " --cls_weight " + str(cls_weight) + \
            " --ins_temp " + str(ins_temp) + " --cls_temp " + str(cls_temp) + " --manualSeed 3483 --nclass_all " + str(
                nclass_all) + \
            " --nclass_seen " + str(nclass_seen) + "")


###################################################################################
#   EVALUATE LAD
###################################################################################

def compute_zsl_acc_lad(split, att_split, preds_file):
    """
    Evaluate LAD in ZSL setting
    :param split: ZSL split to be evaluated (format: 0,1,2,3 ou 4).
    :param att_split: Name of the att_split file (e.g. att_splits_0).
    :param preds_file: Filename containing the predictions (e.g. preds_SAE_att_0.txt).
    :return:
    """

    os.system(
        "python utils/acc_lad_dataset.py --split " + str(split) + " --att_split " + str(att_split) + " --preds_file " + str(
            preds_file) + "")


def compute_harmonic_acc_lad(split, att_split, preds_seen_file, preds_unseen_file):
    """

    :param split:  ZSL split to be evaluated (format: 0,1,2,3 ou 4).
    :param att_split: Name of the att_split file (e.g. att_splits_0).
    :param preds_seen_file: Filename containing the seen predictions (e.g. preds_seen_ESZSL_att_0.txt).
    :param preds_unseen_file: Filename containing the unseen predictions (e.g. preds_unseen_ESZSL_att_0.txt).
    :return:
    """

    if "ESZSL" in preds_seen_file:

        os.system("python utils/harmonic_lad_ESZSL.py --split " + str(split) + " --att_split " + str(
            att_split) + " --preds_seen_file " + str(preds_seen_file) + " --preds_unseen_file " + str(
            preds_unseen_file) + " --seen")
        os.system("python harmonic_lad_ESZSL.py --split " + str(split) + " --att_split " + str(
            att_split) + " --preds_seen_file " + str(preds_seen_file) + " --preds_unseen_file " + str(
            preds_unseen_file) + "")
    else:

        os.system("python utils/harmonic_lad.py --split " + str(split) + " --att_split " + str(
            att_split) + " --preds_seen_file " + str(preds_seen_file) + " --preds_unseen_file " + str(
            preds_unseen_file) + " --seen")
        os.system("python utils/harmonic_lad.py --split " + str(split) + " --att_split " + str(
            att_split) + " --preds_seen_file " + str(preds_seen_file) + " --preds_unseen_file " + str(
            preds_unseen_file) + "")


def compute_harmonic_SAE_LAD(split, att_split, preds):
    """

    :param split: ZSL split to be evaluated (format: 0,1,2,3 ou 4).
    :param att_split: Name of the att_split file (e.g. att_splits_0).
    :param preds: Name of the predictions file (e.g. preds_SAE_GZSL_Att_0.txt).
    :return:
    """
    os.system("python utils/evaluate_LAD_GZSL_SAE.py --split " + str(split) + " --att_split " + str(
        att_split) + " --preds " + str(preds) + "")


def evaluate_LAD_ZSL(filename):
    """

    :param filename: name of the file containing the ZSL results.
    :return:
    """
    os.system("python utils/evaluate_lad_zsl.py --filename " + str(filename) + "")


def evaluate_LAD_GZSL(filename_seen, filename_unseen):
    """

    :param filename_seen: name of the file containing the results of the seen classes.
    :param filename_unseen: name of the file containing the results of the unseen classes.
    :return:
    """
    os.system("python utils/evaluate_gzsl_lad.py --filename_seen " + str(filename_seen) + " --filename_unseen " + str(
        filename_unseen) + "")
