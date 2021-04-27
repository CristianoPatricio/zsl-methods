import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--download_mode", action='store_true', default=False)
parser.add_argument("--train_classifier", action='store_true', default=False)
parser.add_argument("--train_WGAN", action='store_true', default=False)

if __name__ == '__main__':
    # TODO: Change tensorflow import at classifier.py, clswgan.py and classifier2.py to \\
    #  'import tensorflow.compat.v1 as tf \\
    #  tf.compat.v1.disable_v2_behavior()'

    args = parser.parse_args()
    download_mode = args.download_mode
    train_classifier = args.train_classifier
    tran_WGAN = args.train_WGAN

    repo_path = './Feature-Generating-Networks-for-ZSL-master'

    if download_mode:
        # Give 'x' permission to file
        subprocess.run(["chmod", "u+x", "./download_repo.sh"])
        # Exec bash script
        subprocess.run(["./download_repo.sh"])

    if os.path.exists(repo_path):

        if train_classifier:
            ########################################################################
            #   Arguments of classifier.py
            ########################################################################
            manualSeed = 9182
            lr = 0.001
            image_embedding = 'res101'
            class_embedding = 'att'
            nepoch = 50
            dataset = 'AWA1'
            batch_size = 100
            attSize = 85
            resSize = 2048
            modeldir = repo_path + '/models_classifier'
            logdir = repo_path + '/logs_classifier'
            dataroot = '/home/cristianopatricio/Documents/Datasets/xlsa17/data'

            # Exec python script
            subprocess.run(
                ["python3", str(repo_path) + "/classifier.py", "--manualSeed", str(manualSeed), "--preprocessing",
                 "--lr", str(lr), "--image_embedding", str(image_embedding), "--class_embedding", str(class_embedding),
                 "--nepoch", str(nepoch), "--dataset", str(dataset), "--batch_size", str(batch_size), "--attSize",
                 str(attSize), "--resSize", str(resSize), "--modeldir", str(modeldir), "--logdir", str(logdir),
                 "--dataroot", str(dataroot)])

        if tran_WGAN:
            ########################################################################
            #   Arguments of clswgan.py
            ########################################################################
            manualSeed = 9182
            cls_weight = 0.01
            val_every = 1
            lr = 0.00001
            image_embedding = 'res101'
            class_embedding = 'att'
            netG_name = 'MLP_G'
            netD_name = 'MLP_CRITIC'
            nepoch = 30
            syn_num = 300
            ngh = 4096
            ndh = 4096
            lambda1 = 10
            critic_iter = 5
            dataset = 'AWA1'
            batch_size = 64
            nz = 85
            attSize = 85
            resSize = 2048
            modeldir = repo_path + '/models_AWA1'
            logdir = repo_path + '/logs_AWA1'
            dataroot = '/home/cristianopatricio/Documents/Datasets/xlsa17/data'
            classifier_modeldir = repo_path + '/models_classifier'
            classifier_checkpoint = 49

            # Exec python script
            subprocess.run(
                ["python3", str(repo_path) + "/clswgan.py", "--manualSeed", str(manualSeed), "--cls_weight",
                 str(cls_weight),
                 "--preprocessing", "--val_every", str(val_every), "--lr", str(lr), "--image_embedding",
                 str(image_embedding),
                 "--class_embedding", str(class_embedding), "--netG_name", str(netG_name), "--netD_name",
                 str(netG_name),
                 "--nepoch", str(nepoch), "--syn_num", str(syn_num), "--ngh", str(ngh), "--ndh", str(ndh), "--lambda1",
                 str(lambda1),
                 "--critic_iter", str(critic_iter), "--dataset", str(dataset), "--batch_size", str(batch_size), "--nz",
                 str(nz),
                 "--attSize", str(attSize), "--resSize", str(resSize), "--modeldir", str(modeldir), "--logdir",
                 str(logdir), "--dataroot",
                 str(dataroot), "--classifier_modeldir", str(classifier_modeldir), "--classifier_checkpoint",
                 str(classifier_checkpoint)])

    else:
        print(f"[WARNING]: Run the script with --download_mode.")