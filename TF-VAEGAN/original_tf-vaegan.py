import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--download_mode", action='store_true', default=False)
parser.add_argument("--train", action='store_true', default=False)

if __name__ == '__main__':
    # TODO:
    #   # Create Conda Env
    #   conda create -n tfvaegan python=3.6 \\
    #   conda activate tfvaegan \\
    #   pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \\
    #   pip install torchvision==0.2.0 scikit-learn==0.22.1 scipy==1.4.1 h5py==2.10 numpy==1.18.1 \\
    #   # Fix issues
    #   Change import util to import datasets.image_util as util on line 7 in classifiers/classifier_images.py

    args = parser.parse_args()
    download_mode = args.download_mode
    train = args.train

    repo_path = 'tfvaegan-master'

    if download_mode:
        # Give 'x' permission to file
        subprocess.run(["chmod", "u+x", "./download_repo.sh"])
        # Exec bash script
        subprocess.run(["./download_repo.sh"])

    if os.path.exists(repo_path):

        if train:
            ########################################################################
            #   Arguments of classifier.py
            ########################################################################
            CUDA_VISIBLE_DEVICES = 0
            OMP_NUM_THREADS = 4
            gammaD = 10
            gammaG = 10
            gzsl = True
            encoded_noise = True
            manualSeed = 9182
            preprocessing = True
            cuda = True
            image_embedding = 'res101'
            class_embedding = 'att'
            nepoch = 120
            syn_num = 1800
            ngh = 4096
            ndh = 4096
            lambda1 = 10
            critic_iter = 5
            nclass_all = 50
            dataroot = '/home/cristianopatricio/Documents/Datasets/xlsa17/data'
            dataset = 'AWA2'
            batch_size = 64
            nz = 85
            latent_size = 85
            attSize = 85
            resSize = 2048
            lr = 0.00001
            classifier_lr = 0.001
            recons_weight = 0.1
            freeze_dec = 'True'
            feed_lr = 0.0001
            dec_lr = 0.0001
            feedback_loop = 2
            a1 = 0.01
            a2 = 0.01

            # Exec python script
            subprocess.run(
                ["python3", str(repo_path) + "/train_images.py", "--gammaD", str(gammaD), "--gammaG", str(gammaG),
                 "--gzsl" if gzsl else "" "--encoded_noise" if encoded_noise else "", "--manualSeed", str(manualSeed),
                 "--preprocessing" if preprocessing else "" "--cuda" if cuda else "" "--image_embedding",
                 str(image_embedding),
                 "--class_embedding", str(class_embedding), "--nepoch", str(nepoch), "--syn_num", str(syn_num), "--ngh",
                 str(ngh), "--ndh", str(ndh), "--lambda1", str(lambda1), "--critic_iter", str(critic_iter),
                 "--nclass_all",
                 str(nclass_all), "--dataroot", str(dataroot), "--dataset", str(dataset), "--batch_size",
                 str(batch_size),
                 "--nz", str(nz), "--latent_size", str(latent_size), "--attSize", str(attSize), "--resSize",
                 str(resSize),
                 "--lr", str(lr), "--classifier_lr", str(classifier_lr), "--recons_weight", str(recons_weight),
                 "--freeze_dec" if freeze_dec else "" "--feed_lr", str(feed_lr), "--dec_lr", str(dec_lr),
                 "--feedback_loop", str(feedback_loop), "--a1", str(a1), "--a2", str(a2)])

    else:
        print(f"[WARNING]: Run the script with --download_mode.")
