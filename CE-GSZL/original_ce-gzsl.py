import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--download_mode", action='store_true', default=False)
parser.add_argument("--train", action='store_true', default=False)

if __name__ == '__main__':
    # TODO:
    #   # Create Conda Env
    #   conda create -n ce-gzsl python=3.6 \\
    #   conda activate ce-gzsl \\
    #   conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch \\
    #   pip install h5py \\

    args = parser.parse_args()
    download_mode = args.download_mode
    train = args.train

    repo_path = 'CE-GZSL-master'

    if download_mode:
        # Give 'x' permission to file
        subprocess.run(["chmod", "u+x", "./download_repo.sh"])
        # Exec bash script
        subprocess.run(["./download_repo.sh"])

    if os.path.exists(repo_path):

        if train:
            ########################################################################
            #   Arguments of CE_GZSL.py
            ########################################################################
            dataset = 'CUB'
            class_embedding = 'att'
            image_embedding = 'res101'
            syn_num = 100
            batch_size = 2048
            attSize = 312
            nz = 312
            embedSize = 2048
            outzSize = 512
            nhF = 2048
            ins_weight = 0.001
            cls_weight = 0.001
            ins_temp = 0.1
            cls_temp = 0.1
            cls_weight = 3483
            nclass_all = 200
            nclass_seen = 150
            dataroot = '/home/cristianopatricio/Documents/Datasets/xlsa17/data'

            # Exec python script
            subprocess.run(
                ["python3", str(repo_path) + "/CE_GZSL.py", "--dataset", str(dataset), "--class_embedding", str(class_embedding),
                 "--image_embedding", str(image_embedding), "--syn_num", str(syn_num), "--batch_size", str(batch_size),
                 "--attSize", str(attSize), "--nz", str(nz), "--embedSize", str(embedSize), "--outzSize", str(outzSize),
                 "--nhF", str(nhF), "--ins_weight", str(ins_weight), "--cls_weight", str(cls_weight), "--ins_temp", str(ins_temp),
                 "--cls_temp", str(cls_temp), "--cls_weight", str(cls_weight), "--nclass_all", str(nclass_all),
                 "--nclass_seen", str(nclass_seen), "--dataroot", str(dataroot)])

    else:
        print(f"[WARNING]: Run the script with --download_mode.")
