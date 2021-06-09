# *Evaluation Framework for Zero-Shot Learning Methods*

---

## 1. Datasets

The datasets used to evaluate the ZSL methods can be downloaded [here](http://socia-lab.di.ubi.pt/~cristiano_patricio/data/zsl-datasets.zip). 

| Dataset | No. Classes | No. Instances | No. Attributes |
|:--------|:-----------:|:-------------:|:--------------:|
| AWA1 | 50 | 30,475 | 85 |
| AWA2 | 50 | 37,322 | 85 |
| CUB | 200 | 11,788 | 312 |
| SUN | 717 | 14,340 | 102 |
| APY | 32 | 15,339 | 64 |
| LAD | 230 | 78,017 | 359 |

## 2. Methods

We select six state-of-the-art ZSL methods, including projection-based methods (ESZSL, SAE, and DEM), and generative methods (f-CLSWGAN, TF-VAEGAN, and CE-GZSL). 

### 2.1 **ESZSL**

:page_facing_up: **Paper**: http://proceedings.mlr.press/v37/romera-paredes15.pdf

**Class**: `ESZSL(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, APY, LAD}
<dataset_path> : {'./datasets/}
<filename> : name of the features file (without the file extension)
<alpha> : int value [-3,3] 
<gamma> : int value [-3,3]
<att_split> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
ESZSL(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | Hyperparameter |
| ----------- | ----------- | 
| AWA1 | Alpha=3, Gamma=0 | 
| AWA2 | Alpha=3, Gamma=0 | 
| CUB | Alpha=2, Gamma=0 |
| SUN | Alpha=2, Gamma=2 | 
| APY | Alpha=3, Gamma=-1 |
| LAD | Alpha=3, Gamma=1 |

### 2.2 **SAE**

:page_facing_up: **Paper**: https://arxiv.org/pdf/1704.08345.pdf

**Class**: `SAE(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, APY, LAD}
<dataset_path> : {'./datasets/}
<filename> : name of the features file (without the file extension)
<lamb_ZSL> : float value, default=2
<lamb_GZSL> : float value, default=2
<setting> : Type of evaluation {V2S, S2V}
<att_split> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
SAE(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | Setting | Lambda (ZSL) | Lambda (GZSL) |
| ------- | :------:| :----------: | :-----------: |
| AWA1    |   V2S   |  3.0   |  3.2  | 
| AWA2    |   V2S   |  0.6   |  0.8  | 
| CUB     |   V2S   |  100   |  80   | 
| SUN     |   V2S   |  0.32  |  0.32 |
| aPY     |   V2S   |  2.0   |  0.16 |
| LAD     |   V2S   |  51.2   |  51.2 |

| Dataset | Setting | Lambda (ZSL) | Lambda (GZSL) | 
| ------- | :------:| :----------: | :-----------: |
| AWA1    |   S2V   |  0.8   |  0.8  | 
| AWA2    |   S2V   |  0.2   |  0.2  |
| CUB     |   S2V   |  0.2   |  0.2   |
| SUN     |   S2V   |  0.16  |  0.08 |
| aPY     |   S2V   |  4.0   |  2.56 | 
| LAD     |   S2V   |  6.4   |  6.4 | 

### 2.3 **DEM**

:page_facing_up: **Paper**: https://arxiv.org/pdf/1611.05088.pdf

**Class**: `DEM(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, aPY}
<dataset_path> : {'./datasets/}
<filename> : name of the features file (without the file extension)
<lamb> : float value, default=1e-3
<lr> : float value, default=1e-4
<batch_size> : batch size, default=64
<hidden_dim> : Dimension of the hidden layer, default=1600
<att_split> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
DEM(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | Hidden Dim | Lambda | Learning Rate |
| ------- | :------:| :----------: | :-----------: |
| AWA1    |   1600   |  1e-3   |  1e-4 | 
| AWA2    |   1600   |  1e-3   |  1e-4 | 
| CUB     |   1600   |  1e-2   |  1e-4 | 
| SUN     |   1600   |  1e-5   |  1e-4 |
| aPY     |   1600   |  1e-4  |   1e-4 | 
| LAD     |   1600   |  1e-4  |   1e-4 |

### 2.4 f-CLSWGAN

:page_facing_up: **Paper**: https://arxiv.org/pdf/1712.00981.pdf

* Original version (f-CLSWGAN/orig/)

**Run instructions**:
```bash
python original_f-CLSWGAN.py --download_mode
python original_f-CLSWGAN.py --train_classifier
python original_f-CLSWGAN.py --train_WGAN
```

* Modified version :new:

**Class**: `f_CLSWGAN(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, APY, LAD}
<dataroot> : {'./datasets/}
<image_embedding> : name of the features file (without the file extension)
<class_embedding> : name of the class embedding ("att" by default)
<split_no> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
<attSize> : size of the attribute annotations
<resSize> : size of the features
<nepoch> : number of epochs
<lr> : learning rate
<beta1> : beta1 for Adam optimizer
<batch_size> : input batch size
<cls_weight> : weight of the classification loss
<syn_num> : number of features to generate per class
<ngh> : size of the hidden units in generato
<ndh> : size of the hidden units in discriminator
<lambda1> : gradient penalty regularizer, following WGAN-GP
<classifier_checkpoint> : tells which ckpt file of tensorflow model to load
<nz> : size of the latent z vector
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
f_CLSWGAN(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | lambda1 | cls_weight |
| ------- | :------:| :---------:|
| AWA1    |   10   |  0.01   | 
| AWA2    |   10   |  0.01   | 
| CUB    |   10   |  0.01   | 
| SUN    |   10   |  0.01   | 
| APY    |   10   |  0.01   | 
| LAD    |   10   |  0.01   | 


### 2.5 TF-VAEGAN

:page_facing_up: **Paper**: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670477.pdf

#### Pre-requisites:
  
- Python 3.6
- Pytorch 0.3.1
- torchvision 0.2.0
- h5py 2.10
- scikit-learn 0.22.1
- scipy=1.4.1
- numpy 1.18.1
- numpy-base 1.18.1
- pillow 5.1.0

* **Original version**

_Note_: The requirements to run the script are present in the scope of the `main` function in `original_tf-vaegan.py`script.   

**Run instructions**:
```bash
python original_tf-vaegan.py --download_mode
python original_tf-vaegan.py --train
```

* **Modified version** :new:

**Class**: `TF_VAEGAN(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, APY, LAD}
<dataroot> : {'./datasets/}
<gammaD> : weight on the W-GAN loss
<gammaG> : weight on the W-GAN loss
<image_embedding> : name of the features file (without the file extension)
<class_embedding> :  name of the class embedding ("att" by default)
<syn_num> : number of features to generate per class
<ngh> : size of the hidden units in generator
<ndh> : size of the hidden units in discriminator
<lambda1> : gradient penalty regularizer, following WGAN-GP
<nclass_all> : number of all classes
<split> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
<batch_size> : input batch size
<nz> : size of the latent z vector
<latent_size> : size of the latent units in discriminator
<attSize> : size of semantic features
<resSize> : size of visual features
<lr> : learning rate to train GANs
<classifier_lr> : learning rate to train softmax classifier
<recons_weight> : recons_weight for decoder
<feed_lr> : learning rate to train GANs
<dec_lr> : learning rate to train GANs
<feedback_loop> : iterations on feedback loop
<a1> : weight of the feedback layers
<a2> : weight of the feedback layers
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
TF_VAEGAN(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | <span style="color:white;">Hid</span>lr<span style="color:white;">den</span> | syn_num | gammaD | gammaG | classifier_lr | recons_weight | feed_lr | dec_lr | a1 | a2 |
| ------ | :------: | :-------: | :--------: | :--------: | :-------: | :--------: | :------: | :------: | :------: | :------: |
| AWA1   |  1e-5  |  1800  | 10 | 10 | 1e-3 | 0.1 | 1e-4 | 1e-4 | 0.01 | 0.01 | 
| AWA2   |  1e-5  |  2400  | 10 | 10 | 13-e | 0.1 | 1e-4 | 1e-4 | 0.01 | 0.01 | 
| CUB    |  1e-4  |  2400  | 10 | 10 | 1e-3 | 0.01 | 1e-5 | 1e-4 | 1 | 1 | 
| SUN    |  1e-5  |  400  | 1 | 10 | 5e-4 | 0.01 | 1e-4 | 1e-4 | 0.1 | 0.01  |
| APY    |  1e-5  |  300  | 10 | 10 | 1e-3 | 0.1 | 1e-4 | 1e-4 | 0.01 | 0.01 |
| LAD    |  1e-5  |  1800 | 10 | 10 | 1e-3 | 0.1 | 1e-4 | 1e-4 | 0.01 | 0.01 | 

### 2.6 CE-GZSL

:page_facing_up: **Paper**: https://arxiv.org/pdf/2103.16173.pdf


#### Pre-requisites:
  
- Python 3.6
- Pytorch 1.2.0
- scikit-learn

* **Original version**:

_Note_: The requirements to run the script are present in the scope of the `main` function in `original_ce-gzsl.py`script.   

**Run instructions**:
```bash
python original_ce-gzsl.py --download_mode
python original_ce-gzsl.py --train
```

* **Modified version** :new:

**Class**: `CE_GZSL(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, APY, LAD}
<dataroot> : {'./datasets/}
<image_embedding> : name of the features file (without the file extension)
<class_embedding> :  name of the class embedding ("att" by default)
<split> : for the LAD dataset, specify which split is to be evaluated, in the following format "_{i}", i = {0,1,2,3,4}
<batch_size> : the number of the instances in a mini-batch
<nepoch> : number of epochs
<attSize> : size of semantic features
<resSize> : size of visual features
<nz> : size of the Gaussian noise
<lr> : learning rate to train GANs
<embedSize> : size of embedding h
<syn_num> : number synthetic features for each class
<outzSize> : size of non-liner projection z
<nhF> : size of the hidden units comparator network F
<ins_weight> : weight of the classification loss when learning G
<cls_weight> : weight of the score function when learning G
<ins_temp> : temperature in instance-level supervision
<cls_temp> : temperature in class-level supervision
<nclass_all> : number of all classes
<nclass_seen> : number of seen classes
```

**How to Run**:

In the main scope of ```main.py```, insert the following code:

```python
CE_GZSL(dataset="AWA1", filename="MobileNetV2")
```

**Hyperparameters**:

| Dataset | syn_num | ins_temp | cls_temp | batch-size |
| ------ | :------: | :-------: | :--------: | :--------: |
| AWA1   |  1800  |  0.1  | 0.1 | 4096 | 
| AWA2   |  2400  |  10  | 1 | 4096 |
| CUB    |  300  |  0.1  | 0.1 | 2048 |
| SUN    |  400  |  0.1  | 0.1 | 1024 |
| APY    |  300  |  0.1  | 0.1 | 1024 |
| LAD    |  1800 |  0.1 | 0.1  | 1024 |

## 3. Special Case: LAD dataset

LAD dataset must be evaluated on each of the five available splits (*att_splits_0.mat*, *att_splits_1.mat*, *att_splits_2.mat*, *att_splits_3.mat*, *att_splits_4.mat*). This means that each ZSL method should be executed for each of the provided splits.

For example, if we want to evaluate LAD dataset with SAE method, we need to run the following code:

```python
# Run SAE algorithm for LAD dataset
SAE(dataset="LAD", filename="MobileNetV2", att_split="_0")  # att_splits_0
SAE(dataset="LAD", filename="MobileNetV2", att_split="_1")  # att_splits_1
SAE(dataset="LAD", filename="MobileNetV2", att_split="_2")  # att_splits_2
SAE(dataset="LAD", filename="MobileNetV2", att_split="_3")  # att_splits_3
SAE(dataset="LAD", filename="MobileNetV2", att_split="_4")  # att_splits_4
```

After execute the above code, we end up with ten ```.txt``` files containing the predictions for each evaluated split. 
```bash
# ZSL
preds_SAE_ZSL_att_0.txt
preds_SAE_ZSL_att_1.txt
preds_SAE_ZSL_att_2.txt
preds_SAE_ZSL_att_3.txt
preds_SAE_ZSL_att_4.txt

# GZSL
preds_SAE_GZSL_att_0.txt
preds_SAE_GZSL_att_1.txt
preds_SAE_GZSL_att_2.txt
preds_SAE_GZSL_att_3.txt
preds_SAE_GZSL_att_4.txt
```

### 3.1 Evaluating LAD

After evaluating each of the five splits with the chosen ZSL algorithm, the final classification is performed by averaging the results of the five super-classes.

First, compute the Top-1 accuracy for LAD:

```python
# Evaluate LAD
compute_zsl_acc_lad(split=0, att_split="att_splits_0", preds_file="preds_SAE_att_0.txt")
compute_zsl_acc_lad(split=1, att_split="att_splits_1", preds_file="preds_SAE_att_1.txt")
compute_zsl_acc_lad(split=2, att_split="att_splits_2", preds_file="preds_SAE_att_2.txt")
compute_zsl_acc_lad(split=3, att_split="att_splits_3", preds_file="preds_SAE_att_3.txt")
compute_zsl_acc_lad(split=4, att_split="att_splits_4", preds_file="preds_SAE_att_4.txt")

# The above code returns a file named results_ZSL.txt, containing the accuracy for each of the 5 super-classes.

evaluate_LAD_ZSL("results_ZSL.txt")
```

And then, the Harmonic mean is calculated:  

```python
compute_harmonic_SAE_LAD(split=0, att_split="att_splits_0", preds="preds_SAE_GZSL_att_0.txt")
compute_harmonic_SAE_LAD(split=1, att_split="att_splits_1", preds="preds_SAE_GZSL_att_1.txt")
compute_harmonic_SAE_LAD(split=2, att_split="att_splits_2", preds="preds_SAE_GZSL_att_2.txt")
compute_harmonic_SAE_LAD(split=3, att_split="att_splits_3", preds="preds_SAE_GZSL_att_3.txt")
compute_harmonic_SAE_LAD(split=4, att_split="att_splits_4", preds="preds_SAE_GZSL_att_4.txt")

# The above code returns two files named results_seen.txt and results_unseen.txt, containing the accuracy for seen classes and the accuracy for unseen classes, respectively.

evaluate_LAD_GZSL(filename_seen="results_seen.txt", filename_unseen="results_unseen.txt")
```

However, for the remaining ZSL methods, the results on the generalized setting are obtained through the use of ```compute_harmonic_acc_lad(split, att_split, preds_seen_file, preds_unseen_file, seen)``` instead of ```compute_harmonic_SAE_LAD(split, att_split, preds=)```.

## 4. Extracting Custom Features

## 5. Optimizing TensorFlow Models with TensorRT

## 6. Evaluating the Computational Performance of ZSL methods