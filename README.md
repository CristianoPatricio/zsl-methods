# zsl-methods

## 1. **ESZSL**

**Class**: `ESZSL(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, aPY}
<dataset_path> : {'./datasets/}
<filename> : {name.mat, name.pickle}
<alpha> : int value [-3,3] 
<gamma> : int value [-3,3]
```

**Functions**:

```
Name: train(alpha, gamma) 
:param alpha: int value
:param gamma: int value
:return: weights
```
```
Name: test(weights)
:param weights: array of shape (n_att, dim_feat)
:return: zsl_acc, gzsl_seen_acc, gzsl_unseen_acc, gzsl_harmonic_mean
```

| Dataset | Hyperparameter | Per-class Accuracy (%) | GZSL (s, u, H - %) |
| ----------- | ----------- | :------------: | :------------- |
| AWA1 | Alpha=3, Gamma=0 | 56.2 | s=86.8   u=5.3  H=**10.0** |
| AWA2 | Alpha=3, Gamma=0 | 54.5 | s=88.4   u=4.0   H=**7.8** |
| CUB | Alpha=2, Gamma=0 | 51.3 | s=61.7    u=13.5  H=**22.2** |
| SUN | Alpha=2, Gamma=2 | 52.3 | s=29.0    u=12.6  H=**17.5** |
| aPY | Alpha=3, Gamma=-1 | 38.5 | s=80.2   u=2.4   H=**4.7** |

## 2. **SAE**

**Class**: `SAE(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, aPY}
<dataset_path> : {'./datasets/}
<filename> : {name.mat, name.pickle}
<lamb_ZSL> : float value, default=2
<lamb_GZSL> : float value, default=2
<setting> : Type of evaluation {V2S, S2V}
```

**Functions**:

```
Name: train_zsl(lamb_zsl) 
:param lamb_zsl: float value
:return: weights_zsl
```
```
Name: train_gzsl(lamb_gzsl) 
:param lamb_gzsl: float value
:return: weights_gzsl
```
```
Name: test(weights_zsl, weights_gzsl, setting)
:param weights_zsl: array of shape (n_att, dim_feat)
:param weights_gzsl: array of shape (n_att, dim_feat)
:param setting: {V2S, S2V}
:return: zsl_acc, gzsl_seen_acc, gzsl_unseen_acc, gzsl_harmonic_mean
```

| Dataset | Setting | Lambda (ZSL) | Lambda (GZSL) | Per-class Accuracy (%) | GZSL (s, u, H - %) |
| ------- | :------:| :----------: | :-----------: | :--------------------: | :---------------- |
| AWA1    |   V2S   |  3.0   |  3.2  | 51.5   | s=85.0  u=5.9   H=**11.0**
| AWA2    |   V2S   |  0.6   |  0.8  | 51.8 | s=86.0    u=3.7   H=**7.2**
| CUB     |   V2S   |  100   |  80   | 39.1 | s=49.4 u=14.0    H= **21.9**
| SUN     |   V2S   |  0.32  |  0.32 | 52.4 | s=24.7    u=16.8  H=**20.0**
| aPY     |   V2S   |  2.0   |  0.16 | 15.9 | s=18.1    u=0.7  H=**1.4**

| Dataset | Setting | Lambda (ZSL) | Lambda (GZSL) | Per-class Accuracy (%) | GZSL (s, u, H - %) |
| ------- | :------:| :----------: | :-----------: | :--------------------: | :---------------- |
| AWA1    |   S2V   |  0.8   |  0.8  |  60.3  | s=87.0  u=24.4   H=**38.09**
| AWA2    |   S2V   |  0.2   |  0.2  | 60.2 | s=90.1    u=12.3   H=**21.7**
| CUB     |   S2V   |  0.2   |  0.2   | 45.6 | s=57.0 u=16.9    H= **26.1**
| SUN     |   S2V   |  0.16  |  0.08 | 60.2 | s=31.5    u=18.8  H=**23.6**
| aPY     |   S2V   |  4.0   |  2.56 | 16.3 | s=70.1    u=0.7  H=**1.5**

## 3. **DEM**

**Class**: `DEM(args)`

**Arguments**:
```
<dataset> : {AWA1, AWA2, CUB, SUN, aPY}
<dataset_path> : {'./datasets/}
<filename> : {name.mat, name.pickle}
<lamb> : float value, default=1e-3
<lr> : float value, default=1e-4
<batch_size> : batch size, default=64
<hidden_dim> : Dimension of the hidden layer, default=1600
```

**Functions**:

```
Name: train(batch_size) 
:param batch_size: int value
:return: -
```


| Dataset | Hidden Dim | Lambda | Learning Rate | Per-class Accuracy (%) | GZSL (s, u, H - %) |
| ------- | :------:| :----------: | :-----------: | :--------------------: | :---------------- |
| AWA1    |   1600   |  1e-3   |  1e-4 |  | s=  u=   H=****
| AWA2    |   1600   |  1e-3   |  1e-4 |  | s=    u=   H=****
| CUB     |   1600   |  1e-2   |  1e-4 |  | s= u=    H= ****
| SUN     |   1600   |        |   |  | s=    u=  H=****
| aPY     |   1600   |    |   |  | s=    u=  H=****

## 4. f-CLSWGAN

:warning: _Temporary version_

**Run instructions**:
```
python3 original_f-CLSWGAN.py --download_mode
python3 original_f-CLSWGAN.py --train_classifier
python3 original_f-CLSWGAN.py --train_WGAN
```