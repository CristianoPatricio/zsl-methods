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
| AWA1 | Alpha=3, Gamma=0 | 56.2 | s=91.1   u=16.5  H=**27.9** |
| AWA2 | Alpha=3, Gamma=0 | 54.5 | s=92.8   u=6.5   H=**12.2** |
| CUB | Alpha=2, Gamma=0 | 51.3 | s=63.0    u=11.8  H=**20.0** |
| SUN | Alpha=2, Gamma=2 | 52.3 | s=27.3    u=13.1  H=**17.7** |
| aPY | Alpha=3, Gamma=-1 | 38.5 | s=79.7   u=2.3   H=**4.6** |

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