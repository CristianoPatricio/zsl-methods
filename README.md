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
| ----------- | ----------- | :------------: | :-------------: |
| AWA1 | Alpha=3, Gamma=0 | 56.2 | 91.1 / 16.5 / **27.9** |
| AWA2 | Alpha=3, Gamma=0 | 54.5 | 92.8 / 6.5 / **12.2** |
| CUB | Alpha=2, Gamma=0 | 51.3 | 63.0 / 11.8 / **20.0** |
| SUN | Alpha=2, Gamma=2 | 52.3 | 27.3 / 13.1 / **17.7** |
| aPY | Alpha=3, Gamma=-1 | 38.5 | 79.7 / 2.3 / **4.6** |



