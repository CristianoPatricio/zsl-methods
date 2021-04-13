# zsl-methods

1. **ESZSL**

Class: `ESZSL(args)`

Functions:

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
Arguments:

<dataset> : {AWA1, AWA2, CUB, SUN, aPY}
<dataset_path> : {'./datasets/}
<filename> : {name.mat, name.pickle}
<alpha> : int value [-3,3] 
<gamma> : int value [-3,3]
```

| Dataset | Hyperparameter |
| ----------- | ----------- |
| AWA1 | Alpha=3, Gamma=0 |
| AWA2 | Alpha=3, Gamma=0 |
| CUB | Alpha=2, Gamma=0 |
| SUN | Alpha=2, Gamma=2 |
| aPY | Alpha=3, Gamma=-1 |



