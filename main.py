from utils.engine import *
import time

"""
###########################################################################
|                       EVALUATION FRAMEWORK OPTIONS                      |
###########################################################################
| Available Datasets:                                                     |
+-------------------------------------------------------------------------+
|    * AWA1, AWA2, CUB, SUN, APY, LAD                                     |
+-------------------------------------------------------------------------+
| Available Features:                                                     |
+-------------------------------------------------------------------------+
|    * ResNet101                                                          |
|    * MobileNet                                                          |
|    * MobileNetV2                                                        |
|    * Xception                                                           | 
|    * EfficientNetB7                                                     |
+-------------------------------------------------------------------------+
| 1. Run ZSL Methods                                                      |
+-------------------------------------------------------------------------+
|    -> SAE(dataset="AWA2", filename="MobileNet")                         |
|    -> ESZSL(dataset="AWA2", filename="MobileNet")                       |
|    -> DEM(dataset="AWA2", filename="MobileNet")                         |
|    -> f_CLSWGAN(dataset="AWA2", filename="MobileNet")                   |
|    -> TF_VAEGAN(dataset="AWA2", filename="MobileNet")                   |
|    -> CE_GZSL(dataset="AWA2", filename="MobileNet")                     |
|                                                                         | 
|    In case of the LAD dataset, you must evaluate each of the five       |
|    splits in the desired ZSL method, by passing as argument the         |
|    correspondent split (0, 1, 2, 3, and 4).                             |
+-------------------------------------------------------------------------+
| 2. Evaluate LAD                                                         |
+-------------------------------------------------------------------------+
|  After evaluating each of the five att_splits in the chosen ZSL         |
|  algorithm, call the compute_zsl_acc_lad(split, att_split, preds_file)  |
|  for each of the generated preds_file corresponding to each of          |
|  the splits.                                                            |
|  Following the same idea, run the                                       |
|  compute_harmonic_acc_lad(split, att_split, preds_seen_file,            |
|  preds_unseen_file, seen) or the compute_harmonic_SAE_LAD(split,        |
|  att_split, preds), accordingly.                                        |      
|  Finally, run evaluate_LAD_ZSL(split, att_split, filename) and          |
|  evaluate_LAD_GZSL(filename_seen, filename_unseen) to get the results.  |
|#########################################################################|
"""

if __name__ == '__main__':
    # Run SAE algorithm for LAD dataset
    SAE(dataset="LAD", filename="MobileNetV2", att_split="_0")
    SAE(dataset="LAD", filename="MobileNetV2", att_split="_1")
    SAE(dataset="LAD", filename="MobileNetV2", att_split="_2")
    SAE(dataset="LAD", filename="MobileNetV2", att_split="_3")
    SAE(dataset="LAD", filename="MobileNetV2", att_split="_4")

    time.sleep(5)

    # Evaluate LAD
    compute_zsl_acc_lad(split=0, att_split="att_splits_0", preds_file="preds_SAE_att_0.txt")
    compute_zsl_acc_lad(split=1, att_split="att_splits_1", preds_file="preds_SAE_att_1.txt")
    compute_zsl_acc_lad(split=2, att_split="att_splits_2", preds_file="preds_SAE_att_2.txt")
    compute_zsl_acc_lad(split=3, att_split="att_splits_3", preds_file="preds_SAE_att_3.txt")
    compute_zsl_acc_lad(split=4, att_split="att_splits_4", preds_file="preds_SAE_att_4.txt")

    compute_harmonic_SAE_LAD(split=0, att_split="att_splits_0", preds="preds_SAE_GZSL_att_0.txt")
    compute_harmonic_SAE_LAD(split=1, att_split="att_splits_1", preds="preds_SAE_GZSL_att_1.txt")
    compute_harmonic_SAE_LAD(split=2, att_split="att_splits_2", preds="preds_SAE_GZSL_att_2.txt")
    compute_harmonic_SAE_LAD(split=3, att_split="att_splits_3", preds="preds_SAE_GZSL_att_3.txt")
    compute_harmonic_SAE_LAD(split=4, att_split="att_splits_4", preds="preds_SAE_GZSL_att_4.txt")

    time.sleep(5)

    evaluate_LAD_ZSL("results_ZSL.txt")
    evaluate_LAD_GZSL(filename_seen="results_seen.txt", filename_unseen="results_unseen.txt")

