import gc

import numpy as np

gc.enable()

import random

import numpy as np

random.seed(666)

from selene_sdk.utils import load_path

if __name__ == "__main__":

    path = "model_configs/biox_dnase_multi_ct_masked_train.yaml"
    configs = load_path(path, instantiate=False)
    k = configs["dataset"]["dataset_args"]["n_folds"]

    # create cell type masks
    ct_list = list(range(configs["model"]["class_args"]["n_cell_types"]))
    ct_masks = []
    for fold in range(k):
        random.seed(666)
        random.shuffle(ct_list)
        ct_masks.append(np.array_split(ct_list, k))

    print("CT train masks counts:", [c.shape[0] for c in ct_masks[0]])
    print("CT val masks counts:", [c.shape[0] for c in ct_masks[1]])

    np.save(f"results/cell_types_random_ids_k{k}.npy", ct_masks)
