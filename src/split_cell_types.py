import numpy as np
import gc
gc.enable()

import numpy as np
import random
random.seed(666)

from selene_sdk.utils import load_path


if __name__=='__main__':

    path = 'model_configs/biox_dnase_multi_ct_masked_train.yaml'
    configs = load_path(path, instantiate=False)
    n_folds = configs['dataset']['dataset_args']['n_folds']


    # create cell type masks
    ct_list = list(range(configs['model']['class_args']['n_cell_types'])) 
    ct_masks = []
    for fold in range(n_folds):
        random.seed(666)
        random.shuffle(ct_list)
        ct_masks.append(np.array_split(ct_list, n_folds))


    print('CT train masks counts:', [c.shape[0] for c in ct_masks[0]])
    print('CT val masks counts:', [c.shape[0] for c in ct_masks[1]])

    np.save(f'results/cell_types_random_ids_k{n_folds}.npy', ct_masks)
