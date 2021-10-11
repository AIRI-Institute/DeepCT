import numpy as np
from src.transforms import *
from src.utils import interval_from_line
import gc
gc.enable()

import numpy as np
import random
random.seed(666)

from selene_sdk.utils.config_utils import get_full_dataset
from selene_sdk.utils import load_path
from sklearn.model_selection import KFold, StratifiedKFold


if __name__=='__main__':

    path = 'model_configs/biox_dnase_multi_ct_crossval.yaml'
    configs = load_path(path, instantiate=False)

    full_dataset = get_full_dataset(configs)

    n_folds = configs['dataset']['dataset_args']['n_folds']

    dataset_info = configs["dataset"]

    # all intervals
    genome_intervals = []
    with open(dataset_info["sampling_intervals_path"])  as f:
        for line in f:
            chrom, start, end = interval_from_line(line)
            if chrom not in dataset_info["test_holdout"]:
                genome_intervals.append((chrom, start, end))

    print('genome_intervals:', len(genome_intervals))

    with open(dataset_info["distinct_features_path"]) as f:
        distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

    with open(dataset_info["target_features_path"]) as f:
        target_features = list(map(lambda x: x.rstrip(), f.readlines()))

    # новые сплиты для seq
    genome_intervals_arr = np.asarray(genome_intervals, dtype='U10,i8,i8')
    random.seed(666)
    random.shuffle(genome_intervals_arr)
    seq_splits = np.array_split(genome_intervals_arr, n_folds)

    kfold_intervals = []
    for train_intervals in seq_splits:
        val_size = int(len(train_intervals)*0.2)
        random.seed(666)
        val_intervals = random.sample(train_intervals.tolist(), val_size)
        kfold_intervals.append((train_intervals, val_intervals))

    print(len(kfold_intervals))
    print([len(k[0]) for k in kfold_intervals])
    print([len(k[1]) for k in kfold_intervals])
    np.save(f'/home/thurs/DeepCT/results/kfold_intervals_hold.npy', kfold_intervals)


    # маски для seq
    # splits = []
    # for train_idx, test_idx in k_fold.split(genome_intervals):
    #     splits.append((train_idx, test_idx))

    # print('# seq folds:', len(splits))
    # print('seq train/val folds:', len(splits[0][0]), len(splits[0][1]))
    
    # np.save(f'/home/thurs/DeepCT/results/kfold_splits_hold.npy', splits)
    # splits = np.load(f'/home/thurs/DeepCT/results/kfold_splits_hold.npy', allow_pickle=True)


    # маски для cell type
    ct_list = list(range(configs['model']['class_args']['n_cell_types'])) 
    ct_masks = []
    for fold in range(n_folds):
        random.seed(666)
        random.shuffle(ct_list)
        ct_masks.append(np.array_split(ct_list, n_folds))

    print('# variants of val masks', [len(c) for c in ct_masks])
    print('CT train masks:', [c.shape[0] for c in ct_masks[0]])
    print('CT val masks:', [c.shape[0] for c in ct_masks[1]])

    np.save(f'results/ct_random_ids_k{n_folds}.npy', ct_masks)

