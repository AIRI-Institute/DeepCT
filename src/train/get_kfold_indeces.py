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

    n_folds = 10
    k_fold = KFold(n_folds, shuffle=True, random_state=666)

    dataset_info = configs["dataset"]

    # all intervals
    genome_intervals = []
    with open(dataset_info["sampling_intervals_path"])  as f:
        for line in f:
            chrom, start, end = interval_from_line(line)
            if chrom not in dataset_info["test_holdout"]:
                genome_intervals.append((chrom, start, end))

    print(len(genome_intervals))

    with open(dataset_info["distinct_features_path"]) as f:
        distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

    with open(dataset_info["target_features_path"]) as f:
        target_features = list(map(lambda x: x.rstrip(), f.readlines()))


    splits = []
    for train_idx, test_idx in k_fold.split(genome_intervals):
        splits.append((train_idx, test_idx))

    # np.save(f'/home/thurs/DeepCT/results/kfold_splits_hold.npy', splits)
    # splits = np.load(f'/home/thurs/DeepCT/results/kfold_splits_hold.npy', allow_pickle=True)

    print(len(splits))
    print(len(splits[0][0]), len(splits[0][1]))



