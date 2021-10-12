import gc

import numpy as np

from src.transforms import *
from src.utils import interval_from_line

gc.enable()

import random

import numpy as np

random.seed(666)

from selene_sdk.utils import load_path
from selene_sdk.utils.config_utils import get_full_dataset

if __name__ == "__main__":

    path = "model_configs/biox_dnase_multi_ct_masked_train.yaml"
    configs = load_path(path, instantiate=False)

    full_dataset = get_full_dataset(configs)

    n_folds = configs["dataset"]["dataset_args"]["n_folds"]

    dataset_info = configs["dataset"]
    # all intervals
    genome_intervals = []
    with open(dataset_info["sampling_intervals_path"]) as f:
        for line in f:
            chrom, start, end = interval_from_line(line)
            if chrom not in dataset_info["test_holdout"]:
                genome_intervals.append((chrom, start, end))

    print("genome_intervals:", len(genome_intervals))

    with open(dataset_info["distinct_features_path"]) as f:
        distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

    with open(dataset_info["target_features_path"]) as f:
        target_features = list(map(lambda x: x.rstrip(), f.readlines()))

    # split the intervals
    genome_intervals_arr = np.asarray(genome_intervals, dtype="U10,i8,i8")
    random.seed(666)
    random.shuffle(genome_intervals_arr)
    seq_splits = np.array_split(genome_intervals_arr, n_folds)

    splitted_intervals = []
    for train_intervals in seq_splits:
        # mark 20% of intervals as val intervals
        val_size = int(len(train_intervals) * 0.2)
        random.seed(666)
        val_intervals = random.sample(train_intervals.tolist(), val_size)
        splitted_intervals.append((train_intervals, val_intervals))

    print(len(splitted_intervals))
    print("train_intervals counts:", [len(k[0]) for k in splitted_intervals])
    print("val_intervals counts:", [len(k[1]) for k in splitted_intervals])

    np.save(
        "/home/thurs/DeepCT/results/splitted_intervals_hold.npy", splitted_intervals
    )
