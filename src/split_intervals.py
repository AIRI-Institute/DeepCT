import gc
import os
import tempfile

import numpy as np

from src.transforms import *

gc.enable()

import random

import numpy as np

random.seed(666)

from selene_sdk.utils import load_path
from selene_sdk.utils.config_utils import get_full_dataset, interval_from_line

CHROM_SIZES = {
    "chr1": 249250621,
    "chr2": 243199373,
    "chr3": 198022430,
    "chr4": 191154276,
    "chr5": 180915260,
    "chr6": 171115067,
    "chr7": 159138663,
    "chr8": 146364022,
    "chr9": 141213431,
    "chr10": 135534747,
    "chr11": 135006516,
    "chr12": 133851895,
    "chr13": 115169878,
    "chr14": 107349540,
    "chr15": 102531392,
    "chr16": 90354753,
    "chr17": 81195210,
    "chr18": 78077248,
    "chr19": 59128983,
    "chr20": 63025520,
    "chr21": 48129895,
    "chr22": 51304566,
    "chrX": 155270560,
    "chrY": 59373566,
    #'chrM': 16571
}

if __name__ == "__main__":

    path = "/home/msindeeva/DeepCT/model_configs/boix_train_masked_ct.yml"
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

    # split chromosomes into several non-overlapping folds
    # and create respective sequence splits
    target_bin_size = dataset_info["dataset_args"]["center_bin_to_predict"]
    with tempfile.NamedTemporaryFile("w") as f1, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as f2, tempfile.NamedTemporaryFile("w") as f3:
        for chrom, chrom_size in CHROM_SIZES.items():
            fold_size = chrom_size // n_folds
            for i in range(fold_size, chrom_size, fold_size):
                fold_spacing_start = i - target_bin_size // 2
                fold_spacing_end = i + target_bin_size // 2
                chrom_fold_borders = (
                    f"{chrom}\t{fold_spacing_start}\t{fold_spacing_end}\n"
                )
                f1.write(chrom_fold_borders)
        for chrom, start, end in genome_intervals:
            f3.write(f"{chrom}\t{start}\t{end}\n")
        os.system(f"bedtools subtract -A -a {f3.name} -b {f1.name} > {f2.name}")

    seq_splits = [[] for i in range(n_folds)]
    with open(f2.name) as f4:
        for line in f4:
            try:
                chrom, start, end = interval_from_line(line)
                fold_idx = start // (CHROM_SIZES[chrom] // n_folds)
                seq_splits[fold_idx].append((chrom, start, end))
            except:
                import pdb

                pdb.set_trace()
    os.unlink(f2.name)
    seq_splits = np.array(
        [np.asarray(seq_split, dtype="U10,i8,i8") for seq_split in seq_splits]
    )

    # split the intervals randomly
    """
    genome_intervals_arr = np.asarray(genome_intervals, dtype="U10,i8,i8")
    random.seed(666)
    random.shuffle(genome_intervals_arr)
    seq_splits = np.array_split(genome_intervals_arr, n_folds)
    """
    split_intervals = []
    for train_intervals in seq_splits:
        # mark 20% of intervals as val intervals
        val_size = int(len(train_intervals) * 0.2)
        random.seed(666)
        val_intervals = random.sample(train_intervals.tolist(), val_size)
        split_intervals.append((train_intervals, val_intervals))

    print(len(split_intervals))
    train_intervals_cnt = [len(k[0]) for k in split_intervals]
    val_intervals_cnt = [len(k[1]) for k in split_intervals]
    print(
        "train_intervals counts:",
        train_intervals_cnt,
        ", total:",
        sum(train_intervals_cnt),
    )
    print(
        "val_intervals counts:", val_intervals_cnt, ", total:", sum(val_intervals_cnt)
    )

    np.save(f"split_intervals_no_overlap_k{n_folds}.npy", split_intervals)
