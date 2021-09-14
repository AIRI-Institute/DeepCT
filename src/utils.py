import copy
import json
import numpy as np
import torch
from tqdm import tqdm
import os
from selene_sdk.utils.config_utils import module_from_dir, module_from_file
from selene_sdk.utils.config import instantiate
# from src.dataset import CustomSubset

# maximum size of targets stored in-memory for validation metrics computation,
# value derived experimentally using data loader of size 2000, batch size 64,
# 194 cell types, and 201 target features
MAX_TOTAL_VAL_TARGET_SIZE = 2000 * 64 * 194 * 201


def interval_from_line(bed_line, pad_left=0, pad_right=0, chrom_counts=None):
    chrom, start, end = bed_line.rstrip().split('\t')[:3]
    start = max(0, int(start) - pad_left)
    if pad_right:
        end = min(int(end) + pad_right, chrom_counts[chrom])
    else:
        end = int(end)
    return chrom, start, end
    

def expand_dims(x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return x


def batchwise_mean_jaccard(loader, thresholds=[0.3]):
    n_cell_types = loader.dataset.n_cell_types
    n_features = loader.dataset.n_target_features

    cum_intersection = [0 for th in thresholds]
    cum_union = [0 for th in thresholds]
    for sample in tqdm(loader):
        batch = copy.deepcopy(sample)
        del sample

        gt = batch[2]
        mask = batch[3]

        mean_seq_val = (gt * mask).sum(axis=1) / mask.sum(axis=1)
        mean_batch_pred = torch.repeat_interleave(mean_seq_val, n_cell_types, dim=0)
        mean_batch_pred = mean_batch_pred.view(gt.shape[0], -1, n_features)

        for i, threshold in enumerate(thresholds):
            pred = mean_batch_pred > threshold

            intersection = (pred * gt * mask).sum(axis=[0, 1])
            union = (gt * mask).sum(axis=[0, 1]) + (pred * mask).sum(axis=[0, 1])

            cum_intersection[i] += intersection
            cum_union[i] += union

    cum_intersection = torch.vstack(cum_intersection)
    cum_union = torch.vstack(cum_union)
    return cum_intersection / cum_union


def get_skf_datasets(configs):
    """
    """

    if "dataset" in configs:
        dataset_info = configs["dataset"]
        fold_ids_path = configs["dataset"]['fold_ids']
        current_fold = configs["dataset"]['dataset_args']['fold']

        with open(fold_ids_path, 'r') as f:
            skf_idx_dict = json.load(f)


        # all intervals
        genome_intervals = []
        with open(dataset_info["sampling_intervals_path"])  as f:
            for line in f:
                chrom, start, end = interval_from_line(line)
                genome_intervals.append((chrom, start, end))

        print(len(genome_intervals))
        # bedug
        # genome_intervals = random.sample(genome_intervals, k=20)
        # print("DEBUG MODE ON:", len(genome_intervals))

        with open(dataset_info["distinct_features_path"]) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        print(len(distinct_features))

        with open(dataset_info["target_features_path"]) as f:
            target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        module = None
        if os.path.isdir(dataset_info["path"]):
            module = module_from_dir(dataset_info["path"])
        else:
            module = module_from_file(dataset_info["path"])

        dataset_class = getattr(module, dataset_info["class"])
        dataset_info["dataset_args"]["target_features"] = target_features
        dataset_info["dataset_args"]["distinct_features"] = distinct_features

        # load train dataset and loader
        data_config = dataset_info["dataset_args"].copy()
        data_config["intervals"] = genome_intervals

        # train_config = dataset_info["dataset_args"].copy()
        del data_config['fold']
        del data_config['n_folds']
        # train_config["intervals"] = genome_intervals
        if "train_transform" in dataset_info:
            # load transforms
            train_transform = instantiate(dataset_info["train_transform"])
            data_config["transform"] = train_transform
        full_dataset = dataset_class(**data_config)
        print(len(full_dataset))

        skf_tr_subset = torch.utils.data.Subset(
            full_dataset, 
            skf_idx_dict[str(current_fold)]['train_idx']
            )
        skf_val_subset = torch.utils.data.Subset(
            full_dataset, 
            skf_idx_dict[str(current_fold)]['val_idx']
            )

        # skf_tr_subset = CustomSubset(
        #     dataset = full_dataset, 
        #     indices = skf_idx_dict[str(current_fold)]['train_idx'],
        #     # **data_config
        #     )
        # skf_val_subset = CustomSubset(
        #     dataset = full_dataset, 
        #     indices = skf_idx_dict[str(current_fold)]['val_idx'],
        #     # **data_config
        #     )

        return skf_tr_subset, skf_val_subset


def get_full_dataset(configs):
    """
    """
    if "dataset" in configs:
        dataset_info = configs["dataset"]

        # all intervals
        genome_intervals = []
        with open(dataset_info["sampling_intervals_path"])  as f:
            for line in f:
                chrom, start, end = interval_from_line(line)
                genome_intervals.append((chrom, start, end))

        print(len(genome_intervals))
        # bedug
        # genome_intervals = random.sample(genome_intervals, k=20)
        # print("DEBUG MODE ON:", len(genome_intervals))

        with open(dataset_info["distinct_features_path"]) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        print(len(distinct_features))

        with open(dataset_info["target_features_path"]) as f:
            target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        module = None
        if os.path.isdir(dataset_info["path"]):
            module = module_from_dir(dataset_info["path"])
        else:
            module = module_from_file(dataset_info["path"])

        dataset_class = getattr(module, dataset_info["class"])
        dataset_info["dataset_args"]["target_features"] = target_features
        dataset_info["dataset_args"]["distinct_features"] = distinct_features

        # load train dataset and loader
        data_config = dataset_info["dataset_args"].copy()
        data_config["intervals"] = genome_intervals

        # train_config = dataset_info["dataset_args"].copy()
        del data_config['fold']
        del data_config['n_folds']
        # train_config["intervals"] = genome_intervals
        if "train_transform" in dataset_info:
            # load transforms
            train_transform = instantiate(dataset_info["train_transform"])
            data_config["transform"] = train_transform
        full_dataset = dataset_class(**data_config)
        print(len(full_dataset))

        return full_dataset


def get_full_dl(configs):
    """
    """
    if "dataset" in configs:
        dataset_info = configs["dataset"]

        # all intervals
        genome_intervals = []
        with open(dataset_info["sampling_intervals_path"])  as f:
            for line in f:
                chrom, start, end = interval_from_line(line)
                genome_intervals.append((chrom, start, end))

        print(len(genome_intervals))
        # bedug
        # genome_intervals = random.sample(genome_intervals, k=20)
        # print("DEBUG MODE ON:", len(genome_intervals))

        with open(dataset_info["distinct_features_path"]) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        print(len(distinct_features))

        with open(dataset_info["target_features_path"]) as f:
            target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        module = None
        if os.path.isdir(dataset_info["path"]):
            module = module_from_dir(dataset_info["path"])
        else:
            module = module_from_file(dataset_info["path"])

        dataset_class = getattr(module, dataset_info["class"])
        dataset_info["dataset_args"]["target_features"] = target_features
        dataset_info["dataset_args"]["distinct_features"] = distinct_features

        # load train dataset and loader
        data_config = dataset_info["dataset_args"].copy()
        data_config["intervals"] = genome_intervals

        # train_config = dataset_info["dataset_args"].copy()
        del data_config['fold']
        del data_config['n_folds']
        # train_config["intervals"] = genome_intervals
        if "train_transform" in dataset_info:
            # load transforms
            train_transform = instantiate(dataset_info["train_transform"])
            data_config["transform"] = train_transform
        full_dataset = dataset_class(**data_config)
        print(len(full_dataset))

        sampler_class = getattr(module, dataset_info["sampler_class"])
        gen = torch.Generator()
        gen.manual_seed(configs["random_seed"])
        train_sampler = sampler_class(
            full_dataset, replacement=False, generator=gen
        )

        full_dataloader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=dataset_info["loader_args"]["batch_size"],
            num_workers=dataset_info["loader_args"]["num_workers"],
            worker_init_fn=module.encode_worker_init_fn,
            sampler=train_sampler,
        )

        return full_dataloader
