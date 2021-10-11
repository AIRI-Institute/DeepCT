import yaml
import os
import sys
# sys.path.append('../../')
import numpy as np
from collections import Counter
from omegaconf import OmegaConf
# import torchmetrics
# import torch
# from torch import nn
from selene_sdk.utils import load_path, parse_configs_and_run
from selene_sdk.utils.config_utils import module_from_dir, module_from_file, get_full_dataset
from selene_sdk.utils.config import instantiate
from src.dataset import EncodeDataset, LargeRandomSampler, encode_worker_init_fn
from src.transforms import *
from src.utils import interval_from_line
# from torchvision import transforms
# from torchmetrics import BinnedAveragePrecision, AveragePrecision, Accuracy
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import copy
from src.utils import expand_dims
import gc
gc.enable()

from src.metrics import jaccard_score, threshold_wrapper
from sklearn.metrics import average_precision_score
from selene_sdk.utils.performance_metrics import compute_score



def get_full_dl(configs):
    """
    """
    if "dataset" in configs:
        dataset_info = configs["dataset"]

        # all intervals
        # genome_intervals = []
        # with open(dataset_info["sampling_intervals_path"])  as f:
        #     for line in f:
        #         chrom, start, end = interval_from_line(line)

        #         if chrom not in dataset_info["test_holdout"]:
        #             genome_intervals.append((chrom, start, end))

        # bedug
        # genome_intervals = random.sample(genome_intervals, k=20)
        # print("DEBUG MODE ON:", len(genome_intervals))

        # with open(dataset_info["distinct_features_path"]) as f:
        #     distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        # with open(dataset_info["target_features_path"]) as f:
        #     target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        module = None
        if os.path.isdir(dataset_info["path"]):
            module = module_from_dir(dataset_info["path"])
        else:
            module = module_from_file(dataset_info["path"])

        # dataset_class = getattr(module, dataset_info["class"])
        # dataset_info["dataset_args"]["target_features"] = target_features
        # dataset_info["dataset_args"]["distinct_features"] = distinct_features

        # # load train dataset and loader
        # data_config = dataset_info["dataset_args"].copy()
        # data_config["intervals"] = genome_intervals

        # del data_config['n_folds']
        # if "train_transform" in dataset_info:
        #     # load transforms
        #     train_transform = instantiate(dataset_info["train_transform"])
        #     data_config["transform"] = train_transform
        # full_dataset = dataset_class(**data_config)

        full_dataset = get_full_dataset(configs)
        print('full_dataset without hold-out:', len(full_dataset))

        sampler_class = getattr(module, dataset_info["sampler_class"])
        gen = torch.Generator()
        gen.manual_seed(configs["random_seed"])
        train_sampler = sampler_class(
            full_dataset, replacement=False, generator=gen
        )

        full_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=dataset_info["loader_args"]["batch_size"],
            num_workers=dataset_info["loader_args"]["num_workers"],
            worker_init_fn=module.encode_worker_init_fn,
            sampler=train_sampler,
        )
        print('full_datloader without hold-out:', len(full_loader))

        return full_loader


if __name__ == "__main__":

    path = 'model_configs/biox_dnase_multi_ct_crossval.yaml'
    configs = load_path(path, instantiate=False)
    full_dataloader = get_full_dl(configs)

    # mean over seq
    # all_idx = []
    # all_mean_targets = []
    # for batch in tqdm(full_dataloader):
    #     # _, _, targets, _ = batch 
    #     idx, retrieved_sample = batch # __getitem__ should returns idx, retrieved_sample
    #     _, _, targets, _ = retrieved_sample
    #     targets = targets.to('cuda:3')

    #     # mean target over CT dim
    #     mean_target = torch.mean(targets, dim=1).flatten() 
    #     all_mean_targets.append(mean_target)
    #     all_idx.append(idx)

    #  idx_np = torch.cat(all_idx, dim=0).cpu().numpy()

    # mean_target_np = full_mean_targets.cpu().numpy()
    # np.save(f'/home/thurs/DeepCT/results/all_targets_mean.npy', mean_target_np)
    # np.save(f'/home/thurs/DeepCT/results/all_idx.npy', idx_np)

    # y_cat = pd.cut(mean_target_np, 10, labels=range(10))
    # y_cat = np.array(y_cat)

    
    # train_idx = []
    # val_idx = []
    # train_y = []
    # val_y = []

    # skf = StratifiedKFold(n_splits=5, shuffle=False)

    # for train_index, val_index in skf.split(idx_np, y_cat):
    #     train_idx.append(train_index)
    #     val_idx.append(val_index)
    #     train_y.append(y_cat[train_index])
    #     val_y.append(y_cat[val_index])

    # d = dict()
    # for fold in range(len(train_idx)):
    #     d[fold] = (train_idx[fold], train_y[fold], val_idx[fold], val_y[fold])

    # res_df = pd.DataFrame.from_dict(d, orient='index', columns=['train_idx', 'train_y', 'val_idx', 'val_y']) 
    # res_df.to_csv(f'/home/thurs/DeepCT/results/skf_idx.csv')


    # mean over CT
    # all_targets = []
    # for batch in tqdm(full_dataloader):
    #     _, _, targets, _ = batch 
    #     targets = targets.to('cuda:1')
    #     avg_targets = torch.mean(targets, dim=0)
    #     all_targets.append(avg_targets)

    # full_targets = torch.cat(all_targets, dim=1)
    # print(full_targets.shape)

    # ct_mean_targets = torch.mean(full_targets, dim=1).flatten()
    # print(ct_mean_targets.shape)

    # np.save(f'/home/thurs/DeepCT/results/ct_mean_targets_02.npy', ct_mean_targets.cpu().numpy())

    # y_cat = pd.cut(ct_mean_targets.cpu().numpy(), 10, labels=range(10))
    # y_cat = np.array(y_cat)

    # skf = StratifiedKFold(n_splits=5, shuffle=False)

    # train_y = []
    # val_y = []
    # ct_range = np.array(range(631))
    # for train_index, val_index in skf.split(ct_range, y_cat):
    #     train_y.append(y_cat[train_index])
    #     val_y.append(y_cat[val_index])

    # train_y = np.array(train_y)
    # val_y = np.array(val_y)
    # np.save(f'/home/thurs/DeepCT/results/ct_means_train.npy', train_y)
    # np.save(f'/home/thurs/DeepCT/results/ct_means_val.npy', val_y)

