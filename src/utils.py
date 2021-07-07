import copy

import numpy as np
import torch
from tqdm import tqdm

# maximum size of targets stored in-memory for validation metrics computation,
# value derived experimentally using data loader of size 2000, batch size 64,
# 194 cell types, and 201 targets
# MAX_TOTAL_VAL_TARGET_SIZE = 2000 * 64 * 194 * 201
# updated with data loader of size 500, batch size 128, 858 cell types, and 40 targets
MAX_TOTAL_VAL_TARGET_SIZE = 500 * 128 * 858 * 40


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
