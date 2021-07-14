import torch
import torchvision

from dataset import EncodeDataset
from src.transforms import PermuteSequenceChannels, RandomReverseStrand
from data.utils import interval_from_line
from dataset import LargeRandomSampler, encode_worker_init_fn


def load_datasets(
    data_folder,
    train=True,
    val=False,
    test=False,
    train_transform=None,
    val_transform=None,
    test_transform=None,
):
    target_features_path = f"{data_folder}/target_features.txt"
    sampling_intervals_path = f"{data_folder}/target_intervals.bed"
    reference_sequence_path = "/mnt/datasets/DeepCT/male.hg19.fasta"
    target_path = f"{data_folder}/sorted_data.bed.gz"
    distinct_features_path = f"{data_folder}/distinct_features.txt"

    train_intervals = []
    val_intervals = []
    test_intervals = []
    with open(sampling_intervals_path) as f:
        for line in f:
            chrom, start, end = interval_from_line(line)
            if chrom in ["chr6", "chr7"]:
                val_intervals.append((chrom, start, end))
            elif chrom not in ["chr8", "chr9"]:
                train_intervals.append((chrom, start, end))
            else:
                test_intervals.append((chrom, start, end))

    with open(distinct_features_path) as f:
        distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

    with open(target_features_path) as f:
        target_features = list(map(lambda x: x.rstrip(), f.readlines()))

    datasets = []
    if train:
        if train_transform is None:
            train_transform = torchvision.transforms.Compose(
                [
                    PermuteSequenceChannels(),
                    RandomReverseStrand(p=0.5),
                ]
            )

        train_dataset = EncodeDataset(
            reference_sequence_path,
            target_path,
            distinct_features,
            target_features,
            train_intervals,
            cell_wise=True,
            sequence_length=1000,
            center_bin_to_predict=200,
            multi_ct_target=True,
            position_skip=120,
            transform=train_transform,
        )
        datasets.append(train_dataset)
    if val:
        if val_transform is None:
            val_transform = PermuteSequenceChannels()

        val_dataset = EncodeDataset(
            reference_sequence_path,
            target_path,
            distinct_features,
            target_features,
            val_intervals,
            cell_wise=True,
            sequence_length=1000,
            center_bin_to_predict=200,
            multi_ct_target=True,
            position_skip=120,
            transform=val_transform,
        )
        datasets.append(val_dataset)
    if test:
        if test_transform is None:
            test_transform = PermuteSequenceChannels()

        test_dataset = EncodeDataset(
            reference_sequence_path,
            target_path,
            distinct_features,
            target_features,
            test_intervals,
            cell_wise=True,
            sequence_length=1000,
            center_bin_to_predict=200,
            multi_ct_target=True,
            position_skip=120,
            transform=test_transform,
        )
        datasets.append(test_dataset)
    return datasets


def get_loader(dataset, batch_size=128, num_workers=16, shuffle=False):
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(shuffle)
        shuffle = LargeRandomSampler(dataset, replacement=False, generator=gen)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=encode_worker_init_fn,
        shuffle=shuffle,
    )
    return loader
