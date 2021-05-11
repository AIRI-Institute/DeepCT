"""
Based on ExpResNet architecture (see https://www.biorxiv.org/content/10.1101/2020.06.21.163956v1.full.pdf)

A residual architecture with a skip connection for cell type embeddings.
The forward method accepts the cell type embeddings to use during training, 
as well as cell targets which select the cell types (and hence embeddings) 
to train on for each sequence sample (used for balancing the dataset). 
"""
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def bn_relu_conv(in_channels, out_channels, k, d):
    stack = nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=k,
            dilation=d,
            padding=get_padding(k, d),
        ),
    )
    return stack


def get_padding(k, d):
    # Returns padding necessary to retain the same tensor dimensions
    # after 1d convolution given kernel size (k) and dilation size (d)

    return int(((k - 1) * (d - 1) + k - 1) / 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k, d):
        super().__init__()

        self.first_conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=1)

        self.blocks = nn.Sequential(
            bn_relu_conv(out_channels, out_channels // 2, k=1, d=1),
            bn_relu_conv(out_channels // 2, out_channels // 2, k=3, d=d),
            bn_relu_conv(out_channels // 2, out_channels, k=1, d=1),
        )

    def forward(self, x):
        residual = self.first_conv(x)
        x = self.blocks(residual)
        x += residual
        return x


class ExpResNet_Manvel(nn.Module):
    def __init__(
        self,
        sequence_length,
        n_cell_types,
        sequence_embedding_length,
        cell_type_embedding_length,
        final_embedding_length,
        n_genomic_features,
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Length of input sequence.
        n_cell_types : int
            Number of cell types.
        sequence_embedding_length : int
        cell_type_embedding_length : int
        final_embedding_length : int
        n_genomic_features : int
            Number of target features.
        """
        super().__init__()

        self._n_cell_types = n_cell_types

        k_l = [5, 3, 3, 3, 3]  # kernel sizes
        d_l = [1, 2, 2, 4, 2]  # dilation sizes

        p_l = [4, 4, 4, 4, 4]  # pooling sizes

        self.conv_net = nn.Sequential(
            ResidualBlock(4, 160, k=k_l[0], d=d_l[0]),
            nn.AvgPool1d(kernel_size=p_l[0], stride=p_l[0]),
            ResidualBlock(160, 320, k=k_l[1], d=d_l[1]),
            nn.AvgPool1d(kernel_size=p_l[1], stride=p_l[1]),
            ResidualBlock(320, 480, k=k_l[2], d=d_l[2]),
            nn.AvgPool1d(kernel_size=p_l[2], stride=p_l[2]),
            ResidualBlock(480, 640, k=k_l[3], d=d_l[3]),
            nn.AvgPool1d(kernel_size=p_l[3], stride=p_l[3]),
            ResidualBlock(640, 960, k=k_l[4], d=d_l[4]),
            nn.AvgPool1d(kernel_size=p_l[4], stride=p_l[4]),
        )

        length = sequence_length
        for i in range(len(k_l)):
            length = self.calc_reduction(length, k_l[i], d_l[i], p_l[i])

        self._n_channels = int(length)

        self.merge_net = nn.Sequential(
            nn.Linear(cell_type_embedding_length, 960),
        )

        self.sequence_net = nn.Sequential(
            nn.Linear(960 * self._n_channels, sequence_embedding_length),
            nn.BatchNorm1d(sequence_embedding_length),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                sequence_embedding_length + cell_type_embedding_length,
                final_embedding_length,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, n_genomic_features),
        )

    def calc_reduction(self, length, k, d, pool_kernel_size):
        # Calculate length of tensor following convolution and pooling
        # with kernel_size=k, dilation_size=d
        return np.floor((length - (k - 1)) / pool_kernel_size)

    def pick_embeddings(self, embeddings, cell_targets):
        # Returns tensor of embeddings used for the mini-batch

        tiled_embeddings = embeddings.repeat(
            cell_targets.size()[0], 1
        )  # repeat all available embeddings batch_size times
        flat_targets = cell_targets.flatten()
        out_embeddings = tiled_embeddings[
            flat_targets == 1, :
        ]  # select embeddings to use for this mini-batch

        return out_embeddings

    def log_cell_type_embeddings_to_tensorboard(self, cell_type_labels, output_dir):
        writer = SummaryWriter(output_dir)
        writer.add_embedding(
            self.cell_type_net[0].weight.transpose(0, 1), cell_type_labels
        )
        writer.flush()
        writer.close()

    def forward(self, x, cell_targets, embeddings):
        """Forward propagation of a batch.

        Parameters:
        -----------
        x : torch.Tensor
            A batch of encoded sequences.
        cell_targets : torch.Tensor
            A batch of vectors denoting the cell type embeddings to use
            for each encoded sequence.
        embeddings : torch.Tensor
            n_cell_types cell type embeddings.
        """

        batch_size = x.size(0)

        cell_type_embeddings = self.pick_embeddings(embeddings, cell_targets)

        sequence_out = self.conv_net(x)
        sequence_out = sequence_out.repeat_interleave(
            repeats=torch.sum(cell_targets, dim=1), dim=0
        )

        merge_embeddings = self.merge_net(embeddings)
        merge_embeddings = self.pick_embeddings(merge_embeddings, cell_targets)
        merge_embeddings = merge_embeddings.unsqueeze(2)
        emb_size = merge_embeddings.size()
        merge_embeddings = merge_embeddings.expand(
            emb_size[0], emb_size[1], self._n_channels
        )

        seq_type_combined = torch.add(sequence_out, merge_embeddings)

        reshaped_sequence_out = seq_type_combined.view(
            seq_type_combined.size(0), 960 * self._n_channels
        )

        sequence_embedding = self.sequence_net(reshaped_sequence_out)

        sequence_and_cell_type_embeddings = torch.cat(
            (sequence_embedding, cell_type_embeddings), 1
        )

        predict = self.classifier(sequence_and_cell_type_embeddings)
        return predict


def criterion():
    """
    The criterion the model aims to minimize.
    """
    # TODO: Update the criterion to evaluate only features available for the provided
    # cell type.
    return nn.MSELoss()


def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    # return (torch.optim.SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
    return (torch.optim.Adam, {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-08})
