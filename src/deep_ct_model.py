"""
DeepCT architecture (TODO: Add our names).
"""
import numpy as np
import torch
import torch.nn as nn


class DeepCT(nn.Module):
    def __init__(
        self,
        sequence_length,
        n_cell_types,
        cell_type_embedding_length,
        n_genomic_features,
    ):
        """
        Based on a DeepSEA architecture (see https://github.com/FunctionLab/selene/blob/0.4.8/models/deepsea.py)

        Parameters
        ----------
        sequence_length : int
        n_cell_types : int
        cell_type_embedding_length : int
        n_genomic_features : int
        """
        super(DeepCT, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        sequence_embedding_length = 128
        final_embedding_length = 128

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        reduce_by = conv_kernel_size - 1
        pool_kernel_size = float(pool_kernel_size)
        self.n_channels = int(
            np.floor(
                (np.floor((sequence_length - reduce_by) / pool_kernel_size) - reduce_by)
                / pool_kernel_size
            )
            - reduce_by
        )

        self.sequence_net = nn.Sequential(
            nn.Linear(960 * self.n_channels, sequence_embedding_length),
            nn.ReLU(inplace=True),
        )

        self.cell_type_net = nn.Sequential(
            nn.Linear(n_cell_types, cell_type_embedding_length),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                sequence_embedding_length + cell_type_embedding_length,
                final_embedding_length,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward propagation of a batch.

        Parameters:
        -----------
        x : dict(str, torch.Tensor)
            x['sequence_batch'] corresponds to a batch of encoded sequences.
            x['cell_type_batch'] corresponds to a batch of one-hot cell type encodings.

        """
        sequence_out = self.conv_net(x["sequence_batch"])
        reshaped_sequence_out = sequence_out.view(
            sequence_out.size(0), 960 * self.n_channels
        )
        sequence_embedding = self.sequence_net(reshaped_sequence_out)

        cell_type_embedding = self.cell_type_net(x["cell_type_batch"])

        sequence_and_cell_type_embeddings = torch.cat(
            (sequence_embedding, cell_type_embedding), 1
        )

        predict = self.classifier(sequence_and_cell_type_embeddings)
        return predict


def criterion():
    """
    The criterion the model aims to minimize.
    """
    # TODO: Update the criterion to evaluate only features available for the provided
    # cell type.
    return nn.BCELoss()


def get_optimizer(lr):
    """
    The optimizer and the parameters with which to initialize the optimizer.
    At a later time, we initialize the optimizer by also passing in the model
    parameters (`model.parameters()`). We cannot initialize the optimizer
    until the model has been initialized.
    """
    # Option 1:
    # return (torch.optim.SGD, {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})

    # Option 2:
    return (torch.optim.Adam, {"lr": lr})
