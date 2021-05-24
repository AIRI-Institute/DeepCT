"""
DeepCT architecture without Sigmoid layer 
for multiple cell type per position computation at once (TODO: Add our names).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class DeepCT(nn.Module):
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
        Based on a DeepSEA architecture (see https://github.com/FunctionLab/selene/blob/0.4.8/models/deepsea.py)

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
        super(DeepCT, self).__init__()
        self._n_cell_types = n_cell_types
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(320, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),
            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )

        reduce_by = 2 * (conv_kernel_size - 1)
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = int(
            np.floor(
                (np.floor((sequence_length - reduce_by) / pool_kernel_size) - reduce_by)
                / pool_kernel_size
            )
            - reduce_by
        )

        self.sequence_net = nn.Sequential(
            nn.Linear(960 * self._n_channels, sequence_embedding_length),
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
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.Linear(final_embedding_length, final_embedding_length),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_embedding_length),
            nn.Linear(final_embedding_length, n_genomic_features),
            # sigmoid turned off for loss numerical stability
            # nn.Sigmoid(),
        )

    def get_cell_type_embeddings(self):
        """Retrieve cell type embeddings learned by the model."""
        device = next(self.parameters()).device
        with torch.no_grad():
            all_cell_types = torch.eye(self._n_cell_types).to(device)
            embeddings = self.cell_type_net(all_cell_types)
        return embeddings.detach().cpu()

    """
    # Doesn't take linear layer bias into account
    def log_cell_type_embeddings_to_tensorboard(self, cell_type_labels, output_dir):
        writer = SummaryWriter(output_dir)

        writer.add_embedding(
            self.cell_type_net[0].weight.transpose(0, 1), cell_type_labels
        )
        writer.flush()
        writer.close()
    """

    def forward(self, sequence_batch, cell_type_batch):
        """Forward propagation of a batch.

        Parameters:
        -----------
        sequence_batch : torch.Tensor
            A batch of encoded sequences.
        cell_type_batch: torch.Tensor
            A batch of one-hot cell type encodings.

        """
        batch_size = sequence_batch.size(0)

        cell_type_one_hots = torch.eye(self._n_cell_types).to(sequence_batch.device)

        sequence_out = self.conv_net(sequence_batch)
        reshaped_sequence_out = sequence_out.view(
            sequence_out.size(0), 960 * self._n_channels
        )
        # Repeat each sequence embedding to fit cell type embeddings.
        # E.g., with 2 cell types, [seq0_emb, seq1_emb, seq2_emb] becomes
        # [seq0_emb, seq0_emb, seq1_emb, seq1_emb, seq2_emb, seq2_emb]
        sequence_embedding = self.sequence_net(reshaped_sequence_out).repeat_interleave(
            repeats=self._n_cell_types, dim=0
        )

        # Repeat cell type embeddings to fit sequence embeddings.
        # E.g., with batch size of 3, 2 cell types, and [ct0_emb, ct1_emb] cell type
        # embeddings, the embeddings will be converted to
        # [ct0_emb, ct1_emb, ct0_emb, ct1_emb, ct0_emb, ct1_emb].
        cell_type_embeddings = self.cell_type_net(cell_type_one_hots).repeat(
            batch_size, 1
        )
        sequence_and_cell_type_embeddings = torch.cat(
            (sequence_embedding, cell_type_embeddings), 1
        )

        predict = self.classifier(sequence_and_cell_type_embeddings).view(
            batch_size, self._n_cell_types, -1
        )
        return predict


def criterion(**loss_config):
    """
    The criterion the model aims to minimize.
    """
    if "pos_weights_path" in loss_config:
        with open(loss_config["pos_weights_path"]) as f:
            pos_weight = list(map(float, f.readlines()))
        pos_weight = torch.tensor(pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


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
    return (torch.optim.Adam, {"lr": lr, "weight_decay": 1e-6})
