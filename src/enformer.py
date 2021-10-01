from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from src.attention_module import TransformerBlock
from src.enformer_utils import (
    Residual,
    SoftmaxPooling1d,
    TargetLengthCrop1d,
    exponential_linspace_int,
    maxpool1d_samepad,
)

C = 1536
INPUT_CHANNELS = 4

INPUT_LENGTH = 196608
CONTEXT_LENGTH = 40960
BIN_SIZE = 128
TARGET_LENGTH = (INPUT_LENGTH - 2 * CONTEXT_LENGTH) // BIN_SIZE


class Enformer(nn.Module):
    def __init__(
        self,
        channels=C,
        n_downres_blocks=6,
        num_transformer_layers=11,
        num_heads=8,
        pooling_type="attention",
        input_length=INPUT_LENGTH,
        context_length=CONTEXT_LENGTH,
        output_channels=5313,
        n_cell_types=None,
        multi_ct_output=True,
        cell_type_embedding_length=0,
    ):
        super().__init__()
        self.channels = channels
        self.n_downres_blocks = n_downres_blocks
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.pooling_type = pooling_type
        self.input_length = input_length
        self.n_cell_types = n_cell_types
        self.multi_ct_output = multi_ct_output
        self.cell_type_embedding_length = cell_type_embedding_length
        self.output_channels = output_channels

        self.stem = stem(
            INPUT_CHANNELS, self.channels // 2, pooling_type=self.pooling_type
        )
        
        self.conv_tower = conv_tower(
            self.channels // 2,
            self.channels,
            n_blocks=self.n_downres_blocks,
            pooling_type=self.pooling_type,
        )
        mha_kwargs = {
            "value_size": channels // num_heads,
            "key_size": 64,
            "num_heads": 8,
            "scaling": True,
            "attention_dropout_rate": 0.05,
            "relative_positions": True,
            "relative_position_symmetric": False,
            "num_relative_position_features": channels // num_heads,
            "positional_dropout_rate": 0.01,
            "zero_initialize": True,
            "sequence_length": input_length // 128,
        }
        self.transformer = transformer(
            n_blocks=self.num_transformer_layers,
            mha_kwargs=mha_kwargs,
            in_channels=self.channels,
            channels=self.channels,
        )
        self.target_length = (self.input_length - 2 * context_length) // BIN_SIZE
        self.pointwise_block = pointwise_block(
            self.channels, self.target_length, self.channels
        )

        if self.n_cell_types is not None:
            self.cell_type_net = nn.Sequential(
                nn.Linear(self.n_cell_types, self.cell_type_embedding_length),
            )
        self.output_head = output_head(
            2 * self.channels + self.cell_type_embedding_length, self.output_channels
        )
        
        # hack for checkpointing
        self.stem = ModuleWrapperIgnores2ndArg(self.stem)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x, *args):
        # N, C, L = x.shape
        x = checkpoint(self.stem, x, self.dummy_tensor)  # (N, C // 2, L // 2)
        x = checkpoint_sequential(self.conv_tower, self.n_downres_blocks, x)  # (N, C, L // 128)
        x = x.transpose(-1, -2)  # (N, L // 128, C)
        x = checkpoint_sequential(self.transformer, self.num_transformer_layers, x)  # (N, L // 128, C)
        x = x.transpose(-1, -2)  # (N, C, L // 128)
        x = checkpoint(self.pointwise_block, x)  # (N, 2C, TARGET_LENGTH)
        if self.n_cell_types is not None:
            if self.multi_ct_output:
                output_cat_size = (
                    x.shape[0] * self.n_cell_types, 
                    x.shape[1] + self.cell_type_embedding_length, 
                    x.shape[-1]
                )
                cell_type_one_hots = torch.eye(self.n_cell_types, device=x.device)
            else:
                output_cat_size = (
                    x.shape[0], 
                    x.shape[1] + self.cell_type_embedding_length, 
                    x.shape[-1]
                )
                cell_type_one_hots = args[0]
            
            output_cat = torch.empty(*output_cat_size, dtype=x.dtype, device=x.device)
            
            cell_type_embeddings = self.cell_type_net(cell_type_one_hots)
            # Broadcast cell type embeddings to TARGET_LENGTH for concatenation with output
            cell_type_embeddings = cell_type_embeddings.unsqueeze(2).repeat(1, 1, 896)

            if self.multi_ct_output:
                batch_size = x.shape[0]
                """
                # This hack saves a lot of GPU memory, 
                # but significantly increases `loss.backward()` time

                #cell_type_embeddings = cell_type_embeddings.unsqueeze(2).repeat(1, 1, 896)
                # Avoid using `torch.tensor.repeat` not to use additional GPU memory
                for i in range(batch_size):
                    for j in range(self.n_cell_types):
                        sample_idx = i * self.n_cell_types + j
                        output_cat[sample_idx, :x.shape[1], :] = x[i]
                        for k in range(self.target_length):
                            output_cat[sample_idx, x.shape[1]:, k] = cell_type_embeddings[j, :]
                x = output_cat
                """
                # Repeat cell type embeddings to fit sequence embeddings.
                # E.g., with batch size of 3, 2 cell types, and [ct0_emb, ct1_emb]
                # cell type embeddings, the embeddings will be converted to
                # [ct0_emb, ct1_emb, ct0_emb, ct1_emb, ct0_emb, ct1_emb].
                cell_type_embeddings = cell_type_embeddings.repeat(batch_size, 1, 1)

                # Repeat each sequence embedding to fit cell type embeddings.
                # E.g., with 2 cell types, [seq0_emb, seq1_emb, seq2_emb] becomes
                # [seq0_emb, seq0_emb, seq1_emb, seq1_emb, seq2_emb, seq2_emb]
                x = x.repeat_interleave(self.n_cell_types, dim=0)

            output_cat[:, :x.shape[1], :] = x
            output_cat[:, x.shape[1]:, :] = cell_type_embeddings
            x = output_cat

        x = x.transpose(-1, -2)  # (N, TARGET_LENGTH, 2C)
        x = self.output_head(x)  # (N, TARGET_LENGTH, output_channels)
        if self.multi_ct_output:
            return x.view(
                -1, self.n_cell_types, self.target_length, self.output_channels
            )
        else:
            return x.view(
                -1, self.target_length, self.output_channels
            ) 

    def get_cell_type_embeddings(self):
        """Retrieve cell type embeddings learned by the model."""
        assert self.n_cell_types is not None
        device = next(self.parameters()).device
        with torch.no_grad():
            all_cell_types = torch.eye(self.n_cell_types).to(device)
            embeddings = self.cell_type_net(all_cell_types)
        return embeddings.detach().cpu()


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


def conv_block(in_channels, out_channels, kernel_size):
    pad_size = kernel_size // 2
    return nn.Sequential(
        nn.BatchNorm1d(in_channels, momentum=0.1),
        nn.GELU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad_size),
    )


def rconv_block(in_channels, out_channels, kernel_size):
    conv = conv_block(in_channels, out_channels, kernel_size)
    return Residual(conv)


def pooling_module(kind, kernel_size, n_channels=None, input_len=None):
    """Pooling module wrapper."""
    if kind == "attention":
        return SoftmaxPooling1d(
            n_channels=n_channels,
            kernel_size=kernel_size,
            per_channel=True,
            w_init_scale=2.0,
        )
    elif kind == "max":
        return maxpool1d_samepad(input_len, kernel_size=kernel_size)
    else:
        raise ValueError(f"Invalid pooling kind: {kind}.")


def stem(
    in_channels,
    conv_out_channels,
    conv_kernel=15,
    rconv_kernel=1,
    pooling_type="attention",
    name_suffix="",
):
    pad_size = conv_kernel // 2
    conv = nn.Conv1d(in_channels, conv_out_channels, conv_kernel, padding=pad_size)

    rconv = rconv_block(
        in_channels=conv_out_channels,
        out_channels=conv_out_channels,
        kernel_size=rconv_kernel,
    )

    pool = pooling_module(pooling_type, kernel_size=2, n_channels=conv_out_channels)

    return nn.Sequential(
        OrderedDict(
            [
                (f"conv{name_suffix}", conv),
                (f"rconv_block{name_suffix}", rconv),
                (f"pooling{name_suffix}", pool),
            ]
        )
    )


def downres_block(
    in_channels,
    conv_out_channels,
    conv_kernel=15,
    rconv_kernel=1,
    pooling_type="attention",
    name_suffix="",
):
    # conv_out_channels = n_channels // 2
    conv = conv_block(
        in_channels=in_channels,
        out_channels=conv_out_channels,
        kernel_size=conv_kernel,
    )

    rconv = rconv_block(
        in_channels=conv_out_channels,
        out_channels=conv_out_channels,
        kernel_size=rconv_kernel,
    )

    pool = pooling_module(pooling_type, kernel_size=2, n_channels=conv_out_channels)

    return nn.Sequential(
        OrderedDict(
            [
                (f"conv_block{name_suffix}", conv),
                (f"rconv_block{name_suffix}", rconv),
                (f"pooling{name_suffix}", pool),
            ]
        )
    )


def conv_tower(in_channels, out_channels, n_blocks=6, pooling_type="attention"):
    conv_channels = exponential_linspace_int(
        start=in_channels, end=out_channels, num=n_blocks, divisible_by=128
    )
    conv_channels = [in_channels] + conv_channels
    downres_blocks = OrderedDict(
        [
            (
                f"downres_block_{i}",
                downres_block(
                    conv_channels[i],
                    conv_channels[i + 1],
                    conv_kernel=5,
                    pooling_type=pooling_type,
                    # name_suffix=f"_{i}",
                ),
            )
            for i in range(len(conv_channels) - 1)
        ]
    )
    return nn.Sequential(downres_blocks)


def transformer(n_blocks, mha_kwargs, in_channels, channels):
    blocks = [
        (
            f"transformer_block_{i}",
            TransformerBlock(in_channels, channels, mha_kwargs),
        )
        for i in range(n_blocks)
    ]
    return nn.Sequential(OrderedDict(blocks))


def pointwise_block(in_channels, crop_target_length, channels, dropout_rate=0.05):
    return nn.Sequential(
        OrderedDict(
            [
                ("crop", TargetLengthCrop1d(crop_target_length)),
                ("conv_block", conv_block(in_channels, 2 * channels, kernel_size=1)),
                ("dropout", nn.Dropout(p=dropout_rate)),
                ("gelu", nn.GELU()),
            ]
        )
    )


def output_head(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.Softplus())


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
