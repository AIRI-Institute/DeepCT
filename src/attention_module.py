from collections import OrderedDict

import torch
import torch.nn as nn

from .enformer_utils import Residual, variance_scaling_init_


def positional_features_all(
    positions,
    feature_size,
    seq_length=None,
    bin_size=None,
    feature_functions=None,
    symmetric=False,
):
    """Compute relative positional encodings/features.

    Each positional feature function will compute/provide the same fraction of
    features, making up the total of feature_size.

    Args:
    positions: Tensor of relative positions of arbitrary shape.
    feature_size: Total number of basis functions.
    seq_length: Sequence length denoting the characteristic length that
      the individual positional features can use. This is required since the
      parametrization of the input features should be independent of `positions`
      while it could still require to use the total number of features.
    bin_size: Bin sized used to partition the sequence. This can be used to
      compute features on the absolute scale relative to the genome.
    feature_functions: List of different feature functions to use. Each function
      will take as argument: positions, sequence length and number of features
      to compute.
    symmetric: If True, the resulting features will be symmetric across the
      relative position of 0 (i.e. only absolute value of positions will
      matter). If false, then both the symmetric and asymmetric version
      (symmetric multiplied by sign(positions)) of the features will be used.

    Returns:
    Tensor of shape: `positions.shape + (feature_size,)`.
    """

    if feature_functions is None:
        feature_functions = [
            positional_features_exponential,
            positional_features_central_mask,
            positional_features_gamma,
        ]
    num_components = len(feature_functions)  # 1 per each basis function
    if not symmetric:
        num_components = 2 * num_components

    # For now, we do not allow odd sized embeddings.
    if feature_size % num_components != 0:
        raise ValueError(f"feature_size has to be divisible by {num_components}")

    num_basis_per_class = feature_size // num_components
    embeddings = [
        f(torch.abs(positions), num_basis_per_class, seq_length, bin_size)
        for f in feature_functions
    ]
    embeddings = torch.cat(embeddings, dim=-1)
    if not symmetric:
        embeddings = torch.cat(
            [embeddings, torch.sign(positions).unsqueeze(-1) * embeddings], dim=-1
        )
    return embeddings


# cannot be replaced by torch.distribution.gamma.Gamma
# which can't be evaluated at 0
def gamma_pdf(x, concentration, rate):
    """Gamma probability distribution function: p(x|concentration, rate)."""
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(
    positions,
    feature_size,
    seq_length=None,
    bin_size=None,
    stddev=None,
    start_mean=None,
):
    """Positional features computed using the gamma distributions."""
    del bin_size  # Unused.
    if seq_length is None:
        seq_length = torch.max(torch.abs(positions)) + 1
    if stddev is None:
        stddev = seq_length / (2 * feature_size)
    if start_mean is None:
        start_mean = seq_length / feature_size
    mean = torch.linspace(start_mean, seq_length, feature_size)
    mean = mean[(None,) * positions.ndim]

    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probabilities = gamma_pdf(torch.abs(positions).unsqueeze(-1), concentration, rate)
    probabilities += 1e-8  # To ensure numerical stability.
    outputs = probabilities / torch.max(probabilities)
    # tf.TensorShape(outputs.shape).assert_is_compatible_with(
    #  positions.shape + [feature_size])
    return outputs


def positional_features_central_mask(
    positions, feature_size, seq_length=None, bin_size=None
):
    """Positional features using a central mask (allow only central features)."""
    del seq_length  # Unused.
    del bin_size  # Unused.
    center_widths = torch.pow(2.0, torch.arange(1, feature_size + 1, dtype=torch.float))
    # very weird, should be +1, according to formula from the paper
    center_widths = center_widths - 1
    center_widths = center_widths[(None,) * positions.ndim]
    outputs = (center_widths > torch.abs(positions).unsqueeze(-1)).type(torch.float)
    return outputs


def positional_features_exponential(
    positions, feature_size, seq_length=None, bin_size=None, min_half_life=3.0
):
    """Create exponentially decaying positional weights.

    Args:
    positions: Position tensor (arbitrary shape).
    feature_size: Number of basis functions to use.
    seq_length: Sequence length.
    bin_size: (unused). See `positional_features_all`.
    min_half_life: Smallest exponential half life in the grid of half lives.

    Returns:
    A Tensor with shape [2 * seq_length - 1, feature_size].
    """
    del bin_size  # Unused.
    if seq_length is None:
        seq_length = torch.max(torch.abs(positions)) + 1
    # this is identical to Enformer tf code, but their comment is wrong, as it produces
    # the grid of half lifes from [2**3, seq_length] distributed on the log scale,
    # NOT [3, sequence_length] as they state in the paper, and
    # NOT [3, seq_length / 2] as they state in their code comments
    max_range = torch.log(torch.tensor(seq_length)) / torch.log(torch.tensor(2.0))
    half_life = torch.pow(2.0, torch.linspace(min_half_life, max_range, feature_size))

    # approach #1 to produce a grid from [3, seq_length] on the log scale
    # min_range = torch.log(torch.tensor(min_half_life)) / torch.log(torch.tensor(2.0))
    # half_life = torch.pow(2.0, torch.linspace(min_range, max_range, feature_size))

    # approach #2 to produce a grid from [3, seq_length] on the log scale
    # min_range = torch.log(torch.tensor(min_half_life))
    # max_range = torch.log(torch.tensor(seq_length))
    # from math import e
    # half_life = torch.logspace(min_range, max_range, feature_size, base=e)

    half_life = half_life[(None,) * positions.ndim]
    positions = torch.abs(positions)
    outputs = torch.exp(
        -torch.log(torch.tensor(2.0)) / half_life * positions.unsqueeze(-1)
    )
    return outputs


def relative_shift(x):
    """Shift the relative logits like in TransformerXL."""
    # We prepend zeros on the final timescale dimension.
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = x.view(-1, num_heads, t2, t1)
    x = x[:, :, 1:, :]
    x = x.view(-1, num_heads, t1, t2 - 1)
    x = x[:, :, :, : (t2 + 1) // 2]
    return x


class MultiheadAttentionRPE(nn.Module):
    """MultiheadAttention module.

    Args:
        value_size: The size of each value embedding per head.
        key_size: The size of each key and query embedding per head.
        num_heads: The number of independent queries per timestep.
        scaling: Whether to scale the attention logits.
        attention_dropout_rate: Dropout rate for attention logits.
        relative_positions: Whether to use TransformerXL style relative attention.
        relative_position_symmetric: If True, the symmetric version of basis
        functions will be used. If False, a symmetric and asymmetric versions
        will be use.
        relative_position_functions: List of functions used for relative
        positional biases.
        num_relative_position_features: Number of relative positional features
        to compute. If None, `value_size * num_heads` is used.
        positional_dropout_rate: Dropout rate for the positional encodings if
        relative positions are used.
        zero_initialize: if True, the final linear layer will be 0 initialized.
        initializer: Initializer for the projection layers. If unspecified,
        VarianceScaling is used with scale = 2.0.
        name: Name of module.
    """

    def __init__(
        self,
        value_size,
        key_size,
        num_heads,
        n_channels,
        scaling=True,
        attention_dropout_rate=0.1,
        relative_positions=False,
        relative_position_symmetric=False,
        relative_position_functions=None,
        num_relative_position_features=None,
        positional_dropout_rate=0.1,
        zero_initialize=True,
        initializer=variance_scaling_init_,
        sequence_length=None,
    ):
        super().__init__()
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._relative_positions = relative_positions
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        if num_relative_position_features is None:
            # num_relative_position_features needs to be divisible by the number of
            # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_relative_position_features = (
                self._value_size // divisible_by
            ) * divisible_by
        else:
            self._num_relative_position_features = num_relative_position_features
        self._positional_dropout_rate = positional_dropout_rate
        self._initializer = initializer

        key_proj_size = self._key_size * self._num_heads
        embedding_size = self._value_size * self._num_heads

        self.q_layer = nn.Linear(
            n_channels,
            key_proj_size,
            bias=False,
        )
        self._initializer(self.q_layer.weight)

        self.k_layer = nn.Linear(
            n_channels,
            key_proj_size,
            bias=False,
        )
        self._initializer(self.k_layer.weight)

        self.v_layer = nn.Linear(n_channels, embedding_size, bias=False)
        self._initializer(self.v_layer.weight)

        w_init = nn.init.zeros_ if zero_initialize else self._initializer
        self.embedding_layer = nn.Linear(embedding_size, embedding_size)
        w_init(self.embedding_layer.weight)
        w_init(self.embedding_layer.bias)

        # Create additional layers if using relative positions.
        if self._relative_positions:
            self.r_k_layer = nn.Linear(
                self._num_relative_position_features, key_proj_size, bias=False
            )
            self._initializer(self.r_k_layer.weight)

            self._r_w_bias = nn.Parameter(
                torch.empty((1, self._num_heads, 1, self._key_size), dtype=torch.float)
            )
            self._initializer(self._r_w_bias)

            self._r_r_bias = nn.Parameter(
                torch.empty((1, self._num_heads, 1, self._key_size), dtype=torch.float)
            )
            self._initializer(self._r_r_bias)

            self.positional_dropout = nn.Dropout(p=self._positional_dropout_rate)
            self.attention_dropout = nn.Dropout(p=self._attention_dropout_rate)
        self.sequence_length = sequence_length
        if self._relative_positions and self.sequence_length is not None:
            positional_encodings = self._positional_encodings(self.sequence_length)
            self.register_buffer("positional_encodings", positional_encodings)
        else:
            self.positional_encodings = None

    def _positional_encodings(self, sequence_length):
        # For relative positions, we project positions to form relative keys.
        distances = torch.unsqueeze(
            torch.arange(
                -sequence_length + 1,
                sequence_length,
                dtype=torch.float,
                requires_grad=False,
            ),
            dim=0,
        )  # (1, 2L - 1)
        return positional_features_all(
            positions=distances,
            feature_size=self._num_relative_position_features,
            seq_length=sequence_length,
            feature_functions=self._relative_position_functions,
            symmetric=self._relative_position_symmetric,
        )  # (1, 2L - 1, Cr)

    def _multihead_output(self, linear, inputs):
        """Applies a standard linear to inputs and returns multihead-shape output."""
        # switch batch and length dimensions
        # (N, L, E) -> (L, N, E)
        inputs = inputs.transpose(0, 1)

        output = linear(inputs)  # (L, N, H * KV)
        num_kv_channels = output.shape[-1] // self._num_heads
        # Split H * Channels into separate axes.
        output = output.view(
            output.shape[0], output.shape[1], self._num_heads, num_kv_channels
        )  # (L, N, H, KV)
        # (L, N, H, KV) -> (N, H, L, KV)
        return output.permute((1, 2, 0, 3))

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        embedding_size = self._value_size * self._num_heads

        # Compute q, k and v as multi-headed projections of the inputs.
        q = self._multihead_output(self.q_layer, inputs)  # (N, H, L, K)
        k = self._multihead_output(self.k_layer, inputs)  # (N, H, L, K)
        v = self._multihead_output(self.v_layer, inputs)  # (N, H, L, V)

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size ** -0.5

        # import pdb; pdb.set_trace()
        if self._relative_positions:
            if self.sequence_length == seq_len:
                # print('Using pre-computed encodings')
                # use pre-computed encodings for specific sequence length
                positional_encodings = self.positional_encodings
            else:
                # print('Computing encodings')
                # compute encodings for provided sequence length
                positional_encodings = self._positional_encodings(seq_len).to(
                    self.r_k_layer.weight.device
                )

            positional_encodings = self.positional_dropout(positional_encodings)

            # (1, H, 2L - 1, K)
            r_k = self._multihead_output(self.r_k_layer, positional_encodings)

            # Add shifted relative logits to content logits.
            # (N, H, L, L)
            content_logits = torch.matmul(q + self._r_w_bias, k.transpose(-1, -2))
            # (N, H, L, 2L - 1)
            relative_logits = torch.matmul(q + self._r_r_bias, r_k.transpose(-1, -2))
            # (N, H, L, L)
            relative_logits = relative_shift(relative_logits)
            logits = content_logits + relative_logits
        else:
            # (N, H, L, L)
            logits = torch.matmul(q, k, transpose_b=True)

        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.attention_dropout(weights)

        # Transpose and reshape the output.
        output = torch.matmul(weights, v)  # (N, H, L, V)
        output_transpose = output.permute((0, 2, 1, 3))  # (N, L, H, V)

        # Final linear layer.
        global b
        b = output_transpose
        attended_inputs = output_transpose.reshape(
            output_transpose.shape[0], output_transpose.shape[1], embedding_size
        )  # (N, L, H * V)
        output = self.embedding_layer(attended_inputs)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, channels, mha_kwargs, dropout_rate=0.4):
        super().__init__()
        mha_ln = nn.LayerNorm(in_channels)
        mha = MultiheadAttentionRPE(n_channels=in_channels, **mha_kwargs)
        mha_dropout = nn.Dropout(p=dropout_rate)
        self.mha_block = Residual(
            nn.Sequential(
                OrderedDict(
                    [
                        ("layer_norm", mha_ln),
                        ("multihead_attention", mha),
                        ("dropout", mha_dropout),
                    ]
                )
            )
        )

        mha_output_size = mha_kwargs["num_heads"] * mha_kwargs["value_size"]
        mlp_ln = nn.LayerNorm(mha_output_size)
        mlp_linear1 = nn.Linear(mha_output_size, channels * 2)
        mlp_dropout1 = nn.Dropout(dropout_rate)
        relu = nn.ReLU()
        mlp_linear2 = nn.Linear(channels * 2, channels)
        mlp_dropout2 = nn.Dropout(dropout_rate)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("layer_norm", mlp_ln),
                    ("linear_1", mlp_linear1),
                    ("dropout_1", mlp_dropout1),
                    ("relu", relu),
                    ("linear_2", mlp_linear2),
                    ("dropout_2", mlp_dropout2),
                ]
            )
        )

    def forward(self, inputs):
        mha_output = self.mha_block(inputs)
        x = self.mlp(mha_output)
        return x + mha_output
