import numpy as np
import torch
import torch.nn as nn


def _compute_fans(shape):
    """Computes the number of input and output units for a weight shape.
    Args:
        shape: Integer shape tuple .
    Returns:
        A tuple of scalars `(fan_in, fan_out)`.
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1.0
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def variance_scaling_init_(tensor, scale=2.0):
    fan_in, fan_out = _compute_fans(tensor.shape)
    scale = torch.tensor(scale)
    with torch.no_grad():
        scale /= max(1.0, fan_in)
        # ??? No idea why this is happening, but this is in the original VarianceScaling
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        distribution_stddev = 0.87962566103423978
        stddev = scale ** 0.5 / distribution_stddev

        # these two approaches produce identical distributions
        nn.init.trunc_normal_(tensor, mean=0.0, std=stddev, a=-2 * stddev, b=2 * stddev)
        # tensor.data = torch.from_numpy(
        #     truncnorm.rvs(-2., 2., loc=0.0, scale=stddev, size=tensor.shape)
        # )
    return tensor


def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""

    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)

    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base ** i) for i in range(num)]


class Residual(nn.Module):
    """Residual block."""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, inputs, *args, **kwargs):
        return inputs + self._module(inputs, *args, **kwargs)


class SoftmaxPooling1d(nn.Module):
    """Pooling operation with optional weights."""

    def __init__(
        self,
        n_channels,
        kernel_size=2,
        per_channel=False,
        w_init_scale=0.0,
    ):
        """Softmax pooling.

        Args:
          kernel_size: Pooling size, same as in Max/AvgPooling.
          per_channel: If True, the logits/softmax weights will be computed for
            each channel separately. If False, same weights will be used across all
            channels.
          w_init_scale: When 0.0 is equivalent to avg pooling, and when
            ~2.0 and `per_channel=False` it's equivalent to max pooling.
          name: Module name.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._per_channel = per_channel
        self._w_init_scale = w_init_scale
        out_features = n_channels if self._per_channel else 1
        self._logit_linear = nn.Linear(n_channels, out_features, bias=False)
        variance_scaling_init_(self._logit_linear.weight, scale=self._w_init_scale)

    def forward(self, inputs):
        _, n_channels, length = inputs.shape
        inputs = torch.transpose(inputs, -1, -2)  # (batch_size, length, n_channels)
        inputs = inputs.view(
            -1, length // self._kernel_size, self._kernel_size, n_channels
        )

        pool_res = torch.sum(
            inputs * torch.softmax(self._logit_linear(inputs), axis=-2), axis=-2
        )
        return torch.transpose(pool_res, -1, -2)


def maxpool1d_samepad(input_len, kernel_size=2, stride=1, padding=0, dilation=1):
    total_padsize = (stride - 1) * (input_len - 1) + dilation * (kernel_size - 1)
    left_pad = total_padsize // 2
    right_pad = left_pad + total_padsize % 2

    pad_layer = nn.ConstantPad1d((left_pad, right_pad), 0.0)
    pool_layer = nn.MaxPool1d(
        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    return nn.Sequential(pad_layer, pool_layer)


class TargetLengthCrop1d(nn.Module):
    """Crop sequence to match the desired target length."""

    def __init__(self, target_length):
        super().__init__()
        self._target_length = target_length

    def __call__(self, inputs):
        trim = (inputs.shape[-1] - self._target_length) // 2
        if trim < 0:
            print(inputs.shape[-1], self._target_length)
            raise ValueError("inputs shorter than target length")
        return inputs[..., trim:-trim]
