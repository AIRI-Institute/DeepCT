from itertools import starmap

import numpy as np
import torch
import torchvision


class PermuteSequenceChannels(torch.nn.Module):
    """
    Permute channels of retrieved encoded DNA sequence.

    Permutes axes of `np.array` of shape `(sequence_length, alphabet_size)` to
    obtain `np.array` of shape `(alphabet_size, sequence_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        seq = sample[0]
        perm_seq = np.transpose(seq)
        return (perm_seq, *sample[1:])


class RandomReverseStrand(torch.nn.Module):
    """
    Randomly change the strand of retrieved encoded DNA sequence.
    !!! Only works if DNA letter encodings of complementary letters are mirrors
    of each other, e.g. if `[1, 0, 0, 0]` is an encoding of letter A, letter T should
    have encoding `[0, 0, 0, 1]`. Selene letter encodings are like this by default.

    Randomly flips the contents of encoded sequence along all available axes to
    obtain a sequence identical to the one taken from a different DNA strand

    Parameters
    ----------
    p:  float
        Probability that the strand should be reversed.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        seq = sample[0]
        if torch.rand(1) < self.p:
            return (np.flip(seq).copy(), *sample[1:])
        return sample


class MaskFeatures(torch.nn.Module):
    """
    Masks features given by `feature_indices_mask`.

    Sets sample mask of features at given indices to `False` so that
    the loss sample weights set by this mask become 0's

    Parameters
    ----------
    feature_indices_mask:  np.ndarray
        Indices of features to be masked out
    """

    def __init__(self, feature_indices_mask):
        super().__init__()
        self.idx_to_mask = feature_indices_mask

    def forward(self, sample):
        mask = sample[3]
        mask[..., self.idx_to_mask] = False
        return (*sample[:3], mask)


class MaskTracks(torch.nn.Module):
    """
    Masks tracks given by `track_mask`.

    Sets sample mask of given tracks to `False` so that
    the loss sample weights set by this mask become 0's

    Parameters
    ----------
    track_mask:  np.ndarray
        Mask of tracks to apply.
    """

    def __init__(self, track_mask, reverse_mask=False):
        super().__init__()
        self.reverse_mask = reverse_mask
        if self.reverse_mask:
            self.track_mask = ~track_mask
        else:
            self.track_mask = track_mask

    def forward(self, sample):
        mask = sample[3]
        mask[self.track_mask] = False
        return (*sample[:3], mask)


class LogTargets(torch.nn.Module):
    """
    Log targets values
    note that targets will be first incremented by a pseudocount
    to process zero target values

    Parameters
    ----------
    pseudocount:  float
        pseudocount value
    """

    def __init__(self, pseudocount=0):
        super().__init__()
        self.pseudocount = pseudocount

    def forward(self, sample):
        targets = np.log(sample[2] + self.pseudocount)
        return (*sample[:2], targets, sample[3])


class ClipTargets(torch.nn.Module):
    """
    Clip (limit) the values in targets array.

    Parameters
    ----------
    amin, amax:  float or None
        Minimum and maximum value. If None, clipping is not
        performed on the corresponding edge. Only one of a_min and a_max may be None.
        Both are broadcast against a (see numpy.clip).
    """

    def __init__(self, amin=None, amax=None):
        super().__init__()
        self.amin = amin
        self.amax = amax

    def forward(self, sample):
        targets = np.clip(sample[2], self.amin, self.amax)
        return (*sample[:2], targets, sample[3])


class ArrayTransform(torch.nn.Module):
    """
    Abstract class allowing to apply same transform
    to several arrays, typically predict, feature and target

    should define function F which will be applied to arrayTransform

    Parameters
    ----------
    transform_predictions, transform_targets, transform_masks: bool
        define which arrays should be tranformed
    """

    def __init__(
        self, transform_predictions=True, transform_targets=True, transform_masks=False
    ):
        super().__init__()
        self.transform_predictions = transform_predictions
        self.transform_targets = transform_targets
        self.transform_masks = transform_masks

    def F(self, input):
        raise NotImplementedError(
            "Classes that inherit from `ArrayTransform` must define this method"
        )

    def forward(self, inputs):
        prediction, target, target_mask = inputs
        if self.transform_predictions:
            tr_prediction = self.F(prediction)
        else:
            tr_prediction = prediction

        if self.transform_targets:
            tr_target = self.F(target)
        else:
            tr_target = target

        if self.transform_masks and target_mask is not None:
            tr_target_mask = self.F(target_mask)
        else:
            tr_target_mask = target_mask

        return tr_prediction, tr_target, tr_target_mask


class Quantitative2Sigmoid(ArrayTransform):
    """
    Maps quantitative features to the interval 0...1 using sigmoid function

    Parameters
    ----------
        threshold:  float
        threshold substracted from feature values before applying sigmoid function.
    """

    def __init__(self, threshold=4.9, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def F(self, x):
        return map(lambda y: torch.sigmoid(y - self.threshold), x)


class Quantitative2Qualitative(ArrayTransform):
    """
    Converts quantitative features to the binary (i.e. Yes/No) values
    based on specific threshold, i.e. binary = input > threshold

    Parameters
    ----------
    threshold:  float
        threshold substracted from feature values before applying sigmoid function.
    """

    def __init__(self, threshold=4.9, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def F(self, x):
        return map(lambda y: y > self.threshold, x)


class MeanAndDeviation2AbsolutePredication(ArrayTransform):
    """
    Convert targets from mean_positional_value and cell-type specific deviations
    to the form of absolute cell-type specific value, i.e
    result = mean_positional_value + cell-type specific deviations

    Note that this will reshape output from
    [batch_size,n_cell_types+1,n_features]
    to
    [batch_size,n_cell_types,n_features]

    Parameters
    ----------
    input:  np.ndarray
        predictions to be converted.
    mean_scaling: float
        multiply mean by mean_scaling value (default = 1)
    deviation_scaling: float
        multiply deviation by mean_deviation value (default = 1)
    """

    def __init__(self, mean_scaling=1, deviation_scaling=1, **kwargs):
        super().__init__(**kwargs)
        self._mean_scaling = mean_scaling
        self._deviation_scaling = deviation_scaling

    def mean_and_dev2value(self, prediction):
        means = prediction[:, -1:, :] * self._mean_scaling
        deviations = prediction[:, :-1, :] * self._deviation_scaling
        return means + deviations

    def F(self, x):
        return map(self.mean_and_dev2value, x)


class MeanAverageValueBasedPredictor(torch.nn.Module):
    """
    # infer predictions from target's mean positional values
    # and replace predictions with these inferred values
    # !!! WARNING !!!
    # This is direct flow from targets
    Parameters
    ----------
    input:  TODO: add descritption
        predictions to be converted.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        def get_batch_MPV(target, target_mask):
            num_cell_types = target.size()[1]
            _mean_feature_value = torch.sum(
                target * target_mask, 1, keepdim=True
            ) / torch.sum(target_mask, 1, keepdim=True)
            _mean_feature_value = _mean_feature_value.repeat(1, num_cell_types, 1)
            return _mean_feature_value

        _, target, target_mask = inputs
        return starmap(get_batch_MPV, zip(target, target_mask)), target, target_mask


class Concat_batches(ArrayTransform):
    """
    Concatenate all batches and convert to numpy array
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def F(self, x):
        items = tuple(x)
        if torch.is_tensor(items[0]):
            items = tuple(map(lambda y: y.cpu().detach().numpy(), items))
        result = np.concatenate(items)
        return result


# Define a few useful transforms
QUANTITAVE_PREDICTION_THRESHOLD = 1.4816

quant2prob_transform = torchvision.transforms.Compose(
    [
        Quantitative2Sigmoid(
            transform_predictions=True,
            transform_targets=False,
            threshold=QUANTITAVE_PREDICTION_THRESHOLD,
        ),  # transform predictions
        Quantitative2Qualitative(
            transform_predictions=True,
            transform_targets=False,
            threshold=QUANTITAVE_PREDICTION_THRESHOLD,
        ),  # transform targets
        Concat_batches(
            transform_predictions=True,
            transform_targets=True,
            transform_masks=True,
        ),
    ]
)

preds2mpv_transform = torchvision.transforms.Compose(
    [
        MeanAverageValueBasedPredictor(),
        quant2prob_transform,
    ]
)

base_transform = Concat_batches(
    transform_predictions=True,
    transform_targets=True,
    transform_masks=True,
)
