import numpy as np
import torch


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


class quantitative2sigmoid(torch.nn.Module):
    """
    Maps quantitative features to the interval -1...1 using sigmoid function

    Parameters
    ----------
    input:  np.ndarray
        features to be converted.

    threashold:  float
        threashold substracted from feature values before applying sigmoid function.
    """

    def __init__(self, threashold=4.9):
        super().__init__()
        self.threashold = threashold

    def forward(self, input):
        return torch.sigmoid(input - self.threashold)


class quantitative2qualitative(torch.nn.Module):
    """
    Maps quantitative features to the interval -1...1 using sigmoid function

    Parameters
    ----------
    input:  np.ndarray
        features to be converted.

    threashold:  float
        threashold substracted from feature values before applying sigmoid function.
    """

    def __init__(self, threashold=4.9):
        super().__init__()
        self.threashold = threashold

    def forward(self, input):
        return input > self.threashold
