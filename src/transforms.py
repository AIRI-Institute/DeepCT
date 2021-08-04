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
