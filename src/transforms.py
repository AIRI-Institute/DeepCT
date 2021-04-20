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
