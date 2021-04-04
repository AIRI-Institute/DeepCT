"""
This module provides the DNaseFileSampler class.
"""
import numpy as np
from selene_sdk.samplers.file_samplers import FileSampler
from selene_sdk.samplers.samples_batch import SamplesBatch
from tqdm import trange


class DNaseFileSampler(FileSampler):
    """
    A sampler for which the dataset is loaded directly from a `*.bed` file.

    Assumes .bed file's columns are: "chrom start end strand target", separated by "\t".

    Parameters
    ----------
    filepath : str
        The path to the file to load the data from.
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    n_cell_types : int
        `n_cell_types`, the total number of cell types.

    """

    def __init__(
        self,
        filepath,
        reference_sequence,
        n_cell_types,
    ):
        """
        Constructs a new `DNaseFileSampler` object.
        """
        super().__init__()

        self.reference_sequence = reference_sequence

        self._filepath = filepath
        self._file_handle = open(self._filepath, "r")
        self._n_cell_types = n_cell_types

    def sample(self, batch_size=1):
        """
        Draws a mini-batch of examples and their corresponding labels.
        """
        intervals = []
        for _ in range(batch_size):
            line = self._file_handle.readline()
            if not line:
                # TODO(arlapin): add functionality to shuffle the file if sampler
                # reaches the end of the file.
                self._file_handle.close()
                self._file_handle = open(self._filepath, "r")
                line = self._file_handle.readline()
            intervals.append(line.split("\t"))

        sequences = []
        targets = np.zeros((batch_size, self._n_cell_types))
        for index, (chrom, start, end, strand, seq_targets) in enumerate(intervals):
            sequence = self.reference_sequence.get_encoding_from_coords(
                chrom, int(start), int(end), strand=strand
            )
            sequences.append(sequence)

<<<<<<< HEAD
            if len(seq_targets.strip()!=0):
=======
                positive_targets = [int(t) for t in seq_targets.split(";")]
                targets[index, positive_targets] = 1

        sequences = np.array(sequences)
        return SamplesBatch(sequences, target_batch=targets)

    def get_data_and_targets(self, batch_size, n_samples):
        """
        Fetches n_samples splitted into batches as well as corresponding targets.

        Returns:
        --------
        batches, all_targets : List[SamplesBatch], np.ndarray

        """
        batches = []
        all_targets = []

        count = 0
        for _ in trange((n_samples - 1) // batch_size + 1, desc="Sampling dataset"):
            to_sample = min(batch_size, n_samples - count)

            samples_batch = self.sample(batch_size=to_sample)
            batches.append(samples_batch)
            all_targets.append(samples_batch.targets())

            count += to_sample

        all_targets = np.vstack(all_targets).astype(int)

        return batches, all_targets

    def get_data(self, batch_size, n_samples=None):
        # NOTE: Looks like this method is not used.
        raise NotImplementedError()
