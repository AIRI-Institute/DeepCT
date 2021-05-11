"""
This module provides the RPKMFileSampler class.
"""
import numpy as np
from selene_sdk.samplers.file_samplers import FileSampler
from selene_sdk.samplers.samples_batch import SamplesBatch
from tqdm import trange


class RPKMFileSampler(FileSampler):
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
    sequence_length : int
        Length of one-hot sequence to pass to the model
    balance : bool
        If set to True, enables undersampling of samples with target specified
        by parameter 'zero_expression' (useful for balancing a dataset with many zeros).
    zero_expression : str
        The value in the data that represents 0.0 expression (here given as string).
        The original 0.0 value may be replaced by a different constant due to a
        logarithmically transformed dataset (with a small offset).
    keep_zero_percent : float
        The percentage of samples with targets matching parameter 'zero_expression'
        to train on.
    """

    def __init__(
        self,
        filepath,
        reference_sequence,
        n_cell_types,
        sequence_length,
        balance=False,
        zero_expression=None,
        keep_zero_percent=None,
    ):
        """
        Constructs a new `RPKMFileSampler` object.
        """
        super().__init__()

        self.reference_sequence = reference_sequence

        self._filepath = filepath
        self._file_handle = open(self._filepath, "r")
        self._n_cell_types = n_cell_types
        self._sequence_length = sequence_length

        if balance:
            assert isinstance(
                zero_expression, str
            ), "'zero_expression' parameter must be passed as a string to RPKMFileSampler when 'balance' flag is set to True."
            assert isinstance(
                keep_zero_percent, float
            ), "'keep_zero_percent' parameter must be passed as a float to RPKMFileSampler when 'balance' flag is set to True."

        self.balance = balance
        self.zero_expression = zero_expression
        self.keep_zero_percent = keep_zero_percent

    def sample(self, batch_size=1):
        """
        Draws a mini-batch of examples and their corresponding labels.

        Parameters
        ----------
        batch_size : int
            size of the mini-batch
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
        targets = []

        # target_cells is a mask that marks which cell types are trained on for each sample
        # in the mini-batch. The selected cell types are marked with '1'.
        target_cells = np.zeros((batch_size, self._n_cell_types), dtype=np.int64)

        for index, (chrom, start, end, strand, seq_targets) in enumerate(intervals):
            start = int(start)
            end = int(end)

            if strand == "+":  # promoter ends at start position - center around start
                start_seq = start - self._sequence_length // 2
                end_seq = start + self._sequence_length // 2
            else:
                start_seq = end - self._sequence_length // 2
                end_seq = end + self._sequence_length // 2

            sequence = self.reference_sequence.get_encoding_from_coords(
                chrom,
                start_seq,
                end_seq,
                strand=strand,
                pad=True,  # padding due to long sequence length around promoter
            )

            for i, t in enumerate(seq_targets.split(";")):
                if self.balance and t == self.zero_expression:
                    if np.random.uniform() < self.keep_zero_percent:
                        targets.append(float(t))
                        target_cells[index, i] = 1
                else:
                    targets.append(float(t))
                    target_cells[index, i] = 1

            sequences.append(sequence)

        sequences = np.array(sequences)
        targets = np.expand_dims(np.array(targets), axis=1)

        return SamplesBatch(sequences, target_batch=targets), target_cells

    def get_data_and_targets(self, batch_size, n_samples):
        """
        Fetches n_samples splitted into batches as well as corresponding targets.

        Returns:
        --------
        batches, all_targets, all_cell_targets : List[SamplesBatch], np.ndarray, List[np.ndarray]

        """
        batches = []
        all_targets = []
        all_cell_targets = []

        count = 0
        for _ in trange((n_samples - 1) // batch_size + 1, desc="Sampling dataset"):
            to_sample = min(batch_size, n_samples - count)

            samples_batch, target_cells = self.sample(batch_size=to_sample)
            batches.append(samples_batch)
            all_targets.append(samples_batch.targets())
            all_cell_targets.append(target_cells)

            count += to_sample

        all_targets = np.vstack(all_targets)

        return batches, all_targets, all_cell_targets

    def get_data(self, batch_size, n_samples=None):
        # NOTE: Looks like this method is not used.
        raise NotImplementedError()
