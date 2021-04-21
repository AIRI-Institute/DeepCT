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
    sequence_length : int
        Length of one-hot sequence to pass to the model

    """

    def __init__(
        self,
        filepath,
        reference_sequence,
        n_cell_types,
        sequence_length,
    ):
        """
        Constructs a new `DNaseFileSampler` object.
        """
        super().__init__()

        self.reference_sequence = reference_sequence

        self._filepath = filepath
        self._file_handle = open(self._filepath, "r")
        self._n_cell_types = n_cell_types
        self._sequence_length = sequence_length

    def sample(self, batch_size=1, balance=True):
        """
        Draws a mini-batch of examples and their corresponding labels.
        
        Parameters
        ----------
        balance : bool 
            Whether to undersample 0.0 expression datapoints.
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
            intervals.append(line.split('\t'))

        sequences = []
        targets = []
        
        target_cells = np.zeros((batch_size, self._n_cell_types), dtype=np.int64)
        
        for index, (chrom, start, end, strand, seq_targets) in enumerate(intervals):
            start = int(start)
            end = int(end)
            
            if strand == '+': # promoter ends at start position - center around start
                start_seq = start - self._sequence_length//2
                end_seq = start + self._sequence_length//2
            else:
                start_seq = end - self._sequence_length//2
                end_seq = end + self._sequence_length//2
        
            sequence = self.reference_sequence.get_encoding_from_coords(
                chrom, start_seq, end_seq, strand=strand, pad=True # padding due to long sequence length around promoter
            )

            for i, t in enumerate(seq_targets.split(';')):
                t = float(t)
                if balance and t == -9.21: # log 0.0 expression
                    if np.random.uniform() < 0.05:
                        targets.append(t)
                        target_cells[index, i] = 1
                else:
                    targets.append(t)
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

            samples_batch, target_cells = self.sample(batch_size=to_sample, balance=False) # no balancing for test/validation
            batches.append(samples_batch)
            all_targets.append(samples_batch.targets())
            all_cell_targets.append(target_cells)

            count += to_sample
        
        all_targets = np.vstack(all_targets)
        
        return batches, all_targets, all_cell_targets

    def get_data(self, batch_size, n_samples=None):
        # NOTE: Looks like this method is not used.
        raise NotImplementedError()
