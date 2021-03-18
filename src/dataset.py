# TODO: add sampler for dataloaders with shuffle=True
import bisect

import numpy as np
import torch
from selene_sdk.sequences import Genome
from selene_sdk.targets import GenomicFeatures


class EncodeDataset(torch.utils.data.Dataset):
    """
    Dataset of `(sequence, cell_type, feature_values, feature_mask)`

    Parameters
    ----------
    reference_sequence_path : str
        Path to reference sequence `fasta` file from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    distinct_features : list(str)
        List of distinct feature|cell_type pairs available.
    target_features: list(str)
        List of names of features we aim to predict.
    intervals : str
        Intervals to sample from in the format `(chrom, start, end)`.
    transforms: callable, optional
        A callback function that takes `sequence, cell_type,
        feature_values, feature_mask` as arguments and returns
        their transformed version.
    sequence_length : int, optional
        Default is 1000. Dataset contains sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed targets file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0] or None, optional
         Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features.
    intervals : list(int)
        A list of intervals that we can draw samples from.
    intervals_length_sums : list(int)
        A list of the cumulative sums of lengths of the intervals
        that we can draw samples from. Used to convert dataset
        index into a position in a specific interval.
    sequence_length : int
        The length of the sequences to  train the model on.
    center_bin_to_predict: int
        Length of the center sequence piece in which to detect
        a feature annotation.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    strand : str
        Strand to sample from.
    n_cell_types: int
        Total number of cell types present in the dataset.
    """

    def __init__(
        self,
        reference_sequence_path,
        target_path,
        distinct_features,
        target_features,
        intervals,
        transforms=None,
        sequence_length=1000,
        center_bin_to_predict=200,
        feature_thresholds=0.5,
        strand="+",
    ):
        self.reference_sequence_path = reference_sequence_path
        self.reference_sequence = self._construct_ref_genome()
        self.target = GenomicFeatures(
            target_path, distinct_features, feature_thresholds=feature_thresholds
        )
        self.transform = transforms

        self.sequence_length = sequence_length
        self.center_bin_to_predict = center_bin_to_predict
        bin_radius = int(self.center_bin_to_predict / 2)
        self._start_radius = bin_radius
        if self.center_bin_to_predict % 2 == 0:
            self._end_radius = bin_radius
        else:
            self._end_radius = bin_radius + 1
        self.strand = strand
        self._surrounding_sequence_radius = int(
            (self.sequence_length - self.center_bin_to_predict) / 2
        )

        self._cell_types = []
        cell_type_indices_by_feature_index = [[] for i in range(len(target_features))]
        for feature_index, feature in enumerate(distinct_features):
            feature_description = feature.split("|")
            feature_name = feature_description[1]
            if feature_name not in target_features:
                continue
            cell_type = feature_description[0]
            addon = feature_description[2]
            if addon != "None":
                cell_type = cell_type + "_" + addon
            if cell_type not in self._cell_types:
                self._cell_types.append(cell_type)
                for feature_cell_type_indices in cell_type_indices_by_feature_index:
                    feature_cell_type_indices.append(-1)
                cell_type_idx = len(self._cell_types) - 1
            else:
                cell_type_idx = self._cell_types.index(cell_type)

            feature_idx = target_features.index(feature_name)
            cell_type_indices_by_feature_index[feature_idx][
                cell_type_idx
            ] = feature_index
        self._feature_indices_by_cell_type_index = np.array(
            cell_type_indices_by_feature_index
        ).transpose()
        self.n_cell_types = len(self._cell_types)

        self.intervals = intervals
        self.intervals_length_sums = [0]
        for chrom, pos_start, pos_end in self.intervals:
            interval_length = pos_end - pos_start
            self.intervals_length_sums.append(
                self.intervals_length_sums[-1] + interval_length
            )

    def __len__(self):
        return self.n_cell_types * self.intervals_length_sums[-1]

    def __getitem__(self, idx):
        chrom, pos, cell_type_idx = self._get_chrom_pos_cell_by_idx(idx)
        retrieved_sample = self._retrieve(chrom, pos, cell_type_idx)
        if self.transform:
            retrieved_sample = self.transform(*retrieved_sample)
        return retrieved_sample

    def _get_chrom_pos_cell_by_idx(self, idx):
        """
        Translates dataset index into genomic coordinates and
        cell type `(chrom, pos, cell_type_idx)`

        Parameters
        ----------
        idx : int
            Index of item in the dataset

        Returns
        -------
        chrom, pos, cell type:\
        tuple(str, int, int)
            Chromosome identifier, position in the chromosome, cell type
        """
        cell_type_idx = idx % self.n_cell_types
        position_idx = idx // self.n_cell_types
        interval_idx = bisect.bisect(self.intervals_length_sums, position_idx) - 1
        interval_pos = position_idx - self.intervals_length_sums[interval_idx]
        chrom, pos_start, _ = self.intervals[interval_idx]
        return chrom, pos_start + interval_pos, cell_type_idx

    def _retrieve(self, chrom, position, cell_type_idx):
        """
        Retrieves a sample for a position for a given cell type
        from `reference_sequence`.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. "chrX", "YFP")
        position : int
            The position in the query region that we will search around
            for samples.
        cell_type_idx : int
            Cell type index

        Returns
        -------
        retrieved_seq, cell_type, target, target_mask :\
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
            Retrieved encoded sequence, one-hot encoded cell type,
            target features values and target feature mask.
            Target features values (`target`) is a vector of values of target
            features for a given position and cell type. If feature `i` doesn't
            exist for a given cell type, `target[i]` value is meaningless
            but not None. Target feature mask is a binary vector
            corresponding to whether or not a specific feature exists
            in the given dataset for a given cell type.

        """

        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        targets = self.target.get_feature_data(chrom, bin_start, bin_end)

        target_idx = self._feature_indices_by_cell_type_index[cell_type_idx]
        target = targets[target_idx]
        target_mask = target_idx != -1

        cell_type = np.zeros(self.n_cell_types)
        cell_type[cell_type_idx] = 1

        window_start = bin_start - self._surrounding_sequence_radius
        window_end = bin_end + self._surrounding_sequence_radius

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, window_start, window_end, self.strand
        )

        if not self._check_retrieved_sequence(retrieved_seq, chrom, position):
            print(chrom, window_start, window_end)
            return None

        return retrieved_seq, cell_type, target, target_mask

    def _check_retrieved_sequence(self, sequence, chrom, position) -> bool:
        """Checks whether retrieved sequence is acceptable.

        Parameters
        ----------
            sequence : numpy.ndarray
                An array of shape [sequence_length, alphabet_size], defines a sequence.

        """
        if sequence.shape[0] == 0:
            # logger.info(
            print(
                'Full sequence centered at region "{0}" position '
                "{1} could not be retrieved. Sampling again.".format(chrom, position)
            )
            return False
        elif np.sum(sequence) / float(sequence.shape[0]) < 0.60:
            # logger.info(
            print(
                "Over 30% of the bases in the sequence centered "
                "at region \"{0}\" position {1} are ambiguous ('N'). "
                "Sampling again.".format(chrom, position)
            )
            return False
        elif sequence.shape[0] != self.sequence_length:
            return False
        return True

    def _construct_ref_genome(self):
        return Genome(self.reference_sequence_path, blacklist_regions="hg19")


def encode_worker_init_fn(worker_id):
    """Initialization function for multi-processing DataLoader worker"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # reconstruct reference genome object
    # because it reads from fasta `pyfaidx`
    # which is not multiprocessing-safe, see:
    # https://github.com/mdshw5/pyfaidx/issues/167#issuecomment-667591513
    dataset.reference_sequence = dataset._construct_ref_genome()
