import bisect

import numpy as np
import torch
from selene_sdk.sequences import Genome
from selene_sdk.targets import GenomicFeatures

from src.transforms import PermuteSequenceChannels

_FEATURE_NOT_PRESENT = -1


class EncodeDataset(torch.utils.data.Dataset):
    """
    Dataset of ENCODE epigenetic features of either
    `(sequence, cell_type, feature_values, feature_mask)`
    or `(sequence, feature_cell_values)`

    Parameters
    ----------
    reference_sequence_path : str
        Path to reference sequence `fasta` file from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    distinct_features : list(str)
        List of distinct `cell_type|feature_name|info` combinations available,
        e.g. `["K562|ZBTB33|None", "HCF|DNase|None", "HUVEC|DNase|None"]`.
    target_features : list(str)
        List of names of features we aim to predict, e.g. ["CTCF", "DNase"].
    intervals : list(tuple)
        Intervals to sample from in the format `(chrom, start, end)`,
        e.g. [("chr1", 550, 590), ("chr2", 6100, 6315)].
    cell_wise : bool
        Whether the dataset is supposed to return samples cell-wise,
        i.e. whether samples are `(sequence, cell_type, feature_values, feature_mask)`
        or `(sequence, feature_cell_values)`
    transform : callable, optional
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
    strand : str
        Default is '+'. Strand to sample from.
    multi_ct_target : bool, optional
        Default is False. Make samples positional, like with cell_wise=False but
        fetch targets as if cell_wise=True and for all cell types at once, i.e.
        a sample would look like `(sequence, 0.0, target, target_mask)`,
        where `target` and `target_mask` have shape `(n_cell_types, n_target_features)`.
    position_skip : int, optional
        Default is 1. Use sequences centered at points that are `position_skip`
        positions apart to avoid samples with big sequence overlaps.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features.
    target_features : list(str)
        List of names of features we aim to predict, e.g. ["CTCF", "DNase"].
    cell_wise : bool
        Whether each sample is cell type specific or not
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
    multi_ct_target : bool
        Whether data is meant for multiple cell type model input or not.
    n_cell_types : int
        Total number of cell types present in the dataset.
    position_skip : int
        Number of sequence positions to skip between samples.
    """

    def __init__(
        self,
        reference_sequence_path,
        target_path,
        distinct_features,
        target_features,
        intervals,
        cell_wise=True,
        transform=PermuteSequenceChannels(),
        sequence_length=1000,
        center_bin_to_predict=200,
        feature_thresholds=0.5,
        strand="+",
        multi_ct_target=False,
        position_skip=1,
    ):
        self.reference_sequence_path = reference_sequence_path
        self.reference_sequence = self._construct_ref_genome()

        self.distinct_features = distinct_features
        self.target_features = target_features
        self.target_path = target_path
        self.feature_thresholds = feature_thresholds
        self.target = self._construct_target()

        if not cell_wise and multi_ct_target:
            raise ValueError("cell_wise=True must be used with multi_ct_target=True")
        self.cell_wise = cell_wise
        if self.cell_wise:
            self.n_target_features = len(self.target_features)
        else:
            self.n_target_features = len(self.distinct_features)
        self.multi_ct_target = multi_ct_target
        self.transform = transform

        self.sequence_length = sequence_length
        self.center_bin_to_predict = center_bin_to_predict
        bin_radius = int(self.center_bin_to_predict / 2)
        self._start_radius = bin_radius
        self._end_radius = bin_radius + self.center_bin_to_predict % 2

        self.strand = strand
        self._surrounding_sequence_radius = (
            self.sequence_length - self.center_bin_to_predict
        ) // 2

        if self.cell_wise:
            self._cell_types = []
            cell_type_indices_by_feature_index = [
                [] for i in range(len(self.target_features))
            ]
            for distinct_feature_index, distinct_feature in enumerate(
                self.distinct_features
            ):
                feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
                if feature_name not in self.target_features:
                    continue
                if cell_type not in self._cell_types:
                    self._cell_types.append(cell_type)

            self.n_cell_types = len(self._cell_types)
            self._feature_indices_by_cell_type_index = np.full(
                (self.n_cell_types, self.n_target_features), _FEATURE_NOT_PRESENT
            )

            for distinct_feature_index, distinct_feature in enumerate(
                self.distinct_features
            ):
                feature_name, cell_type = self._parse_distinct_feature(distinct_feature)
                if feature_name not in self.target_features:
                    continue
                feature_index = self.target_features.index(feature_name)
                cell_type_index = self._cell_types.index(cell_type)
                self._feature_indices_by_cell_type_index[cell_type_index][
                    feature_index
                ] = distinct_feature_index

            if self.multi_ct_target:
                self.target_mask = (
                    self._feature_indices_by_cell_type_index != _FEATURE_NOT_PRESENT
                )

        self.position_skip = position_skip

        self.intervals = intervals
        self.intervals_length_sums = [0]
        for chrom, pos_start, pos_end in self.intervals:
            interval_length = (pos_end - pos_start) // self.position_skip + 1
            self.intervals_length_sums.append(
                self.intervals_length_sums[-1] + interval_length
            )
        if self.cell_wise:
            if self.multi_ct_target:
                self.target_size = self.target_mask.size
            else:
                self.target_size = self.n_target_features
        else:
            self.target_size = len(self.distinct_features)

    def __len__(self):
        if not self.cell_wise or self.multi_ct_target:
            return self.intervals_length_sums[-1]
        return self.n_cell_types * self.intervals_length_sums[-1]

    def __getitem__(self, idx):
        chrom, pos, cell_type_idx = self._get_chrom_pos_cell_by_idx(idx)
        retrieved_sample = self._retrieve(chrom, pos, cell_type_idx)
        if self.transform is not None:
            retrieved_sample = self.transform(retrieved_sample)
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
        if self.cell_wise and not self.multi_ct_target:
            cell_type_idx = idx % self.n_cell_types
            position_idx = idx // self.n_cell_types
        else:
            cell_type_idx = 0
            position_idx = idx
        interval_idx = bisect.bisect(self.intervals_length_sums, position_idx) - 1
        interval_pos = (
            position_idx - self.intervals_length_sums[interval_idx]
        ) * self.position_skip + self.position_skip // 2

        # handle the edge case when interval_pos is out of interval boundaries
        interval_pos = min(
            interval_pos,
            self.intervals[interval_idx][2] - self.intervals[interval_idx][1],
        )

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
        if self.cell_wise:
            if self.multi_ct_target:
                target = []
                for cell_type_idx in range(self.n_cell_types):
                    ct_target_idx = self._feature_indices_by_cell_type_index[
                        cell_type_idx
                    ]
                    ct_target = targets[ct_target_idx].astype(np.float32)
                    target.append(ct_target)
                target = np.array(target).astype(np.float32)
                target_mask = self.target_mask
                cell_type = 0.0
            else:
                target_idx = self._feature_indices_by_cell_type_index[cell_type_idx]
                target = targets[target_idx].astype(np.float32)
                target_mask = target_idx != _FEATURE_NOT_PRESENT
                cell_type = np.zeros(self.n_cell_types, dtype=np.float32)
                cell_type[cell_type_idx] = 1
        else:
            target = targets.astype(np.float32)
            target_mask = np.ones_like(target)
            cell_type = None

        window_start = bin_start - self._surrounding_sequence_radius
        window_end = bin_end + self._surrounding_sequence_radius

        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, window_start, window_end, self.strand
        )

        if not self._check_retrieved_sequence(retrieved_seq, chrom, position):
            return None

        if self.cell_wise:
            retrieved_sample = (retrieved_seq, cell_type, target, target_mask)
        else:
            retrieved_sample = (retrieved_seq, target)
        return retrieved_sample

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
                "{1} could not be retrieved.".format(chrom, position)
            )
            return False
        elif np.sum(sequence) / float(sequence.shape[0]) < 0.60:
            # logger.info(
            print(
                "Over 30% of the bases in the sequence centered "
                "at region \"{0}\" position {1} are ambiguous ('N'). ".format(
                    chrom, position
                )
            )
            return False
        elif sequence.shape[0] != self.sequence_length:
            print(
                f"Sequence retrieved at {chrom} position {position}\
                length {sequence.shape[0]} does not match \
                specified sequence length {self.sequence_length}"
            )
            return False
        return True

    def _construct_ref_genome(self):
        return Genome(self.reference_sequence_path)

    def _construct_target(self):
        return GenomicFeatures(
            self.target_path,
            self.distinct_features,
            feature_thresholds=self.feature_thresholds,
        )

    def _parse_distinct_feature(self, distinct_feature):
        """
        Parse a combination of `cell_type|feature_name|info` into
        `(feature_name, cell_type)`
        """
        feature_description = distinct_feature.split("|")
        feature_name = feature_description[1]
        cell_type = feature_description[0]
        addon = feature_description[2]
        if addon != "None":
            cell_type = cell_type + "_" + addon
        return feature_name, cell_type


class LargeRandomSampler(torch.utils.data.RandomSampler):
    """
    Samples elements randomly by splitting the dataset into chunks and permuting
    indices within these chunks. If without replacement, then sample from a
    dataset shuffled in chunks.
    If with replacement, then user can specify `num_samples` to draw.

    Parameters
    ----------
    reference_sequence_path : str
        Path to reference sequence `fasta` file from which to create examples.
    data_source : torch.utils.data.Dataset
        Dataset to sample from.
    replacement : bool
        Samples are drawn on-demand with replacement if ``True``,
        default=``False``.
    num_samples : int
        Number of samples to draw, default=`len(dataset)`. This argument
        is supposed to be specified only when `replacement` is ``True``.
    generator : torch.Generator
        Generator used in sampling.
    chunk_size : int
        Size of chunks that dataset is divided into for shuffling.
    """

    def __init__(
        self,
        data_source,
        replacement=False,
        num_samples=None,
        generator=None,
        chunk_size=10000000,
    ):
        super().__init__(
            data_source,
            replacement=replacement,
            num_samples=num_samples,
            generator=generator,
        )

        self.chunk_size = chunk_size
        self.m_chunks = (len(self.data_source) - 1) // self.chunk_size + 1

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            self.chunks_order = self._generate_chunks_order()
            for chunk_idx in self.chunks_order:
                self.cur_chunk = chunk_idx
                chunk_offset = self.chunk_size * self.cur_chunk
                chunk_perm = self._permute_chunk(self.cur_chunk)
                for idx in chunk_perm:
                    yield chunk_offset + idx.item()

    def _generate_chunks_order(self):
        return torch.randperm(self.m_chunks, generator=self.generator).tolist()

    def _permute_chunk(self, chunk_idx):
        n = len(self.data_source)
        if chunk_idx == self.m_chunks - 1:
            chunk_size = n % self.chunk_size
        else:
            chunk_size = self.chunk_size
        return torch.randperm(chunk_size, generator=self.generator)


def encode_worker_init_fn(worker_id):
    """Initialization function for multi-processing DataLoader worker"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # reconstruct reference genome object
    # because it reads from fasta `pyfaidx`
    # which is not multiprocessing-safe, see:
    # https://github.com/mdshw5/pyfaidx/issues/167#issuecomment-667591513
    dataset.reference_sequence = dataset._construct_ref_genome()
    dataset.target = dataset._construct_target()
