"""
Based on selene's IntervalsSampler (see https://github.com/FunctionLab/selene/blob/0.4.8/samplers/intervals_sampler.py)

This module provides the `IntervalsSampler` class and supporting methods.
"""
import logging
import random
from collections import namedtuple
from os import replace

import numpy as np
from selene_sdk.samplers.online_sampler import OnlineSampler
from selene_sdk.samplers.samples_batch import SamplesBatch
from selene_sdk.utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)


SampleIndices = namedtuple("SampleIndices", ["indices", "weights"])
"""
A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

Parameters
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

Attributes
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

"""


# @TODO: Extend this class to work with stranded data.
class IntervalsSampler(OnlineSampler):
    """
    Draws samples from pre-specified windows in the reference sequence.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    target_path : str
        Path to tabix-indexed, compressed BED file (`*.bed.gz`) of genomic
        coordinates mapped to the genomic features we want to predict.
    features : list(str)
        List of distinct features that we aim to predict.
    intervals_path : str
        The path to the file that contains the intervals to sample from.
        In this file, each interval should occur on a separate line.
    sample_negative : bool, optional
        Default is `False`. This tells the sampler whether negative
        examples (i.e. with no positive labels) should be drawn when
        generating samples. If `True`, both negative and positive
        samples will be drawn. If `False`, only samples with at least
        one positive label will be drawn.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['X', 'Y']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0] or None, optional
         Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list of str
        Default is `["test"]`. The list of modes for which we should
        save the sampled data to file.
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    sample_from_intervals : list(tuple(str, int, int))
        A list of coordinates that specify the intervals we can draw
        samples from.
    interval_lengths : list(int)
        A list of the lengths of the intervals that we can draw samples
        from. The probability that we will draw a sample from an
        interval is a function of that interval's length and the length
        of all other intervals.
    sample_negative : bool
        Whether negative examples (i.e. with no positive label) should
        be drawn when generating samples. If `True`, both negative and
        positive samples will be drawn. If `False`, only samples with at
        least one positive label will be drawn.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These Regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.
    random_generator : numpy.random.Generator
        Random values generator.

    """

    def __init__(
        self,
        reference_sequence,
        target_path,
        features,
        intervals_path,
        sample_negative=False,
        seed=436,
        validation_holdout=["chr6", "chr7"],
        test_holdout=["chr8", "chr9"],
        sequence_length=1000,
        center_bin_to_predict=200,
        feature_thresholds=0.5,
        mode="train",
        save_datasets=["test"],
        output_dir=None,
    ):
        """
        Constructs a new `IntervalsSampler` object.
        """
        super(IntervalsSampler, self).__init__(
            reference_sequence,
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir,
        )

        self.random_generator = np.random.default_rng(self.seed)

        self._cell_type_index_by_feature_index = [None] * len(features)
        self._cell_type_by_index = {}
        self._index_by_cell_type = {}
        for feature_index, feature in enumerate(features):
            cell_type, _, addon = feature.split("|")
            if addon != "None":
                cell_type = cell_type + "_" + addon

            if cell_type not in self._index_by_cell_type:
                index = len(self._index_by_cell_type)
                self._index_by_cell_type[cell_type] = index
                self._cell_type_by_index[index] = cell_type

            self._cell_type_index_by_feature_index[
                feature_index
            ] = self._index_by_cell_type[cell_type]
        self._n_cell_types = len(self._cell_type_by_index)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []

        if self._holdout_type == "chromosome":
            self._partition_dataset_chromosome(intervals_path)
        else:
            self._partition_dataset_proportion(intervals_path)

        for mode in self.modes:
            self._update_randcache(mode=mode)

        self.sample_negative = sample_negative

    def _partition_dataset_proportion(self, intervals_path):
        """
        When holdout sets are created by randomly sampling a proportion
        of the data, this method is used to divide the data into
        train/test/validate subsets.

        Parameters
        ----------
        intervals_path : str
            The path to the file that contains the intervals to sample
            from. In this file, each interval should occur on a separate
            line.

        """
        with open(intervals_path, "r") as file_handle:
            for line in file_handle:
                cols = line.strip().split("\t")
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)
        n_intervals = len(self.sample_from_intervals)

        # all indices in the intervals list are shuffled
        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)

        # the first section of indices is used as the validation set
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate]
        )
        self._sample_from_mode["validate"] = SampleIndices(val_indices, val_weights)

        if self.test_holdout:
            # if applicable, the second section of indices is used as the
            # test set
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end],
            )
            self._sample_from_mode["test"] = SampleIndices(test_indices, test_weights)

            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:]
            )
            self._sample_from_mode["train"] = SampleIndices(tr_indices, tr_weights)
        else:
            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:]
            )
            self._sample_from_mode["train"] = SampleIndices(tr_indices, tr_weights)

    def _partition_dataset_chromosome(self, intervals_path):
        """
        When holdout sets are created by selecting all samples from a
        specified region (e.g. a chromosome) this method is used to
        divide the data into train/test/validate subsets.

        Parameters
        ----------
        intervals_path : str
            The path to the file that contains the intervals to sample
            from. In this file, each interval should occur on a separate
            line.

        """
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        with open(intervals_path, "r") as file_handle:
            for index, line in enumerate(file_handle):
                cols = line.strip().split("\t")
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                if chrom in self.validation_holdout:
                    self._sample_from_mode["validate"].indices.append(index)
                elif self.test_holdout and chrom in self.test_holdout:
                    self._sample_from_mode["test"].indices.append(index)
                else:
                    self._sample_from_mode["train"].indices.append(index)
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices
            )
            self._sample_from_mode[mode] = self._sample_from_mode[mode]._replace(
                indices=indices, weights=weights
            )

    def _check_retrieved_sequence(self, sequence, chrom, position) -> bool:
        """Checks whether retrieved sequence is acceptable.

        Parameters
        ----------
            sequence : numpy.ndarray
                An array of shape [sequence_length, alphabet_size], defines a sequence.

        """
        if sequence.shape[0] == 0:
            logger.info(
                'Full sequence centered at region "{0}" position '
                "{1} could not be retrieved. Sampling again.".format(chrom, position)
            )
            return False
        elif np.sum(sequence) / float(sequence.shape[0]) < 0.60:
            logger.info(
                "Over 30% of the bases in the sequence centered "
                "at region \"{0}\" position {1} are ambiguous ('N'). "
                "Sampling again.".format(chrom, position)
            )
            return False

        return True

    def _retrieve_positive_and_some_zero_samples(
        self, targets, max_to_draw
    ) -> np.array:
        """Retrieves max possible amount of positive targets (not more than
        `max_to_draw`) and some zero targets (not more than a number of selected
        positives).

        Returns
        -------
            numpy.array : Selected target indices without repetition.
        """
        positive_samples_to_retrieve = min(max_to_draw, len(np.nonzero(targets)[0]))
        retrieved_positive_target_indices = self.random_generator.choice(
            np.nonzero(targets)[0], positive_samples_to_retrieve, replace=False
        )

        negative_samples_to_retrieve = max(
            min(
                max_to_draw - positive_samples_to_retrieve,
                positive_samples_to_retrieve,
                len(targets) - positive_samples_to_retrieve,
            ),
            0,
        )
        retrieved_negative_target_indices = self.random_generator.choice(
            np.where(targets == 0)[0], negative_samples_to_retrieve, replace=False
        )

        return np.concatenate(
            (retrieved_positive_target_indices, retrieved_negative_target_indices)
        )

    def _retrieve(self, chrom, position, max_to_draw):
        """
        Retrieves a batch of samples around a position in the `reference_sequence`.

        Parameters
        ----------
        chrom : str
            The name of the region (e.g. "chrX", "YFP")
        position : int
            The position in the query region that we will search around
            for samples.
        max_to_draw : int
            An upper-bound for the number of retrieved samples.
            NOTE: Can retrieve less or equal, but not more.

        Returns
        -------
        inputs, targets :\
        list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray
            Retrieved inputs and outputs. Inputs are represented as a list of sequence
            and cell type pairs. Targets are represented as a single array. So the
            length of inputs list equals to the length of targets array.

            NOTE: Returns None if wasn't able to retrieve at this position.

        """
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        targets = self.target.get_feature_data(chrom, bin_start, bin_end)

        if not self.sample_negative and np.sum(targets) == 0:
            logger.info(
                "No features picked in region surrounding "
                'region "{0}" position {1}. Sampling again.'.format(chrom, position)
            )
            return None

        window_start = bin_start - self.surrounding_sequence_radius
        window_end = bin_end + self.surrounding_sequence_radius
        strand = self.STRAND_SIDES[random.randint(0, 1)]
        retrieved_seq = self.reference_sequence.get_encoding_from_coords(
            chrom, window_start, window_end, strand
        )

        if not self._check_retrieved_sequence(retrieved_seq, chrom, position):
            return None

        retrieved_target_indices = self._retrieve_positive_and_some_zero_samples(
            targets, max_to_draw
        )

        if self.mode in self._save_datasets:
            # TODO(arlapin): Might need to change something here, if this is used.
            feature_indices = ";".join(
                [str(f) for f in np.nonzero(targets[retrieved_target_indices])[0]]
            )
            self._save_datasets[self.mode].append(
                [chrom, window_start, window_end, strand, feature_indices]
            )
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)

        all_retrieved_inputs = []
        for feature_index in retrieved_target_indices:
            cell_type_index = self._cell_type_index_by_feature_index[feature_index]
            cell_type_one_hot_encoding = np.zeros(self._n_cell_types)
            cell_type_one_hot_encoding[cell_type_index] = 1.0
            all_retrieved_inputs.append((retrieved_seq, cell_type_one_hot_encoding))

        return (all_retrieved_inputs, targets[retrieved_target_indices])

    def _update_randcache(self, mode=None):
        """
        Updates the cache of indices of intervals. This allows us
        to randomly sample from our data without having to use a
        fixed-point approach or keeping all labels in memory.

        Parameters
        ----------
        mode : str or None, optional
            Default is `None`. The mode that these samples should be
            used for. See `selene_sdk.samplers.IntervalsSampler.modes` for
            more information.

        """
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = self.random_generator.choice(
            self._sample_from_mode[mode].indices,
            size=len(self._sample_from_mode[mode].indices),
            replace=True,
            p=self._sample_from_mode[mode].weights,
        )
        self._randcache[mode]["sample_next"] = 0

    def sample(self, batch_size=1):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        SamplesBatch
            A batch containing the numeric representation of the sequence examples,
            one-hot encodings for cell types, and their corresponding targets.

            The shape of `sequences` will be :math:`B \\times L \\times N`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length, and :math:`N`
            is the size of the sequence type's alphabet.

            The shape of `cell_types` will be :math:`B \\times C`, where :math:`C` is
            a number of cell types in the provided features.

            The shape of `targets` will be :math:`B \\times 1`,

        """
        sequence_batch = np.zeros((batch_size, self.sequence_length, 4))
        cell_types_batch = np.zeros((batch_size, self._n_cell_types))
        targets_batch = np.zeros((batch_size, 1))

        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[self.mode]["sample_next"]
            if sample_index == len(self._sample_from_mode[self.mode].indices):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = self._randcache[self.mode]["cache_indices"][
                sample_index
            ]
            self._randcache[self.mode]["sample_next"] += 1

            interval_info = self.sample_from_intervals[rand_interval_index]
            interval_length = self.interval_lengths[rand_interval_index]

            chrom = interval_info[0]
            position = int(interval_info[1] + random.uniform(0, 1) * interval_length)

            retrieve_output = self._retrieve(
                chrom, position, batch_size - n_samples_drawn
            )
            if not retrieve_output:
                continue
            inputs, targets = retrieve_output

            for (sequence, cell_type), target in zip(inputs, targets):
                sequence_batch[n_samples_drawn, :, :] = sequence
                cell_types_batch[n_samples_drawn, :] = cell_type
                targets_batch[n_samples_drawn, :] = target
                n_samples_drawn += 1

        return SamplesBatch(
            sequence_batch,
            other_input_batches={"cell_type_batch": cell_types_batch},
            target_batch=targets_batch,
        )