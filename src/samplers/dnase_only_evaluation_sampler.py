"""
Based on selene's IntervalsSampler (see https://github.com/FunctionLab/selene/blob/0.4.8/samplers/intervals_sampler.py)
"""
import numpy as np
from selene_sdk.samplers.intervals_sampler import IntervalsSampler
from selene_sdk.samplers.samples_batch import SamplesBatch


class DNaseOnlyEvaluationSampler(IntervalsSampler):
    """
    This sampler is used to compare DeepCT and DeepSEA metrics.

    Draws 1 sample with DeeperDeepSEA strategy and converts it to DeepCT input/target
    format.
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
        Constructs a new `DNaseOnlyEvaluationSampler` object.
        """
        super().__init__(
            reference_sequence,
            target_path,
            features,
            intervals_path,
            sample_negative,
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

        # TODO(arlapin): Need to modify this once we start predicting more than one
        # feature per cell type.
        self._n_cell_types = len(features)

    def sample(self, batch_size=1):
        n_cell_types = self._n_cell_types

        assert (
            batch_size % n_cell_types == 0
        ), "Batch size should be a multiple of n_cell_types"

        one_deeper_deep_sea_sample = super().sample(batch_size // n_cell_types)
        sequence_batch = one_deeper_deep_sea_sample.inputs()

        # Repeat each sequence n_cell_types times; its shape becomes
        # (batch_size, sequence_length, 4).
        sequences = np.repeat(sequence_batch, n_cell_types, axis=0)
        # Repeat identity matrix (batch_size // n_cell_types) times; its shape becomes
        # (batch_size, n_cell_types).
        cell_type_one_hots = np.tile(
            np.eye(n_cell_types), (batch_size // n_cell_types, 1)
        )
        # Target's shape is (batch_size // n_cell_types, n_cell_types).
        # NOTE: This should be taken into account on evaluation step.
        targets = one_deeper_deep_sea_sample.targets()

        return SamplesBatch(
            sequences,
            other_input_batches={"cell_type_batch": cell_type_one_hots},
            target_batch=targets,
        )
