import os

from src.dataset import EncodeDataset

TEST_DATA_PATH = "data/test_data/"


class TestEncodeDataset:
    def setup_method(self, test_method):
        # read test intervals
        sampling_intervals_path = os.path.join(TEST_DATA_PATH, "target_intervals.bed")
        intervals = []
        with open(sampling_intervals_path) as f:
            for line in f:
                chrom, start, end = line.split()
                intervals.append((chrom, int(start), int(end)))

        # read distinct features
        distinct_features_path = os.path.join(TEST_DATA_PATH, "distinct_features.txt")
        with open(distinct_features_path) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        # read target features
        target_features_path = os.path.join(TEST_DATA_PATH, "target_features.txt")
        with open(target_features_path) as f:
            target_features = list(map(lambda x: x.rstrip(), f.readlines()))

        reference_sequence_path = os.path.join(TEST_DATA_PATH, "mini_male.hg19.fasta")
        target_path = os.path.join(TEST_DATA_PATH, "mini_sorted_data.bed.gz")

        # initialize the cell-wise dataset
        self.cellwise_dataset = EncodeDataset(
            reference_sequence_path=reference_sequence_path,
            target_path=target_path,
            distinct_features=distinct_features,
            target_features=target_features,
            intervals=intervals,
            cell_wise=True,
            transform=None,
            sequence_length=100,
            center_bin_to_predict=20,
            feature_thresholds=0.5,
            strand="+",
        )

        # initialize the dataset without cell types
        self.cellfree_dataset = EncodeDataset(
            reference_sequence_path=reference_sequence_path,
            target_path=target_path,
            distinct_features=distinct_features,
            target_features=target_features,
            intervals=intervals,
            cell_wise=False,
            transform=None,
            sequence_length=100,
            center_bin_to_predict=20,
            feature_thresholds=0.5,
            strand="+",
        )

    def test_sample__cellfree_shape(self):
        sample = self.cellfree_dataset[100]
        assert len(sample[1]) == len(self.cellfree_dataset.distinct_features)

    def test_sample__cellwise_positive(self):
        sample = self.cellwise_dataset[
            100 * len(self.cellfree_dataset.distinct_features) + 1
        ]
        assert sample[2] == 1

    def test_sample__cellwise_negative(self):
        sample = self.cellwise_dataset[
            100 * len(self.cellfree_dataset.distinct_features)
        ]
        assert sample[2] == 0
