import os

from src.dataset import EncodeDataset


class TestEncodeDataset:
    test_data_path = "data/test_data/"

    def test_initilization(self):
        # read test intervals
        sampling_intervals_path = os.path.join(
            self.test_data_path, "target_intervals.bed"
        )
        intervals = []
        with open(sampling_intervals_path) as f:
            for line in f:
                chrom, start, end = line.split()
                intervals.append((chrom, int(start), int(end)))
        # read distinct features
        distinct_features_path = os.path.join(
            self.test_data_path, "distinct_features.txt"
        )
        with open(distinct_features_path) as f:
            distinct_features = list(map(lambda x: x.rstrip(), f.readlines()))

        # initialize the dataset
        TestEncodeDataset.dataset = EncodeDataset(
            reference_sequence_path=os.path.join(
                self.test_data_path, "mini_male.hg19.fasta"
            ),
            target_path=os.path.join(self.test_data_path, "mini_sorted_data.bed.gz"),
            distinct_features=distinct_features,
            target_features=["DNase"],
            intervals=intervals,
            transforms=None,
            sequence_length=100,
            center_bin_to_predict=20,
            feature_thresholds=0.5,
            strand="+",
        )
