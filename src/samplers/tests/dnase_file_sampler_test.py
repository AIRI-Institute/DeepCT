import os

from selene_sdk.sequences.genome import Genome

from src.samplers.dnase_file_sampler import DNaseFileSampler

TEST_REFERENCE_SEQUENCE_FILE = "data/test_data/mini_male.hg19.fasta"

# Contains 11 intervals
TEST_BED_FILE = "data/test_data/samples_for_tests.bed"


class TestDNaseFileSampler:
    def setup_method(self, test_method):
        reference_sequence = Genome(TEST_REFERENCE_SEQUENCE_FILE)
        self.sampler = DNaseFileSampler(
            TEST_BED_FILE, reference_sequence, n_cell_types=125
        )

    def test_sample__single_sample(self):
        batch = self.sampler.sample()
        assert batch.targets().shape[0] == 1

    def test_sample__sample_whole_file(self):
        batch = self.sampler.sample(batch_size=10)
        assert batch.targets().shape[0] == 10

    def test_sample__sample_more_than_file_contains(self):
        batch = self.sampler.sample(batch_size=34)
        assert batch.targets().shape[0] == 34

    def test_get_data_and_targets__single_sample(self):
        batches, all_target = self.sampler.get_data_and_targets(
            batch_size=1, n_samples=1
        )
        assert len(batches) == 1

    def test_get_data_and_targets__batch_larger_than_n_samples(self):
        batches, all_target = self.sampler.get_data_and_targets(
            batch_size=5, n_samples=1
        )
        assert len(batches) == 1

    def test_get_data_and_targets__n_samples_larger_than_batch(self):
        batches, all_target = self.sampler.get_data_and_targets(
            batch_size=2, n_samples=5
        )
        assert len(batches) == 3

    def test_get_data_sample_wo_positive_cases(self):
        batches, all_target = self.sampler.get_data_and_targets(
            batch_size=11, n_samples=11
        )
        assert sum(all_target[9]) == 0