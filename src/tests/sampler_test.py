import torch

from src.dataset import LargeRandomSampler


class TestLargeRandomSampler:
    test_data_path = "data/test_data/"

    def test_initilization(self):
        dataset = range(1000)
        generator = torch.Generator()
        generator.manual_seed(14)
        sampler = LargeRandomSampler(
            dataset, replacement=False, generator=generator, chunk_size=100
        )
