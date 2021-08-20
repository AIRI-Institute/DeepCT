import numpy as np
import torch

from src.criterion import WeightedMSELoss
from src.deepct_model_multi_ct_q import qDeepCT


class TestqDeepCT:
    def test_init(self):
        model = qDeepCT(
            sequence_length=1000,
            n_cell_types=1,
            sequence_embedding_length=32,
            cell_type_embedding_length=16,
            final_embedding_length=32,
            n_genomic_features=1,
        )
        assert True
