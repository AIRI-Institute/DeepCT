import torch
import numpy as np

from src.deepct_model_multi_ct_q import qDeepCT, WeightedMSELoss

class TestqDeepCT:

    def test_loss(self):
        np.random.seed(10)
        sample_size = 200
        sample_x = np.random.normal(0,1,sample_size)
        sample_y = np.random.normal(0,1,sample_size)
        weights = np.random.normal(0,1,sample_size)
        l = WeightedMSELoss(weights)
        loss_manual = np.mean([w*((x-y)**2) for x,y,w in zip(sample_x,sample_y,weights)])
        with torch.no_grad():
            loss = l.forward(torch.from_numpy(sample_x),
                             torch.from_numpy(sample_y),
                             )
        assert loss - loss_manual < 0.00000001

    def test_init(self):
        model = qDeepCT(
            sequence_length=1000,
            n_cell_types=1,
            sequence_embedding_length=32,
            cell_type_embedding_length=16,
            final_embedding_length=32,
            n_genomic_features=1
        )
        assert True