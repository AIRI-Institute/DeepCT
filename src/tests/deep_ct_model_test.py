from src.deep_ct_model import DeepCT


class TestDeepCT:
    def test_initialization(self):
        cmd = DeepCT(
            sequence_length=1000,
            n_cell_types=500,
            sequence_embedding_length=128,
            cell_type_embedding_length=20,
            final_embedding_length=128,
            n_genomic_features=1,
        )
