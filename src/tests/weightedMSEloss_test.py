import os

import numpy as np
import pyBigWig
import torch

from src.criterion import WeightedMSELoss, WeightedMSELossWithMPI
from src.dataset import EncodeDataset, encode_worker_init_fn

TEST_DATA_PATH = "data/test_data/"


class TestWeightedMSELoss:
    def test_loss(self):
        np.random.seed(10)
        sample_size = 200
        sample_x = np.random.normal(0, 1, sample_size)
        sample_y = np.random.normal(0, 1, sample_size)
        weights = np.random.randint(0, 2, sample_size)
        l = WeightedMSELoss(weights)
        loss_manual = np.mean(
            [
                w * ((x - y) ** 2)
                for x, y, w in zip(sample_x, sample_y, weights)
                if w != 0
            ]
        )
        with torch.no_grad():
            loss = l.forward(
                torch.from_numpy(sample_x),
                torch.from_numpy(sample_y),
            )
        assert loss - loss_manual < 0.00000001


class TestWeightedMSELossWithMPI:
    def setup_method(self, test_method):

        # create a dataset first
        reference_sequence_path = os.path.join(TEST_DATA_PATH, "mini_male.hg19.fasta")
        self.n_cell_types = 4
        self.n_features = 2
        ct_by_features = np.ones(self.n_cell_types * self.n_features).reshape(
            self.n_cell_types, self.n_features
        )

        # say some features are not present
        ct_by_features[0, 0] = 0
        ct_by_features[1, 1] = 0
        ct_by_features[3, 1] = 0
        self.ct_by_features = ct_by_features

        # create a sample bigWig files
        intervals = [("chr1", 4000, 5000), ("chr2", 5000, 6000)]
        distinct_features = []
        target_features = list(map(str, range(self.n_features)))
        features2bigWig_mapping = []
        for cell_type in range(self.n_cell_types):
            for feature in range(self.n_features):
                if not self.ct_by_features[cell_type, feature]:
                    continue
                fname = os.path.join(
                    TEST_DATA_PATH,
                    "testqDataset" + str(cell_type) + "_" + str(feature) + ".bw",
                )
                distinct_features_name = "|".join(
                    map(str, [cell_type, feature, "None"])
                )
                distinct_features.append(distinct_features_name)
                features2bigWig_mapping.append(distinct_features_name + "\t" + fname)
                if os.path.isfile(fname):
                    # continue
                    pass
                bw = pyBigWig.open(fname, "w")
                bw.addHeader(([("chr1", 1000000), ("chr2", 1500000)]))
                bw.addEntries(
                    ["chr1", "chr2"],
                    [0, 0],
                    ends=[20000, 20000],
                    values=[
                        float((feature + 1) * 100 + cell_type - 2),
                        float((feature + 1) * 200 + (cell_type - 2) * 3),
                    ],
                )
                bw.close()

        # create bigwig-feature mapping
        with open(os.path.join(TEST_DATA_PATH, "testqDatasetMapping.tsv"), "w") as f:
            f.write("\n".join(features2bigWig_mapping))

        target_path = os.path.join(TEST_DATA_PATH, "testqDatasetMapping.tsv")

        # initialize the quantitative cell-wise dataset
        self.cellwise_qdataset = EncodeDataset(
            reference_sequence_path=reference_sequence_path,
            target_path=target_path,
            distinct_features=distinct_features,
            target_features=target_features,
            intervals=intervals,
            cell_wise=True,
            multi_ct_target=True,
            transform=None,
            quantitative_features=True,
            sequence_length=100,
            center_bin_to_predict=20,
            feature_thresholds=0.5,
            strand="+",
        )

        self.batch_size = 5
        self.DataLoader = torch.utils.data.DataLoader(
            self.cellwise_qdataset,
            batch_size=self.batch_size,
            num_workers=1,
            worker_init_fn=encode_worker_init_fn,
        )

    def test_loss(self):
        def manual_loss(alpha, input, target):
            losses_mean = []
            losses_dev = []
            for batch_item in range(self.batch_size):
                # compute means
                means = [[] for i in range(self.n_features)]
                for feature in range(self.n_features):
                    for cell_type in range(self.n_cell_types):
                        if self.ct_by_features[cell_type, feature] != 0:
                            means[feature].append(
                                target[batch_item][cell_type, feature]
                            )
                means = [np.mean(i) for i in means]
                deviations = np.array(target[batch_item])
                for feature in range(self.n_features):
                    for cell_type in range(self.n_cell_types):
                        deviations[cell_type, feature] = (
                            deviations[cell_type, feature] - means[feature] 
                        )
                predicted_mean, predicted_deviation = (
                    input[batch_item, -1, :],
                    input[batch_item, :-1, :],
                )
                loss_mean = np.mean(
                    [(means[i] - predicted_mean[i]) ** 2 for i in range(len(means))]
                )
                loss_dev = np.mean(
                    [
                        (deviations[i, j] - predicted_deviation[i, j]) ** 2
                        for i in range(self.n_cell_types)
                        for j in range(self.n_features)
                        if self.ct_by_features[i, j] != 0
                    ]
                )
                losses_mean.append(loss_mean)
                losses_dev.append(loss_dev)
            return alpha * np.mean(losses_mean) + (1 - alpha) * np.mean(losses_dev)

        CLOSE = 0.00000001
        batch = next(iter(self.DataLoader))
        sequence_batch = batch[0]
        cell_type_batch = batch[1]
        targets = batch[2]
        target_mask = batch[3]

        assert torch.sum(target_mask) == np.sum(self.ct_by_features) * self.batch_size

        predicts = np.random.normal(
            100, 10, size=(self.batch_size, self.n_cell_types + 1, self.n_features)
        )

        for alpha in [1, 0.5, 0]:
            loss = WeightedMSELossWithMPI(alpha=alpha)
            loss.weight = target_mask
            assert (
                abs(
                    manual_loss(alpha, predicts, targets)
                    - loss.forward(torch.tensor(predicts), targets)
                )
                < CLOSE
            )

        # check that alpha value works correctly
        loss = WeightedMSELossWithMPI(alpha=0)
        loss.weight = target_mask
        l1 = loss.forward(predicts, targets)
        predicts2 = np.array(predicts)
        predicts2[0][-1, :] += 10
        l2 = loss.forward(predicts2, targets)
        assert abs(l1 - l2) < CLOSE

        loss = WeightedMSELossWithMPI(alpha=1)
        loss.weight = target_mask
        l1 = loss.forward(predicts, targets)
        predicts2 = np.array(predicts)
        predicts2[0][0, :] += 10
        l2 = loss.forward(predicts2, targets)
        assert abs(l1 - l2) < CLOSE

        loss = WeightedMSELossWithMPI(alpha=0.5)
        loss.weight = target_mask
        l1 = loss.forward(predicts, targets)
        predicts2 = np.array(predicts)
        predicts2[0][0, :] += 10
        l2 = loss.forward(predicts2, targets)
        assert abs(l1 - l2) > CLOSE

        # test target_mask effect
        loss = WeightedMSELossWithMPI(alpha=0.5)
        loss.weight = target_mask
        l1 = loss.forward(predicts, targets)
        predicts2 = np.array(predicts)
        predicts2[0][0, 0] += 10
        l2 = loss.forward(predicts2, targets)
        assert abs(l1 - l2) < CLOSE

        loss = WeightedMSELossWithMPI(alpha=0.5)
        loss.weight = target_mask
        l1 = loss.forward(predicts, targets)
        predicts2 = np.array(predicts)
        predicts2[0][0, 1] += 10
        l2 = loss.forward(predicts2, targets)
        assert abs(l1 - l2) > CLOSE
