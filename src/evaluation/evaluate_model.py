"""
This module provides the EvaluateModel class.
"""
import logging
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from selene_sdk.sequences import Genome
from selene_sdk.utils import (
    PerformanceMetrics,
    initialize_logger,
    load_model_from_state_dict,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import trange

logger = logging.getLogger("selene")


class EvaluateModel(object):
    """
    Evaluate model on a test set of sequences with known targets.

    Mostly copied from
        https://github.com/FunctionLab/selene/blob/master/selene_sdk/evaluate_model.py

    Parameters
    ----------
    model : torch.nn.Module
        The model architecture.
    criterion : torch.nn._Loss
        The loss function that was optimized during training.
    data_sampler : selene_sdk.samplers.Sampler
        Used to retrieve samples from the test set for evaluation.
    features : list(str)
        List of distinct features the model predicts.
    trained_model_path : str
        Path to the trained model file, saved using `torch.save`.
    batch_size : int, optional
        Default is 64. Specify the batch size to process examples.
    n_test_samples : int or None, optional
        Default is `None`. Use `n_test_samples` if you want to limit the
        number of samples on which you evaluate your model. If you are
        using a sampler of type `selene_sdk.samplers.OnlineSampler`,
        by default it will draw 640000 samples if `n_test_samples` is `None`.
    report_gt_feature_n_positives : int, optional
        Default is 10. In the final test set, each class/feature must have
        more than `report_gt_feature_n_positives` positive samples in order to
        be considered in the test performance computation. The output file that
        states each class' performance will report 'NA' for classes that do
        not have enough positive samples.
    use_cuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.

    Attributes
    ----------
    model : torch.nn.Module
        The trained model.
    criterion : torch.nn._Loss
        The model was trained using this loss function.
    sampler : selene_sdk.samplers.Sampler
        The example generator.
    features : list(str)
        List of distinct features the model predicts.
    use_cuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.

    """

    def __init__(
        self,
        model,
        criterion,
        data_sampler,
        features,
        trained_model_path,
        batch_size=64,
        n_test_samples=None,
        report_gt_feature_n_positives=10,
        use_cuda=False,
        data_parallel=False,
        metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score),
    ):
        self.n_test_samples = n_test_samples
        self.batch_size = batch_size
        self.sampler = data_sampler
        self.features = features
        self.use_cuda = use_cuda

        self.output_dir = os.path.join(
            os.path.dirname(trained_model_path), "evaluation/"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)), verbosity=2
        )

        logger.info("Evaluation results will be saved at\n{}".format(self.output_dir))

        self.criterion = criterion

        trained_model = torch.load(
            trained_model_path, map_location=lambda storage, location: storage
        )
        if "state_dict" in trained_model:
            self.model = load_model_from_state_dict(trained_model["state_dict"], model)
        else:
            self.model = load_model_from_state_dict(trained_model, model)
        self.model.model.log_cell_type_embeddings_to_tensorboard(
            self.features, self.output_dir
        )

        self.model.eval()
        if data_parallel:
            self.model = nn.DataParallel(self.model)
            logger.debug("Wrapped model in DataParallel")
        if self.use_cuda:
            self.model.cuda()

        self._metrics = PerformanceMetrics(
            self._get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics,
        )

    def evaluate(self):
        """
        Passes all samples retrieved from the sampler to the model in batches and
        returns the predictions. Also reports the model's performance on these examples.

        Returns
        -------
        dict
            A dictionary, where keys are the features and the values are each a dict of
            the performance metrics (currently ROC AUC and AUPR) reported for each
            feature the model predicts.

        """
        assert self.n_test_samples % self.batch_size == 0

        batch_losses = []
        all_predictions = []
        all_test_targets = []
        for _ in trange(
            self.n_test_samples // self.batch_size, desc="Evaluating batch..."
        ):
            samples_batch = self.sampler.sample(self.batch_size)
            all_test_targets.append(samples_batch.targets())

            inputs, targets = samples_batch.torch_inputs_and_targets(self.use_cuda)

            with torch.no_grad():
                predictions = self.model.forward(inputs)
                loss = self.criterion(predictions.reshape(targets.shape), targets)
                all_predictions.append(
                    predictions.data.cpu().numpy().reshape(targets.shape)
                )

                batch_losses.append(loss.item())

        all_predictions = np.vstack(all_predictions)
        all_test_targets = np.vstack(all_test_targets)

        average_scores = self._metrics.update(all_predictions, all_test_targets)

        self._metrics.visualize(all_predictions, all_test_targets, self.output_dir)

        loss = np.average(batch_losses)
        logger.info("test loss: {0}".format(loss))
        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(self.output_dir, "test_performance.txt")
        feature_scores_dict = self._metrics.write_feature_scores_to_file(
            test_performance
        )

        return feature_scores_dict

    def _get_feature_from_index(self, index):
        """
        Gets the feature at an index in the features list.

        Parameters
        ----------
        index : int

        Returns
        -------
        str
            The name of the feature/target at the specified index.

        """
        return self.features[index]
