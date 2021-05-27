"""
This module provides the EvaluateModel class.
"""
import logging
import math
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import MAX_TOTAL_VAL_TARGET_SIZE, expand_dims

logger = logging.getLogger("selene")


class EvaluateModel(object):
    """
    Evaluate model on a test set of sequences with known targets.

    Parameters
    ----------
    model : torch.nn.Module
        The model architecture.
    criterion : torch.nn._Loss
        The loss function that was optimized during training.
    data_loader : torch.utils.data.DataLoader
        Data loader that fetches test batches.
    trained_model_path : str
        Path to the trained model file, saved using `torch.save`.
    report_gt_feature_n_positives : int, optional
        Default is 10. In the final test set, each class/feature must have
        more than `report_gt_feature_n_positives` positive samples in order to
        be considered in the test performance computation. The output file that
        states each class' performance will report 'NA' for classes that do
        not have enough positive samples.
    device : str, optional
        Default is `cpu`. Specify a CUDA-device, e.g. 'cuda:2' for on-GPU training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    log_cell_type_embeddings_to_tensorboard : bool, optional
        Default is `True`. Whether to publish cell type embeddings to tensorboard.
        NOTE: If True, a model should have `get_cell_type_embeddings` method.
    metrics : dict(metric_name: metric_fn)
        Default is `dict(roc_auc=roc_auc_score, average_precision=average_precision_score)`.
        Metric functions to log.


    Attributes
    ----------
    model : torch.nn.Module
        The trained model.
    criterion : torch.nn._Loss
        The model was trained using this loss function.
    data_loader : torch.utils.data.DataLoader
        Test data loader.
    target_features : list(str)
        List of features the model predicts.
    device : torch.device
        Device on which the computation is carried out.

    """

    def __init__(
        self,
        model,
        criterion,
        data_loader,
        trained_model_path,
        report_gt_feature_n_positives=10,
        device="cpu",
        data_parallel=False,
        log_cell_type_embeddings_to_tensorboard=True,
        metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score),
    ):
        self.data_loader = data_loader

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

        if log_cell_type_embeddings_to_tensorboard:
            self.log_cell_type_embeddings_to_tensorboard()

        self.model.eval()

        self.device = torch.device(device)
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")
        else:
            self.model.to(self.device)
            self.criterion.to(self.device)
            logger.debug(f"Set modules to use device {device}")

        self.target_features = self.data_loader.dataset.target_features
        self._metrics = PerformanceMetrics(
            self._get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics,
        )

        self.masked_targets = self.data_loader.dataset.cell_wise

        self.val_reduction_factor = 1
        val_batch_size = self.data_loader.batch_size
        val_target_size = self.data_loader.dataset.target_size
        total_val_target_size = len(self.data_loader) * val_batch_size * val_target_size
        if total_val_target_size > MAX_TOTAL_VAL_TARGET_SIZE:
            self.val_reduction_factor = math.ceil(
                total_val_target_size / MAX_TOTAL_VAL_TARGET_SIZE
            )

    def log_cell_type_embeddings_to_tensorboard(self):
        embeddings = self.model.get_cell_type_embeddings()
        cell_type_labels = self.data_loader.dataset._cell_types

        writer = SummaryWriter(self.output_dir)
        writer.add_embedding(embeddings, cell_type_labels)
        writer.flush()
        writer.close()

    def evaluate(self):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        data_in_batches : list(SamplesBatch)
            A list of tuples of the data, where the first element is
            the example, and the second element is the label.

        Returns
        -------
        tuple(float, list(numpy.ndarray))
            Returns the average loss, and the list of all predictions.

        """
        self.model.eval()

        batch_losses = []
        all_predictions = []
        all_targets = []
        if self.masked_targets:
            all_target_masks = []
        else:
            all_target_masks = None

        for batch in tqdm(self.data_loader):
            if self.masked_targets:
                sequence_batch = batch[0].to(self.device)
                cell_type_batch = batch[1].to(self.device)
                targets = batch[2].to(self.device)
                target_mask = batch[3].to(self.device)
            else:
                # retrieved_seq, target
                sequence_batch = batch[0].to(self.device)
                targets = batch[1].to(self.device)

            with torch.no_grad():
                if self.masked_targets:
                    outputs = self.model(sequence_batch, cell_type_batch)
                    self.criterion.weight = target_mask
                else:
                    outputs = self.model(sequence_batch)
                loss = self.criterion(outputs, targets)
                predictions = torch.sigmoid(outputs)

                predictions = predictions.view(-1, predictions.shape[-1])
                targets = targets.view(-1, targets.shape[-1])
                if self.masked_targets:
                    target_mask = target_mask.view(-1, target_mask.shape[-1])

                if self.val_reduction_factor > 1:
                    reduced_val_batch_size = (
                        predictions.shape[0] // self.val_reduction_factor
                    )
                    reduced_index = np.random.choice(
                        predictions.shape[0], reduced_val_batch_size
                    )

                    predictions = predictions[reduced_index]
                    targets = targets[reduced_index]
                    if self.masked_targets:
                        target_mask = target_mask[reduced_index]

                all_predictions.append(predictions.data.cpu().numpy())
                all_targets.append(targets.data.cpu().numpy())
                if self.masked_targets:
                    all_target_masks.append(target_mask.data.cpu().numpy())

                batch_losses.append(loss.item())
        all_predictions = expand_dims(np.concatenate(all_predictions))
        all_targets = expand_dims(np.concatenate(all_targets))
        if self.masked_targets:
            all_target_masks = expand_dims(np.concatenate(all_target_masks))

        average_scores = self._metrics.update(
            all_predictions, all_targets, all_target_masks
        )

        self._metrics.visualize(
            all_predictions, all_targets, self.output_dir, all_target_masks
        )

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
        return self.target_features[index]
