"""
This module provides the `TrainModel` class and supporting methods.
"""
import logging
import math
import os
import shutil
from time import strftime, time
import itertools
import numpy as np
import torch
import torch.nn as nn
from selene_sdk.utils import (
    PerformanceMetrics,
    initialize_logger,
    load_model_from_state_dict,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from tqdm import tqdm

from src.utils import MAX_TOTAL_VAL_TARGET_SIZE, expand_dims

import warnings
warnings.filterwarnings("ignore") 


logger = logging.getLogger("selene")


def _metrics_logger(name, out_filepath):
    logger = logging.getLogger("{0}".format(name))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(
        os.path.join(out_filepath, "{0}.txt".format(name))
    )
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    return logger


class TrainEncodeDatasetModel(object):
    """
    This class ties together the various objects and methods needed to
    train and validate a model.

    TrainEncodeDatasetModel saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest
    validation performance is better than the previous best-performing
    model) to `output_dir`.

    TrainEncodeDatasetModel also outputs 2 files that can be used to monitor training
    as Selene runs: `selene_sdk.train_model.train.txt` (training loss) and
    `selene_sdk.train_model.validation.txt` (validation loss & average
    ROC AUC). The columns in these files can be used to quickly visualize
    training history (e.g. you can use `matplotlib`, `plt.plot(auc_list)`)
    and see, for example, whether the model is still improving, if there are
    signs of overfitting, etc.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizer_kwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    train_loader: torch.utils.data.DataLoader
        Data loader that fetches batches to train on.
    val_loader: torch.utils.data.DataLoader
        Data loader that fetches validation batches.
    n_epochs : int
        The maximum number of epochs to iterate over the dataset.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * batch_size`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    output_dir : str
        The output directory to save model checkpoints and logs in.
    scheduler_class: torch.optim.lr_scheduler, optional
        The LR scheduler class to use with specified optimizer.
    scheduler_kwargs: dict, optional
        The dictionary of keyword arguments to pass to the LR scheduler's
        constructor.
    save_checkpoint_every_n_steps : int or None, optional
        Default is 1000. If None, set to the same value as
        `report_stats_every_n_steps`
    save_new_checkpoints_after_n_steps : int or None, optional
        Default is None. The number of steps after which Selene will
        continually save new checkpoint model weights files
        (`checkpoint-<TIMESTAMP>.pth.tar`) every
        `save_checkpoint_every_n_steps`. Before this point,
        the file `checkpoint.pth.tar` is overwritten every
        `save_checkpoint_every_n_steps` to limit the memory requirements.
    log_embeddings_every_n_steps : int or None, optional
        Default is 8000. The number of steps after which the embeddings learnt
        by the model will be saved.
    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    device : str, optional
        Default is `cpu`. Specify a CUDA-device, e.g. 'cuda:2' for on-GPU training.
    data_parallel : bool, optional
        Default is `False`. Specify whether multiple GPUs are available
        for torch to use during training.
    logging_verbosity : {0, 1, 2}, optional
        Default is 2. Set the logging verbosity level.

            * 0 - Only warnings will be logged.
            * 1 - Information and warnings will be logged.
            * 2 - Debug messages, information, and warnings will all be\
                  logged.

    checkpoint_resume : str or None, optional
        Default is `None`. If `checkpoint_resume` is not None, it should be the
        path to a model file generated by `torch.save` that can now be read
        using `torch.load`.
    metrics : dict(metric_name: metric_fn)
        Default is `dict(roc_auc=roc_auc_score, average_precision=average_precision_score)`. 
        Metric functions to log.
    log_confusion_matrix : bool, optional
        Default is `True`. Specify whether confusion matrix should be logged.
    score_threshold : int, optional
        Default is 0.5. Score threshold to determine prediction based on the model output score.


    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer : torch.optim.Optimizer
        The optimizer to minimize loss with.
    scheduler : torch.optim.lr_scheduler
        The LR scheduler to use with optimizer.
    train_loader : torch.utils.data.DataLoader
        Training data loader.
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    masked_targets : bool
        Whether the training dataset generates targets with a mask of existing targets or not
    n_epochs : int
        The maximum number of epochs to iterate over the dataset.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    nth_step_log_embeddings : int or None
        The frequency with which to save cell type embeddings.
    device : torch.device
        Device on which the computation is carried out.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    output_dir : str
        The directory to save model checkpoints and logs.
    training_loss : list(float)
        The current training loss.
    metrics : dict
        A dictionary that maps metric names (`str`) to metric functions.
        By default, this contains `"roc_auc"`, which maps to
        `sklearn.metrics.roc_auc_score`, and `"average_precision"`,
        which maps to `sklearn.metrics.average_precision_score`.

    """

    def __init__(
        self,
        model,
        # configs,
        # n_folds,
        # fold,
        ct_masks,
        n_cell_types,
        loss_criterion,
        optimizer_class,
        optimizer_kwargs,
        dataloaders,
        # train_loader,
        # val_loader,
        n_epochs,
        report_stats_every_n_steps,
        output_dir,
        scheduler_class=None,
        scheduler_kwargs=None,
        save_checkpoint_every_n_steps=1000,
        save_new_checkpoints_after_n_steps=None,
        log_embeddings_every_n_steps=8000,
        report_gt_feature_n_positives=10,
        cpu_n_threads=1,
        device="cpu",
        data_parallel=False,
        logging_verbosity=2,
        checkpoint_resume=None,
        metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score),
        log_confusion_matrix=True,
        score_threshold=0.5,
    ):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model
        self.checkpoint_resume = checkpoint_resume
        # self.cfg = configs,
        # self.n_folds = n_folds  # self.cfg["dataset"]["dataset_args"]["n_folds"]
        self.n_cell_types = n_cell_types  # self.cfg["dataset"]["dataset_args"]["n_cell_types"]
        # self.fold = fold  # self.cfg["dataset"]["dataset_args"]["fold"]
        self.ct_masks = ct_masks
        self.dataloaders = dataloaders
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        if scheduler_class is not None:
            if scheduler_kwargs is None:
                scheduler_kwargs = dict()
            self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)

        self.masked_targets = True
        self.batch_size = dataloaders[0][0].batch_size
        self.n_epochs = n_epochs
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        self.nth_step_log_embeddings = log_embeddings_every_n_steps

        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self.save_new_checkpoints = save_new_checkpoints_after_n_steps

        logger.info(
            "Training parameters set: batch size {0}, "
            "reporting every {1} steps, "
            "number of epochs: {2}".format(
                self.batch_size, self.nth_step_report_stats, self.n_epochs
            )
        )

        torch.set_num_threads(cpu_n_threads)

        self.device = torch.device(device)
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")
        else:
            self.model.to(self.device)
            self.criterion.to(self.device)
            logger.debug(f"Set modules to use device {device}")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._writer = SummaryWriter(os.path.join(self.output_dir))

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity,
        )

        self._validation_metrics = PerformanceMetrics(
            lambda idx: self.dataloaders[0][0].dataset.dataset.target_features[idx],
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics,
        )
        self._test_metrics = PerformanceMetrics(
            lambda idx: self.dataloaders[0][0].dataset.dataset.target_features[idx],
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics,
        )
        self.log_confusion_matrix = log_confusion_matrix

        self._start_step = 0
        # TODO: Should this be set when it is used later? Would need to if we want to
        # train model 2x in one run.
        self._min_loss = float("inf")

        if checkpoint_resume is not None:
            checkpoint = torch.load(
                checkpoint_resume, map_location=lambda storage, location: storage
            )
            if "state_dict" not in checkpoint:
                raise ValueError(
                    "Selene does not support continued "
                    "training of models that were not originally "
                    "trained using Selene."
                )

            self.model = load_model_from_state_dict(
                checkpoint["state_dict"], self.model
            )

            self._start_step = checkpoint["step"]
            # if self._start_step >= self.n_epochs:
            #    self.n_epochs += self._start_step

            self._min_loss = checkpoint["min_loss"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            logger.info(
                ("Resuming from checkpoint: step {0}, min loss {1}").format(
                    self._start_step, self._min_loss
                )
            )

        self._train_logger = _metrics_logger(
            "{0}.train".format(__name__), self.output_dir
        )
        self._validation_logger = _metrics_logger(
            "{0}.validation".format(__name__), self.output_dir
        )

        self._train_logger.info("loss")
        self._validation_logger.info(
            "\t".join(
                ["loss"] + sorted([x for x in self._validation_metrics.metrics.keys()])
            )
        )

        self.score_threshold = score_threshold

        self.val_reduction_factor = 1
        # val_batch_size = self.val_loader.batch_size
        # val_target_size = self.val_loader.dataset.target_size
        # total_val_target_size = len(self.val_loader) * val_batch_size * val_target_size
        # if total_val_target_size > MAX_TOTAL_VAL_TARGET_SIZE:
        #     self.val_reduction_factor = math.ceil(
        #         total_val_target_size / MAX_TOTAL_VAL_TARGET_SIZE
        #     )


    def train_and_crossval(self):
        
        min_loss = self._min_loss

        time_per_batch = []
        report_train_losses = []
        report_train_predictions = []
        report_train_targets = []
        report_train_target_masks = []
 
        total_steps = self._start_step

        # make necessary LR scheduler steps
        # if they are not based on validation loss
        for step in range(1, total_steps + 1):
            self._update_and_log_lr_if_needed(step, log=False)

        time_per_batch = []
        report_train_losses = []
        report_train_predictions = []
        report_train_targets = []
        report_train_target_masks = [] 

        checkpoint_epoch = 0
        if self.checkpoint_resume is not None:
            
            checkpoint_fold = 5

            print(f'Start training from {checkpoint_epoch} epoch, fold {checkpoint_fold}')

            self.cur_fold = checkpoint_fold
            for _, (train_batch_loader, valid_batch_loader) in enumerate(self.dataloaders[checkpoint_fold:]):
                print('epoch:', checkpoint_epoch, 'fold:', self.cur_fold)
                # train
                for batch in tqdm(train_batch_loader):
                    t_i = time()
                    prediction, target, target_mask, loss = self.train(batch)
                    t_f = time()
                    time_per_batch.append(t_f - t_i)
                    report_train_losses.append(loss)
                    report_train_predictions.append(prediction)
                    report_train_targets.append(target)
                    if self.masked_targets:
                        report_train_target_masks.append(target_mask)
                    total_steps += 1
                    self._update_and_log_lr_if_needed(total_steps)
                
                # метрики на train для отчета
                self._log_train_metrics_and_clean_cache(
                    checkpoint_epoch,
                    total_steps,
                    time_per_batch,
                    report_train_losses,
                    report_train_predictions,
                    report_train_targets,
                    report_train_target_masks,
                )

                # val
                validation_loss = self._validate_and_log_metrics(
                    val_loader=valid_batch_loader, 
                    step=total_steps
                    )
                
                self._update_and_log_lr_if_needed(
                    total_steps, math.ceil(validation_loss * 1000.0) / 1000.0
                )

                if validation_loss < min_loss:
                    min_loss = validation_loss
                    self._save_checkpoint(total_steps, min_loss, is_best=True)
                    logger.info("Updating `best_model.pth.tar`")

                self._writer.add_scalar("fold", self.cur_fold, total_steps)

                self.cur_fold += 1

            # if total_steps % self.nth_step_save_checkpoint == 0:
            checkpoint_basename = "checkpoint"
            if (self.save_new_checkpoints is not None
                and self.save_new_checkpoints >= total_steps):
                checkpoint_basename = "checkpoint-{0}".format(
                    strftime("%m%d%H%M%S")
                )
            self._save_checkpoint(
                total_steps,
                min_loss,
                is_best=False,
                filename=checkpoint_basename,
            )
            logger.debug(
                "Saving checkpoint `{0}.pth.tar`".format(checkpoint_basename)
            )

            self._log_embeddings(total_steps)

            checkpoint_epoch += 1

            print(f'Epoch from checkpoint completed; Start {checkpoint_epoch} epoch')

        for epoch in tqdm(range(checkpoint_epoch, self.n_epochs)):

            for fold, (train_batch_loader, valid_batch_loader) in enumerate(self.dataloaders):
                self.cur_fold = fold
                print('epoch:', epoch, 'fold:', self.cur_fold)

                # train
                for batch in tqdm(train_batch_loader):
                    t_i = time()
                    prediction, target, target_mask, loss = self.train(batch)
                    t_f = time()
                    time_per_batch.append(t_f - t_i)
                    report_train_losses.append(loss)
                    report_train_predictions.append(prediction)
                    report_train_targets.append(target)
                    if self.masked_targets:
                        report_train_target_masks.append(target_mask)
                    total_steps += 1
                    self._update_and_log_lr_if_needed(total_steps)

                # метрики на train для отчета
                self._log_train_metrics_and_clean_cache(
                    epoch,
                    total_steps,
                    time_per_batch,
                    report_train_losses,
                    report_train_predictions,
                    report_train_targets,
                    report_train_target_masks,
                )

                # val
                validation_loss = self._validate_and_log_metrics(
                    val_loader=valid_batch_loader, 
                    step=total_steps
                    )
                
                self._update_and_log_lr_if_needed(
                    total_steps, math.ceil(validation_loss * 1000.0) / 1000.0
                )

                if validation_loss < min_loss:
                    min_loss = validation_loss
                    self._save_checkpoint(total_steps, min_loss, is_best=True)
                    logger.info("Updating `best_model.pth.tar`")

                self._writer.add_scalar("fold", self.cur_fold, total_steps)

            # if total_steps % self.nth_step_save_checkpoint == 0:
            checkpoint_basename = "checkpoint"
            if (self.save_new_checkpoints is not None
                and self.save_new_checkpoints >= total_steps):
                checkpoint_basename = "checkpoint-{0}".format(
                    strftime("%m%d%H%M%S")
                )
            self._save_checkpoint(
                total_steps,
                min_loss,
                is_best=False,
                filename=checkpoint_basename,
            )
            logger.debug(
                "Saving checkpoint `{0}.pth.tar`".format(checkpoint_basename)
            )

            self._log_embeddings(total_steps)
        self._writer.flush()

        return None


    def train(self, batch):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        self.model.train()

        if self.masked_targets:
            # retrieved_seq, cell_type, target, target_mask
            sequence_batch = batch[0].to(self.device)
            cell_type_batch = batch[1].to(self.device)
            targets = batch[2].to(self.device)
            target_mask = batch[3].to(self.device)

            # make train mask
            target_mask_tr = target_mask.clone()
            target_mask_tr[:, self.ct_masks[self.cur_fold]] = False
            # self.target_mask_tr = target_mask_tr

            outputs = self.model(sequence_batch, cell_type_batch)

            self.criterion.weight = target_mask_tr
        else:
            # retrieved_seq, target
            sequence_batch = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            outputs = self.model(sequence_batch)

        loss = self.criterion(outputs, targets)

        if self.criterion.reduction == "sum":
            loss = loss / self.criterion.weight.sum()
        predictions = torch.sigmoid(outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.masked_targets:
            target_mask = target_mask_tr.cpu().detach().numpy()
        else:
            target_mask = None

        return (
            predictions.cpu().detach().numpy(),
            targets.cpu().detach().numpy(),
            target_mask,
            loss.item(),
        )

    def _evaluate_on_ct(self, data_loader):
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
        all_baselines = []
        if self.masked_targets:
            all_target_masks = []
        else:
            all_target_masks = None

        for batch in tqdm(data_loader):
            sequence_batch = batch[0].to(self.device)
            cell_type_batch = batch[1].to(self.device)
            targets = batch[2].to(self.device)
            target_mask = batch[3].to(self.device)
            # print('targets shape:', targets.shape)

            # val mask
            target_mask_tr = target_mask.clone()
            target_mask_tr[:, self.ct_masks[self.cur_fold]] = False
            target_mask_val = ~target_mask_tr

            # compute a baseline
            baseline = (targets * target_mask_tr).sum(axis=1) / target_mask_tr.sum(axis=1)
            baseline = torch.repeat_interleave(baseline.unsqueeze(1), self.n_cell_types, dim=1)

            with torch.no_grad():
                if self.masked_targets:
                    outputs = self.model(sequence_batch, cell_type_batch)
                    self.criterion.weight = target_mask_val
                else:
                    outputs = self.model(sequence_batch)

                loss = self.criterion(outputs, targets)
                if self.criterion.reduction == "sum":
                    loss = loss / self.criterion.weight.sum()

                predictions = torch.sigmoid(outputs)
                predictions = predictions.view(-1, predictions.shape[-1])
                targets = targets.view(-1, targets.shape[-1])
                baseline = baseline.view(-1, baseline.shape[-1])

                if self.masked_targets:
                    target_mask = target_mask_val.view(-1, target_mask_val.shape[-1])

                if self.val_reduction_factor > 1:
                    reduced_val_batch_size = (
                        predictions.shape[0] // self.val_reduction_factor
                    )
                    reduced_index = np.random.choice(
                        predictions.shape[0], reduced_val_batch_size
                    )

                    predictions = predictions[reduced_index]
                    targets = targets[reduced_index]
                    baseline = baseline[reduced_index]
                    if self.masked_targets:
                        target_mask = target_mask[reduced_index]

                all_predictions.append(predictions.data.cpu().numpy())
                all_targets.append(targets.data.cpu().numpy())
                all_baselines.append(baseline.data.cpu().numpy())
                if self.masked_targets:
                    all_target_masks.append(target_mask.data.cpu().numpy())
                batch_losses.append(loss.item())

        all_predictions = expand_dims(np.concatenate(all_predictions))
        all_targets = expand_dims(np.concatenate(all_targets))
        all_baselines = expand_dims(np.concatenate(all_baselines))
        if self.masked_targets:
            all_target_masks = expand_dims(np.concatenate(all_target_masks))

        return (
            np.average(batch_losses), 
            all_predictions, 
            all_targets, 
            all_baselines, 
            all_target_masks
        )


    def _update_and_log_lr_if_needed(self, total_steps, validation_loss=None, log=True):
        # torch.optim.lr_scheduler.ReduceLROnPlateau is the only scheduler
        # that takes some value as input to `.step()`
        if self.scheduler is not None:
            if validation_loss is None and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()
                if log:
                    self._log_lr(total_steps)
            elif validation_loss is not None and isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(validation_loss)
                if log:
                    self._log_lr(total_steps)

    def _log_train_metrics_and_clean_cache(
        self,
        epoch,
        step,
        time_per_batch,
        train_losses,
        train_predictions,
        train_targets,
        train_target_masks,
    ):
        logger.info(
            (
                "[epoch {0} / step {1}] average number " "of steps per second: {2:.1f}"
            ).format(epoch, step, 1.0 / np.average(time_per_batch))
        )

        train_loss = np.average(train_losses)
        self._train_logger.info(train_loss)
        self._writer.add_scalar("loss/train", train_loss, step)

        if self.masked_targets:
            train_target_masks = expand_dims(np.concatenate(train_target_masks))
        train_scores = self._compute_metrics(
            expand_dims(np.concatenate(train_predictions)),
            expand_dims(np.concatenate(train_targets)),
            train_target_masks,
            log_prefix="train",
        )

        for k in sorted(self._validation_metrics.metrics.keys()):
            if k in train_scores and train_scores[k]:
                self._writer.add_scalar(f"model_{k}/train", train_scores[k], step)

        logger.info("training loss: {0}".format(train_loss))


    def _validate_and_log_metrics(self, val_loader, step):
        (
            valid_scores, 
            baselines_scores, 
            all_predictions, 
            all_targets, 
            all_target_masks
         ) = self.validate(val_loader)

        validation_loss = valid_scores["loss"]
        self._writer.add_scalar("loss/test", validation_loss, step)
        to_log = [str(validation_loss)]

        # log model metrics
        for k in sorted(self._validation_metrics.metrics.keys()):
            if k in valid_scores and valid_scores[k]:
                to_log.append(str(valid_scores[k]))
                to_log.append(str(baselines_scores[k]))
                self._writer.add_scalar(f"model_{k}/val", valid_scores[k], step)
                self._writer.add_scalar(f"baseline_{k}/val", baselines_scores[k], step)
            else:
                to_log.append("NA")

        self._validation_logger.info("\t".join(to_log))

        logger.info("validation loss: {0}".format(validation_loss))

        if self.log_confusion_matrix:
            if self.masked_targets:
                masked_targets = all_targets.flatten()[all_target_masks.flatten()]
                masked_predictions = all_predictions.flatten()[
                    all_target_masks.flatten()
                ]
                cm = confusion_matrix(
                    masked_targets, masked_predictions > self.score_threshold
                )
            else:
                cm = confusion_matrix(
                    all_targets.flatten(),
                    all_predictions.flatten() > self.score_threshold,
                )
            cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_plot.plot()
            self._writer.add_figure(
                "confusion_matrix", cm_plot.figure_, global_step=step
            )

        return validation_loss


    def _evaluate_on_data(self, data_loader):
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

        for batch in tqdm(data_loader):
            if self.masked_targets:
                sequence_batch = batch[0].to(self.device)
                cell_type_batch = batch[1].to(self.device)
                targets = batch[2].to(self.device)
                target_mask = batch[3].to(self.device)
                # print('targets', targets.shape)

                # val mask
                batch_mask = next(self.mask_iterator)
                target_mask_val = target_mask.clone()
                target_mask_val[:, batch_mask[0]: batch_mask[-1]] = False
                self.target_mask_val = target_mask_val
                # self.target_mask_val = ~self.target_mask_tr
                if self.target_mask_val.shape[0] != targets.shape[0]:
                    self.target_mask_val = self.target_mask_val[:targets.shape[0], ...]

                # targets = targets * self.target_mask_val
                # print('masked targets', targets.shape)
            else:
                # retrieved_seq, target
                sequence_batch = batch[0].to(self.device)
                targets = batch[1].to(self.device)

            with torch.no_grad():
                if self.masked_targets:
                    outputs = self.model(sequence_batch, cell_type_batch)
                    # print('outputs', outputs.shape)
                    self.criterion.weight = self.target_mask_val
                else:
                    outputs = self.model(sequence_batch)
                loss = self.criterion(outputs, targets)
                if self.criterion.reduction == "sum":
                    loss = loss / self.criterion.weight.sum()
                predictions = torch.sigmoid(outputs)
                # predictions = predictions * self.target_mask_val

                predictions = predictions.view(-1, predictions.shape[-1])
                targets = targets.view(-1, targets.shape[-1])
                if self.masked_targets:
                    target_mask = self.target_mask_val.view(-1, self.target_mask_val.shape[-1])
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
        return np.average(batch_losses), all_predictions, all_targets, all_target_masks


    def _compute_metrics(self, predictions, targets, target_mask, log_prefix=None):
        """
        Measures performance on given predictions and targets.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        # TODO(arlapin): Should use _train_metrics for "train"?
        scores = self._validation_metrics.update(
            predictions, targets, target_mask)
        if log_prefix:
            for name, score in scores.items():
                logger.info(f"{log_prefix} {name}: {score}")

        return scores


    def _compute_baseline_score(self, baseline, targets, target_mask, log_prefix=None):
        """
        Measures performance on given predictions and targets.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        
        scores = self._validation_metrics.update(baseline, targets, target_mask)
        if log_prefix:
            for name, score in scores.items():
                logger.info(f"{log_prefix} baseline {name}: {score}")

        return scores


    def validate(self, val_loader):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        (
            average_loss,
            all_predictions,
            all_targets,
            baseline,
            all_target_masks
        ) = self._evaluate_on_ct(val_loader)

        average_scores = self._compute_metrics(
            all_predictions, all_targets, all_target_masks, log_prefix="validation"
        )
        baselines_scores = self._compute_baseline_score(
            baseline, all_targets, all_target_masks, log_prefix="validation"
        )
        average_scores["loss"] = average_loss

        return average_scores, baselines_scores, all_predictions, all_targets, all_target_masks


    def _save_checkpoint(self, step, min_loss, is_best, filename="checkpoint"):
        """
        Saves snapshot of the model state to file. Will save a checkpoint
        with name `<filename>.pth.tar` and, if this is the model's best
        performance so far, will save the state to a `best_model.pth.tar`
        file as well.
        Models are saved in the state dictionary format. This is a more
        stable format compared to saving the whole model (which is another
        option supported by PyTorch). Note that we do save a number of
        additional, Selene-specific parameters in the dictionary
        and that the actual `model.state_dict()` is stored in the `state_dict`
        key of the dictionary loaded by `torch.load`.
        See: https://pytorch.org/docs/stable/notes/serialization.html for more
        information about how models are saved in PyTorch.
        Parameters
        ----------
        state : dict
            Information about the state of the model. Note that this is
            not `model.state_dict()`, but rather, a dictionary containing
            keys that can be used for continued training in Selene
            _in addition_ to a key `state_dict` that contains
            `model.state_dict()`.
        is_best : bool
            Is this the model's best performance so far?
        filename : str, optional
            Default is "checkpoint". Specify the checkpoint filename. Will
            append a file extension to the end of the `filename`
            (e.g. `checkpoint.pth.tar`).
        Returns
        -------
        None
        """
        state = {
            "step": step,
            "arch": self.model.__class__.__name__,
            "state_dict": self.model.state_dict(),
            "min_loss": min_loss,
            "optimizer": self.optimizer.state_dict(),
        }

        logger.debug("[TRAIN] {0}: Saving model state to file.".format(state["step"]))
        cp_filepath = os.path.join(self.output_dir, filename)
        torch.save(state, "{0}.pth.tar".format(cp_filepath))
        if is_best:
            best_filepath = os.path.join(self.output_dir, "best_model")
            shutil.copyfile(
                "{0}.pth.tar".format(cp_filepath), "{0}.pth.tar".format(best_filepath)
            )

    def _log_lr(self, step):
        lrs = [group["lr"] for group in self.optimizer.param_groups]
        for index, lr in enumerate(lrs):
            self._writer.add_scalar("lr_{}".format(index), lr, step)

    def _log_embeddings(self, step):
        embeddings = self.model.get_cell_type_embeddings()
        cell_type_labels = self.dataloaders[0][0].dataset.dataset._cell_types
        self._writer.add_embedding(
            embeddings, metadata=cell_type_labels, global_step=step
        )
        