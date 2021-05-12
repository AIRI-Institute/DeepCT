"""
This module provides the `TrainModel` class and supporting methods.

It has been modified to retrieve the cell type embeddings used during training.

It also works with RPKMFileSampler to pass cell targets to the model as input, 
which specify the cell types to train on for each sequence in the mini-batch.

The batch size can now be artifically increased by aggregating loss across 
multiple "sub" batches, and then backpropagating. 
"""
import csv
import logging
import math
import os
import shutil
from time import strftime, time

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
from torch.autograd import Variable
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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


class TrainModel(object):
    """
    This class ties together the various objects and methods needed to
    train and validate a model.

    TrainModel saves a checkpoint model (overwriting it after
    `save_checkpoint_every_n_steps`) as well as a best-performing model
    (overwriting it after `report_stats_every_n_steps` if the latest
    validation performance is better than the previous best-performing
    model) to `output_dir`.

    TrainModel also outputs 2 files that can be used to monitor training
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
    data_sampler : selene_sdk.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    optimizer_kwargs : dict
        The dictionary of keyword arguments to pass to the optimizer's
        constructor.
    batch_size : int
        Specify the batch size to process examples.
    sub_batch_size : int
        The number of samples used for each forward pass.
        This allows the effective training batch_size to be larger than allowed
        by memory constraints by splitting the batch into sub-batches, 
        aggregating loss across the sub-batches, and doing a backward pass every 
        batch_size/sub_batch_size sub-batches. Defaults to batch_size when not provided.
    validate_batch_size : int
        Specify the batch size to process validation examples.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    report_stats_every_n_steps : int
        The frequency with which to report summary statistics. You can
        set this value to be equivalent to a training epoch
        (`n_steps * _batch_size`) being the total number of samples
        seen by the model so far. Selene evaluates the model on the validation
        dataset every `report_stats_every_n_steps` and, if the model obtains
        the best performance so far (based on the user-specified loss function),
        Selene saves the model state to a file called `best_model.pth.tar` in
        `output_dir`.
    embeddings_path : str
        Path to file that stores cell type embeddings of all cell types used 
        during training.
    output_dir : str
        The output directory to save model checkpoints and logs in.
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
    n_validation_samples : int or None, optional
        Default is `None`. Specify the number of validation samples in the
        validation set. If `n_validation_samples` is `None` and the data sampler
        used is the `selene_sdk.samplers.IntervalsSampler` or
        `selene_sdk.samplers.RandomSampler`, we will retrieve 32000
        validation samples. If `None` and using
        `selene_sdk.samplers.MultiFileSampler`, we will use all
        available validation samples from the appropriate data file.
    n_test_samples : int or None, optional
        Default is `None`. Specify the number of test samples in the test set.
        If `n_test_samples` is `None` and

            - the sampler you specified has no test partition, you should not
              specify `evaluate` as one of the operations in the `ops` list.
              That is, Selene will not automatically evaluate your trained
              model on a test dataset, because the sampler you are using does
              not have any test data.
            - the sampler you use is of type `selene_sdk.samplers.OnlineSampler`
              (and the test partition exists), we will retrieve 640000 test
              samples.
            - the sampler you use is of type
              `selene_sdk.samplers.MultiFileSampler` (and the test partition
              exists), we will use all the test samples available in the
              appropriate data file.

    cpu_n_threads : int, optional
        Default is 1. Sets the number of OpenMP threads used for parallelizing
        CPU operations.
    use_cuda : bool, optional
        Default is `False`. Specify whether a CUDA-enabled GPU is available
        for torch to use during training.
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

    Attributes
    ----------
    model : torch.nn.Module
        The model to train.
    sampler : selene_sdk.samplers.Sampler
        The example generator.
    loss_criterion : torch.nn._Loss
        The loss function to optimize.
    optimizer_class : torch.optim.Optimizer
        The optimizer to minimize loss with.
    max_steps : int
        The maximum number of mini-batches to iterate over.
    nth_step_report_stats : int
        The frequency with which to report summary statistics.
    nth_step_save_checkpoint : int
        The frequency with which to save a model checkpoint.
    use_cuda : bool
        If `True`, use a CUDA-enabled GPU. If `False`, use the CPU.
    data_parallel : bool
        Whether to use multiple GPUs or not.
    embeddings_path : str
        The path to the file with cell type embeddings.
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
        data_sampler,
        loss_criterion,
        optimizer_class,
        optimizer_kwargs,
        batch_size,
        validate_batch_size,
        max_steps,
        report_stats_every_n_steps,
        embeddings_path,
        output_dir,
        sub_batch_size=None,
        save_checkpoint_every_n_steps=1000,
        save_new_checkpoints_after_n_steps=None,
        report_gt_feature_n_positives=10,
        n_validation_samples=None,
        n_test_samples=None,
        cpu_n_threads=1,
        use_cuda=False,
        data_parallel=False,
        logging_verbosity=2,
        checkpoint_resume=None,
        metrics=dict(roc_auc=roc_auc_score, average_precision=average_precision_score),
    ):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

        if sub_batch_size is None:
            self._sub_batch_size = batch_size
            self._aggregate_steps = 1
        else:
            assert (
                batch_size % sub_batch_size == 0
            ), "batch_size must be a multiple of sub_batch_size."
            self._sub_batch_size = sub_batch_size
            self._aggregate_steps = batch_size // sub_batch_size

        self._batch_size = batch_size
        self._validate_batch_size = validate_batch_size
        self.max_steps = max_steps
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self.save_new_checkpoints = save_new_checkpoints_after_n_steps

        logger.info(
            "Training parameters set: batch size {0}, "
            "number of steps per 'epoch': {1}, "
            "maximum number of steps: {2}".format(
                self._batch_size, self.nth_step_report_stats, self.max_steps
            )
        )

        torch.set_num_threads(cpu_n_threads)

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._writer = SummaryWriter(os.path.join(self.output_dir))

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity,
        )

        self._create_validation_set(n_samples=n_validation_samples)

        self._validation_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics,
        )

        if "test" in self.sampler.modes:
            self._test_data = None
            self._n_test_samples = n_test_samples
            self._test_metrics = PerformanceMetrics(
                self.sampler.get_feature_from_index,
                report_gt_feature_n_positives=report_gt_feature_n_positives,
                metrics=metrics,
            )

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
            if self._start_step >= self.max_steps:
                self.max_steps += self._start_step

            self._min_loss = checkpoint["min_loss"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

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

        self.embeddings = self.get_embeddings(embeddings_path)

    def _create_validation_set(self, n_samples=None):
        """
        Generates the set of validation examples.

        Parameters
        ----------
        n_samples : int or None, optional
            Default is `None`. The size of the validation set. If `None`,
            will use all validation examples in the sampler.

        """
        logger.info("Creating validation dataset.")
        t_i = time()
        (
            self._validation_data,
            self._all_validation_targets,
            self._all_validation_cell_targets,
        ) = self.sampler.get_validation_set(
            self._validate_batch_size, n_samples=n_samples
        )
        t_f = time()
        logger.info(
            (
                "{0} s to load {1} validation examples ({2} validation "
                "batches) to evaluate after each training step."
            ).format(
                t_f - t_i,
                len(self._validation_data) * self._validate_batch_size,
                len(self._validation_data),
            )
        )

    def create_test_set(self):
        """
        Loads the set of test samples.
        We do not create the test set in the `TrainModel` object until
        this method is called, so that we avoid having to load it into
        memory until the model has been trained and is ready to be
        evaluated.

        """
        logger.info("Creating test dataset.")
        t_i = time()
        (
            self._test_data,
            self._all_test_targets,
            self._all_test_cell_targets,
        ) = self.sampler.get_test_set(
            self._validate_batch_size, n_samples=self._n_test_samples
        )
        t_f = time()
        logger.info(
            (
                "{0} s to load {1} test examples ({2} test batches) "
                "to evaluate after all training steps."
            ).format(
                t_f - t_i,
                len(self._test_data) * self._validate_batch_size,
                len(self._test_data),
            )
        )

    def _get_batch(self, batch_size):
        """
        Fetches a mini-batch of examples

        Returns
        -------
        SamplesBatch
            A batch containing the examples and targets.
        cell_targets
            ndarray which maps targets to cell types used for each
            sample within the mini-batch.

        """
        t_i_sampling = time()
        samples_batch, cell_targets = self.sampler.sample(batch_size=batch_size)
        t_f_sampling = time()
        logger.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                batch_size, t_f_sampling - t_i_sampling
            )
        )
        return samples_batch, cell_targets

    def get_embeddings(self, embeddings_path):
        """
        Fetches all embeddings used for training the model

        Returns
        -------
        embeddings
            Torch tensor.
        """
        rows = []
        with open(embeddings_path) as csvfile:
            csvreader = csv.reader(csvfile)

            for row in csvreader:
                rows.append(row)

        embeddings = []
        for row in rows:
            embeddings.append(list(map(float, row[1:])))

        embeddings = torch.Tensor(np.array(embeddings, dtype="float32"))

        if self.use_cuda:
            embeddings = embeddings.cuda()

        return embeddings

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.

        """
        min_loss = self._min_loss
        self._scheduler = CosineAnnealingLR(self.optimizer, self.max_steps)

        time_per_step = []
        train_losses = []
        train_predictions = []
        train_targets = []

        self.optimizer.zero_grad()

        for step in tqdm(range(self._start_step, self.max_steps)):
            t_i = time()
            prediction, target, loss = self.train(step)
            t_f = time()
            train_losses.append(loss)
            train_predictions.append(prediction)
            train_targets.append(target)
            time_per_step.append(t_f - t_i)

            if step % self.nth_step_save_checkpoint == 0:
                checkpoint_basename = "checkpoint"
                if (
                    self.save_new_checkpoints is not None
                    and self.save_new_checkpoints >= step
                ):
                    checkpoint_basename = "checkpoint-{0}".format(
                        strftime("%m%d%H%M%S")
                    )

                self._save_checkpoint(
                    step, min_loss, is_best=False, filename=checkpoint_basename
                )
                logger.debug(
                    "Saving checkpoint `{0}.pth.tar`".format(checkpoint_basename)
                )

            if step and step % self.nth_step_report_stats == 0:
                self._log_train_metrics_and_clean_cache(
                    step, time_per_step, train_losses, train_predictions, train_targets
                )

                validation_loss = self._validate_and_log_metrics(step)

                if validation_loss < min_loss:
                    min_loss = validation_loss
                    self._save_checkpoint(step, min_loss, is_best=True)
                    logger.info("Updating `best_model.pth.tar`")

        self.sampler.save_dataset_to_file("train", close_filehandle=True)
        self._writer.flush()

    def train(self, step):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        self.model.train()
        self.sampler.set_mode("train")

        samples_batch, cell_targets = self._get_batch(batch_size=self._sub_batch_size)
        inputs, targets = samples_batch.torch_inputs_and_targets(self.use_cuda)

        cell_targets = torch.from_numpy(cell_targets)
        if self.use_cuda:
            cell_targets = cell_targets.cuda()

        predictions = self.model(inputs, cell_targets, self.embeddings)
        loss = self.criterion(predictions, targets)

        loss.backward()
        if (step + 1) % self._aggregate_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._scheduler.step()
            self._log_lr(step)

        return (
            predictions.cpu().detach().numpy(),
            targets.cpu().detach().numpy(),
            loss.item(),
        )

    def _log_train_metrics_and_clean_cache(
        self, step, time_per_step, train_losses, train_predictions, train_targets
    ):
        logger.info(
            ("[STEP {0}] average number " "of steps per second: {1:.1f}").format(
                step, 1.0 / np.average(time_per_step)
            )
        )
        time_per_step[:] = []

        train_loss = np.average(train_losses)
        self._train_logger.info(train_loss)
        self._writer.add_scalar("loss/train", train_loss, step)
        train_losses[:] = []

        train_scores = self._compute_metrics(
            np.concatenate(train_predictions),
            np.concatenate(train_targets),
            log_prefix="train",
        )
        train_predictions[:] = []
        train_targets[:] = []

        for k in sorted(self._validation_metrics.metrics.keys()):
            if k in train_scores and train_scores[k]:
                self._writer.add_scalar("{}/train".format(k), train_scores[k], step)

        logger.info("training loss: {0}".format(train_loss))

    def _validate_and_log_metrics(self, step):
        valid_scores, all_predictions = self.validate()
        validation_loss = valid_scores["loss"]
        self._writer.add_scalar("loss/test", validation_loss, step)
        to_log = [str(validation_loss)]

        for k in sorted(self._validation_metrics.metrics.keys()):
            if k in valid_scores and valid_scores[k]:
                to_log.append(str(valid_scores[k]))
                self._writer.add_scalar("{}/test".format(k), valid_scores[k], step)
            else:
                to_log.append("NA")
        self._validation_logger.info("\t".join(to_log))

        logger.info("validation loss: {0}".format(validation_loss))

        #         cm = confusion_matrix(
        #             self._all_validation_targets.flatten(), all_predictions.flatten() > 0.5
        #         )
        #         cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm)
        #         cm_plot.plot()
        #         self._writer.add_figure("confusion_matrix", cm_plot.figure_, global_step=step)

        return validation_loss

    def _evaluate_on_data(self, data_in_batches, all_cell_targets):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        data_in_batches : list(SamplesBatch)
            A list of tuples of the data, where the first element is
            the example, and the second element is the label.
        all_cell_targets: list(numpy.ndarray)
            A list of (batch_size, n_cell_types) ndarrays that maps
            targets to the cell types they refer to.

        Returns
        -------
        tuple(float, list(numpy.ndarray))
            Returns the average loss, and the list of all predictions.

        """
        self.model.eval()

        batch_losses = []
        all_predictions = []

        for index, samples_batch in enumerate(data_in_batches):
            inputs, targets = samples_batch.torch_inputs_and_targets(self.use_cuda)

            cell_targets = all_cell_targets[index]
            cell_targets = torch.from_numpy(cell_targets)
            if self.use_cuda:
                cell_targets = cell_targets.cuda()

            with torch.no_grad():
                predictions = self.model(inputs, cell_targets, self.embeddings)
                loss = self.criterion(predictions, targets)

                all_predictions.append(predictions.data.cpu().numpy())

                batch_losses.append(loss.item())
        all_predictions = np.vstack(all_predictions)
        return np.average(batch_losses), all_predictions

    def _compute_metrics(self, predictions, targets, log_prefix=None):
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
        scores = self._validation_metrics.update(predictions, targets)
        if log_prefix:
            for name, score in scores.items():
                logger.info("{} {}: {}".format(log_prefix, name, score))

        return scores

    def validate(self):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        average_loss, all_predictions = self._evaluate_on_data(
            self._validation_data, self._all_validation_cell_targets
        )
        average_scores = self._compute_metrics(
            all_predictions, self._all_validation_targets, log_prefix="validation"
        )
        average_scores["loss"] = average_loss

        return average_scores, all_predictions

    def evaluate(self):
        """
        Measures the model test performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the test set.

        """
        if self._test_data is None:
            self.create_test_set()
        average_loss, all_predictions = self._evaluate_on_data(
            self._test_data, self._all_test_cell_targets
        )

        average_scores = self._test_metrics.update(
            all_predictions, self._all_test_targets
        )
        np.savez_compressed(
            os.path.join(self.output_dir, "test_predictions.npz"), data=all_predictions
        )

        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(self.output_dir, "test_performance.txt")
        feature_scores_dict = self._test_metrics.write_feature_scores_to_file(
            test_performance
        )

        average_scores["loss"] = average_loss

        self._test_metrics.visualize(
            all_predictions, self._all_test_targets, self.output_dir
        )

        return (average_scores, feature_scores_dict)

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
