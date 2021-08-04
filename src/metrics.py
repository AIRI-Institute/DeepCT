import functools

import numpy as np
import scipy.stats
import sklearn.metrics as metrics

# Metric helpers which convert float values to binary (0/1).
# Regression metrics helper functions.


def _to_binary(x: np.ndarray, threshold=0.5) -> np.ndarray:
    return np.where(x > threshold, 1, 0).astype(int)


def binary_inputs(score_func):
    """Wrapper function for input binarization using specified threshold(s)"""

    def binary_wrapper(y_true, y_pred, threshold=0.5, **kwargs):
        if isinstance(threshold, float):
            binary_y_true = _to_binary(y_true, threshold)
            binary_y_pred = _to_binary(y_pred, threshold)
        else:
            mask = np.logical_or(y_pred > threshold[1], y_pred < threshold[0])
            binarization_thresh = (threshold[0] + threshold[1]) / 2
            y_pred = _to_binary(y_pred, binarization_thresh)
            binary_y_true = y_true[mask]
            binary_y_pred = y_pred[mask]
        return score_func(binary_y_true, binary_y_pred, **kwargs)

    return binary_wrapper


def threshold_wrapper(score_func, threshold):
    return functools.partial(score_func, threshold=threshold)


@binary_inputs
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    return metrics.accuracy_score(y_true, y_pred, **kwargs)


@binary_inputs
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    return metrics.f1_score(y_true, y_pred, **kwargs)


@binary_inputs
def precision_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    return metrics.precision_score(y_true, y_pred, **kwargs)


@binary_inputs
def recall_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    return metrics.recall_score(y_true, y_pred, **kwargs)


@binary_inputs
def jaccard_score(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    return metrics.jaccard_score(y_true, y_pred, **kwargs)


def jaccard_multi_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
    **kwargs
) -> float:
    multi_thresh_jaccard = []
    for threshold in thresholds:
        multi_thresh_jaccard.append(jaccard_score(y_true, y_pred, threshold=threshold))
    multi_thresh_jaccard = np.array(multi_thresh_jaccard)
    return multi_thresh_jaccard


def spearmanr_cc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Returns Spearman's correlation coefficient
    return scipy.stats.spearmanr(y_true, y_pred)[0]


def pearsonr_cc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Returns Pearson's r
    return scipy.stats.pearsonr(y_true, y_pred)[0]
