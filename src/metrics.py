import numpy as np
import scipy.stats
import sklearn.metrics as metrics

# Metric helpers which convert float values to binary (0/1).
# Regression metrics helper functions.


def _to_binary(x: np.ndarray, threshold=0.5) -> np.ndarray:
    return np.where(x > threshold, 1, 0).astype(int)


def accuracy_score(x: np.ndarray, y: np.ndarray) -> float:
    binary_x = _to_binary(x)
    binary_y = _to_binary(y)
    return metrics.accuracy_score(binary_x, binary_y)


def f1_score(x: np.ndarray, y: np.ndarray) -> float:
    binary_x = _to_binary(x)
    binary_y = _to_binary(y)
    return metrics.f1_score(binary_x, binary_y)


def precision_score(x: np.ndarray, y: np.ndarray) -> float:
    binary_x = _to_binary(x)
    binary_y = _to_binary(y)
    return metrics.precision_score(binary_x, binary_y)


def recall_score(x: np.ndarray, y: np.ndarray) -> float:
    binary_x = _to_binary(x)
    binary_y = _to_binary(y)
    return metrics.recall_score(binary_x, binary_y)


def spearmanr_cc(x: np.ndarray, y: np.ndarray) -> float:
    # Returns Spearman's correlation coefficient
    return scipy.stats.spearmanr(x, y)[0]


def pearsonr_cc(x: np.ndarray, y: np.ndarray) -> float:
    # Returns Pearson's r
    return scipy.stats.pearsonr(x, y)[0]
