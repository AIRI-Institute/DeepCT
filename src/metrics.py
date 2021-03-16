import numpy as np
import sklearn.metrics as metrics

# Metric helpers which convert float values to binary (0/1).


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
