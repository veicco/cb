import numpy as np
from typing import Protocol, Union


Action = Union[int, float]
Cost = Union[int, float]
Prob = float


class RegressionModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        ...

    def predict(self, X: np.ndarray):
        ...
