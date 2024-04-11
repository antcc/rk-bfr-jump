"""
Utility functions.
"""

import os
import warnings

import arviz as az
import numpy as np
from numba import njit
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

###
# Custom context managers for handling warnings
###


class IgnoreWarnings:
    key = "PYTHONWARNINGS"

    def __enter__(self):
        if self.key in os.environ:
            self.state = os.environ[self.key]
        else:
            self.state = "default"

        os.environ[self.key] = "ignore"
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="One or more of the test scores are non-finite:",
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        warnings.resetwarnings()
        os.environ[self.key] = self.state


###
# Helper classes for parameter transformations
###


class LogSqrtTransform:
    """Transformation x --> log(sqrt(x))"""

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def forward(x):
        return np.log(np.sqrt(x))

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def backward(y):
        return np.exp(y) ** 2


class LogTransform:
    """Transformation x --> log(x)"""

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def forward(x):
        return np.log(x)

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def backward(y):
        return np.exp(y)


class SqrtTransform:
    """Transformation x --> sqrt(x)"""

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def forward(x):
        return np.sqrt(x)

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def backward(y):
        return y**2


###
# Misc.
###


def mode_kde(arr):
    x, density = az.kde(arr)
    return x[np.argmax(density)]


def color_reference_methods(x, df):
    return [
        "color: orange; hide: axis" if val in list(df["Estimator"]) else "" for val in x
    ]