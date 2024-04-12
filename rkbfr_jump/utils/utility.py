"""
Utility functions.
"""

import os
import warnings

import arviz as az
import numpy as np
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning


# Custom context managers for handling warnings
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


# Compute the "mode" of a continuous kde
def mode_kde(arr):
    x, density = az.kde(arr)
    return x[np.argmax(density)]


# Function to color a list of estimators present in another DataFrame
def color_reference_methods(x, df):
    return [
        "color: orange; hide: axis" if val in list(df["Estimator"]) else "" for val in x
    ]