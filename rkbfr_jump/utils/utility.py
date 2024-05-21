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
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Variables are collinear",
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        warnings.resetwarnings()
        os.environ[self.key] = self.state


# Compute the "mode" of a continuous kde
def mode_kde(arr):
    if len(arr) == 1:
        return arr[0]

    x, density = az.kde(arr)
    return x[np.argmax(density)]


# Function to color a list of estimators present in another DataFrame
def color_reference_methods(x, df):
    format_str = []
    for name in x:
        if name == "flin":
            format_str.append("color: crimson; hide: axis")
        elif name in list(df["Estimator"]):
            format_str.append("color: orange; hide: axis")
        else:
            format_str.append("")

    return format_str