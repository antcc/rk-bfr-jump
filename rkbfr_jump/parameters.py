###
# Helper classes for parameter transformations
###

import numpy as np
from numba import njit


class ThetaVars:
    def __init__(self, names, idx):
        self.names = names
        self.idx_beta = idx[0]
        self.idx_tau = idx[1]
        self.idx_alpha0 = idx[2]
        self.idx_sigma2 = idx[3]


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