###
# Helper classes for parameters
###

import numpy as np
from numba import njit


class ThetaSpace:
    def __init__(self, grid, names, idx, transform_sigma):
        self.grid = grid
        self.names = names
        self.idx_beta = idx[0]
        self.idx_tau = idx[1]
        self.idx_alpha0 = idx[2]
        self.idx_sigma2 = idx[3]
        self.transform_sigma = transform_sigma

    def get_idx_tau_grid(self, tau, add_dimension=True):
        # An extra dimension at the end is needed to compare with every
        # value of the 1D grid. Users must ensure to do it outside
        # if add_dimension is False
        if add_dimension:
            tau = tau[..., None]

        idx_tau_grid = np.abs(self.grid - tau).argmin(axis=-1)
        return idx_tau_grid


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