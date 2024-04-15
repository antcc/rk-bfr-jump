"""
Classes that represent the likelihood of the RKHS model given the data.
"""

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import cdist

from .parameters import LogSqrtTransform


class RKHSLikelihood:
    def __init__(self, theta_space, grid, X, y, transform_sigma=False):
        self.ts = theta_space
        self.grid = grid
        self.X_T = np.ascontiguousarray(X.T)
        self.y = y
        self.n = X.shape[0]
        self.transform_sigma = transform_sigma

    # METHOD 1: Sequential computation for future parallelization with pool argument in Eryn
    def evaluate_sequential(self, theta):
        """Computes the log-likelihood log P(Y1,...,Y_n|theta, X_1, ..., X_n) for a single walker.

        - theta is a list [theta_components, theta_common] that represent the parameters for a single walker.
            * theta_components is an array (p, 2) of the parameters (b_j, t_j) for each component (leaf), where 1<=p<=p_max.
              In other words, on each call to this function the shape varies according to how many leaves are active.
            * theta_common is an array (1, 2) of the parameters alpha_0 and sigma^2, which are common for all models (leaves).
        - t is the discretized grid of shape (ngrid,)
        - X, y are the data, of shape (nfunc, ngrid) and (nfunc,) respectively.

        ========
        Return: the log-likelihood for the walker, i.e., log P(Y|theta, X).

        ========
        Note: Eryn already checks that the theta input values are valid (i.e. within the bounds of the prior).
        """
        theta_components, theta_common = theta

        # Avoid calling functions for getting the individual variables; in this part
        # of the code efficiency is critical
        beta = np.ascontiguousarray(theta_components[:, self.ts.idx_beta])
        tau = np.ascontiguousarray(theta_components[:, self.ts.idx_tau])
        alpha0 = theta_common[0][self.ts.idx_alpha0]

        if self.transform_sigma:
            log_sigma = theta_common[0][self.ts.idx_sigma2]
            sigma2 = LogSqrtTransform.backward(log_sigma)
        else:
            sigma2 = theta_common[0][self.ts.idx_sigma2]
            log_sigma = LogSqrtTransform.forward(sigma2)

        # Compute the indices of the grid corresponding to the parameter tau
        idx_tau_grid = cdist(tau[:, None], self.grid[:, None], "cityblock").argmin(
            axis=-1
        )  # == np.abs(self.grid - tau[:, None]).argmin(axis=-1)

        diff = self.y - alpha0 - self.X_T[idx_tau_grid].T @ beta
        ll = -self.n * log_sigma - 0.5 * (diff @ diff) / sigma2

        return ll

    # Method 2: Vectorized computation with parallel execution of inner loop
    def evaluate_vectorized(self, theta, groups):
        """Computes the log-likelihood log P(Y_1,...,Y_n|[Theta], X_1, ..., X_n) for all walkers and temps.

        - theta is a list [theta_components, theta_common] that represents the parameters for _all_ walkers (across
          all temperatures).
            * theta_components is an array (N, 2) of the parameters (b_j, t_j) for all walkers and temps. N is the total
              number of active parameters in the RJ branch, i.e., N=sum(inds["components"]). Informally, we can
              compute N as nwalkers*ntemps*(active leaves on each (nwalker, ntemp)). In these arrays we have no
              information about which walker the parameters belong to; we just know their values.
            * theta_common is an array (M, 2) of the parameters alpha_0 and sigma^2 for all walkers and temps. M is
              the total number of such parameters (which is fixed), i.e., M=nwalkers*ntemps. In general, N >= M.
        - groups is a list [groups_components, groups_common] that represents the correspondence between parameters
          in the flattened array theta_* and the specific walker they belong to. Each group_* array contains integers,
          and each position within them represents the walker #id of the corresponding parameter in the same position
          on the theta_* array of parameters. The range of possible values is {1, 2, ..., M}. Eryn internally converts
          the inds array to groups information (eryn.utils.groups_from_inds), so that only active parameters are
          considered for the likelihood computation.
            * groups_components is an array (N,).
            * groups_common is an array (M,), and since the corresponding parameters are fixed and nleaves_min=1 in this
              branch, groups_common=np.arange(M). This parameter can be safely ignored.
        - t is the grid of shape (ngrid,)
        - X, y are the data, of shape (nfunc, ngrid) and (nfunc,) respectively.

        ========
        Example: suppose ntemps=2, nwalkers=2, and on each of the 4 total walkers we have 1, 2, 3 and 2 components active,
                 respectively. Then, N=1+2+3+2=8, while M=2*2=4. We would have groups_components = [0,1,1,2,2,2,3,3]
                 and groups_common = [0,1,2,3]. To compute the likelihood for walker i, we have to collect all theta_components
                 corresponding to this walker #id (whose indices are given by np.where(groups_components == i)), as well as
                 the corresponding alpha_0 and sigma^2 (theta_common[i]), and perform the computation per the model.

        ========
        Return: array (M,) in which each position is the log-likelihood for a specific walker, i.e.,
                log P(Y|Theta_i, X) (i=1,...,M).

        ========
        Note: Eryn already checks that the theta input values are valid (i.e. within the bounds of the prior).
        """
        theta_components, theta_common = theta
        groups_components, _ = groups
        unique_indices = np.unique(groups_components)

        beta = np.ascontiguousarray(theta_components[:, self.ts.idx_beta])
        tau = np.ascontiguousarray(
            theta_components[:, self.ts.idx_tau]
        )  # get tau as a column
        alpha0 = np.ascontiguousarray(theta_common[:, self.ts.idx_alpha0])

        if self.transform_sigma:
            log_sigma = np.ascontiguousarray(theta_common[:, self.ts.idx_sigma2])
            sigma2 = LogSqrtTransform.backward(log_sigma)
        else:
            sigma2 = np.ascontiguousarray(theta_common[:, self.ts.idx_sigma2])
            log_sigma = LogSqrtTransform.forward(sigma2)

        idx_tau_grid = cdist(tau[:, None], self.grid[:, None], "cityblock").argmin(
            axis=-1
        )  # == np.abs(self.grid - tau[:, None]).argmin(axis=-1)

        ll = self._compute_ll_parallel(
            groups_components,
            unique_indices,
            beta,
            idx_tau_grid,
            alpha0,
            sigma2,
            log_sigma,
            self.X_T,
            self.y,
            self.n,
        )

        return ll

    # Actual parallel computation of log_likelihood for all groups
    @staticmethod
    @njit(
        parallel=True,
        fastmath=True,
        cache=True,
    )
    def _compute_ll_parallel(
        groups_components: np.ndarray,
        unique_indices: np.ndarray,
        beta: np.ndarray,
        idx_tau_grid: np.ndarray,
        alpha0: np.ndarray,
        sigma2: np.ndarray,
        log_sigma: np.ndarray,
        X_T: np.ndarray,
        y: np.ndarray,
        n: np.ndarray,
    ) -> np.ndarray:
        # Preallocate memory
        ll = np.empty_like(unique_indices, dtype=np.float64)
        X_tau_masked = np.empty_like(X_T[0])
        beta_masked = np.empty_like(beta[0])
        diff = np.empty_like(y)
        n_unique = len(unique_indices)

        for i in prange(n_unique):
            idx = unique_indices[i]
            mask = np.where(groups_components == idx)
            X_tau_masked = np.ascontiguousarray(X_T[idx_tau_grid[mask]])
            beta_masked = beta[mask]
            alpha0_masked = alpha0[idx]
            sigma2_masked = sigma2[idx]
            log_sigma_masked = log_sigma[idx]

            diff = y - alpha0_masked - np.dot(X_tau_masked.T, beta_masked)
            ll[i] = -n * log_sigma_masked - 0.5 * np.dot(diff, diff) / sigma2_masked

        return ll