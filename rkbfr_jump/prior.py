"""
Classes that represent the priors of the RKHS model.
"""

import numpy as np
from eryn.prior import ProbDistContainer
from numba import njit, prange
from scipy.stats import norm, uniform


class jeffreys_prior:
    """Jeffrey's prior P(alpha0, sigma2) ∝ 1/sigma2. It is equivalent to a flat prior
    on (alpha0, log_sigma)"""

    def __init__(self, theta_space):
        self.ts = theta_space

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log P(alpha0, sigma2) under Jeffrey's prior.

        It checks the bounds on sigma2 and fills positions of invalid values with -np.inf.

        Parameters
        ----------
        x : np.ndarray (D1, ..., Dn, 2)
            Input array.

        Returns
        -------
        np.ndarray (D1, ..., Dn)
            Array of logpdf values.
        """
        res = np.full(x.shape[:-1], -np.inf)
        idx_valid = np.where(
            x[..., self.ts.idx_sigma2] > 0
        )  # identify samples where sigma2 > 0
        res[idx_valid] = -np.log(
            x[*idx_valid, self.ts.idx_sigma2]
        )  # log P(alpha0, sigma2) ∝ -log(sigma2)
        return res


class flat_prior:
    """Flat prior P(alpha0, log_sigma) ∝ 1"""

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log P(alpha0, sigma2) under flat prior.

        Parameters
        ----------
        x : np.ndarray (D1, ..., Dn, 2)
            Input array.

        Returns
        -------
        np.ndarray (D1, ..., Dn)
            Array of logpdf values.
        """
        return np.zeros_like(x.shape[:-1])  # log P(alpha0, sigma2) ∝ -log(1) = 0


def uniform_dist(a, b):
    """Wrapper around scipy.stats.uniform; similar to eryn.prior.uniform_dist, but without
    the error that it needs floating numbers as input to logpdf or else it fails."""
    return uniform(loc=a, scale=b - a)


class RKHSPriorLinear:
    """Simple prior where all parameters are independent"""

    def __init__(
        self,
        theta_space,
        sd_beta,
        lambda_p=None,
        min_dist_tau=1,
    ):
        # Indices in the coords array
        self.ts = theta_space

        # Prior on p
        self.lambda_p = lambda_p

        # Independent priors
        self.prior_beta = norm(0, sd_beta)
        self.prior_tau = uniform_dist(self.ts.grid.min(), self.ts.grid.max())
        self.prior_alpha0_sigma2 = (
            flat_prior() if self.ts.transform_sigma else jeffreys_prior(theta_space)
        )

        # Other information
        self.sd_beta = sd_beta
        self.min_dist_tau = min_dist_tau

        # Eryn-compatible prior container (would not be possible if our prior was more complex)
        self.container = {
            "components": ProbDistContainer(
                {
                    theta_space.idx_beta: self.prior_beta,
                    theta_space.idx_tau: self.prior_tau,
                }
            ),
            "common": ProbDistContainer(
                {
                    (
                        theta_space.idx_alpha0,
                        theta_space.idx_sigma2,
                    ): self.prior_alpha0_sigma2,
                }
            ),
        }

    def logpmf_rate_p(self, p):
        """Compute log(P(p)/P(p-1)) using the prior on p."""
        if self.lambda_p is None:  # uniform distribution
            return np.zeros_like(p)  # np.log(1)=0

        # P(p) = \lambda^p/Cp!, with C a constant
        return np.log(self.lambda_p / p)

    def logpdf_components(self, coords_components, inds_components=None):
        # Get current values of theta
        beta = coords_components[..., self.ts.idx_beta]
        tau = coords_components[..., self.ts.idx_tau]

        if inds_components is not None:
            beta = beta[inds_components]
            tau = tau[inds_components]

        # Compute logpdf for beta and tau
        lp_beta = self.prior_beta.logpdf(beta)
        lp_tau = self.prior_tau.logpdf(tau)

        return lp_beta + lp_tau

    def logpdf(self, coords, inds, supps=None, branch_supps=None):
        # Get current values of theta
        beta = coords["components"][..., self.ts.idx_beta]
        tau = coords["components"][..., self.ts.idx_tau]
        alpha0_sigma2 = coords["common"]

        # Compute logpdf for beta and tau
        lp_beta = self.prior_beta.logpdf(beta)  # (ntemps, nwalkers, nleaves_max)
        lp_tau = self.prior_tau.logpdf(tau)  # (ntemps, nwalkers, nleaves_max)

        # Compute logpdf for alpha0 and sigma2
        lp_alpha0_sigma2 = self.prior_alpha0_sigma2.logpdf(alpha0_sigma2)[
            ..., 0
        ]  # (ntemps, nwalkers)

        # Do not allow very close values of tau on the same branch
        idx_valid = check_valid_t(
            self.ts.get_idx_tau_grid(tau), inds["components"], self.min_dist_tau
        )
        lp_tau[~idx_valid] = -1e300  # -np.inf fails for MT computations

        # Turn off contribution of inactive leaves
        lp_beta[~inds["components"]] = 0.0
        lp_tau[~inds["components"]] = 0.0

        return lp_beta.sum(axis=-1) + lp_tau.sum(axis=-1) + lp_alpha0_sigma2

    def rvs(self, size, coords=None, inds=None):
        """arguments coords and inds are present for compatibility"""

        if isinstance(size, tuple) or isinstance(size, np.ndarray):
            size_tuple = tuple(size)
        else:
            size_tuple = (size,)

        out = np.zeros(size_tuple + (2,))

        # Only generate samples for the RJ moves (b and t)
        out[..., self.ts.idx_beta] = self.prior_beta.rvs(size=size)
        out[..., self.ts.idx_tau] = self.prior_tau.rvs(size=size)

        return out


@njit(cache=True, parallel=True, fastmath=True)
def check_valid_t(t_grid, inds_t=None, min_dist=1):
    ntemps, nwalkers, nleaves_max = t_grid.shape
    valid = np.ones((ntemps, nwalkers), dtype=np.bool_)

    # Need to create new variables so that numba is happy
    if inds_t is None:
        _inds_t = np.ones_like(t_grid, dtype=np.bool_)
    else:
        _inds_t = inds_t

    for i in prange(ntemps):
        for j in range(nwalkers):
            t_ij = t_grid[i, j]
            inds_t_ij = _inds_t[i, j]

            for k in range(nleaves_max):
                if not inds_t_ij[k]:
                    continue

                for k_next in range(nleaves_max):
                    if (
                        k_next != k
                        and inds_t_ij[k_next]
                        and np.abs(t_ij[k] - t_ij[k_next]) <= min_dist
                    ):
                        valid[i, j] = False
                        break

                if not valid[i, j]:
                    break

    return valid


@njit(cache=True, parallel=True, fastmath=True)
def generate_valid_t(size, grid, min_dist=1, shuffle=True, seed=None):
    if seed is None:
        seed = 0

    M, K, N = size
    n_grid = len(grid)
    t = np.zeros((M, K, N))
    max_range = n_grid // N

    for m in prange(M):
        np.random.seed(seed + m)
        for k in range(K):
            start = 0
            for n in range(N):
                end = (n + 1) * max_range
                value = np.random.randint(start, end)
                t[m, k, n] = grid[value]
                start = value + min_dist + 1

    if shuffle:
        for m in prange(M):
            np.random.seed(seed + m)
            for k in range(K):
                np.random.shuffle(t[m, k])

    return t