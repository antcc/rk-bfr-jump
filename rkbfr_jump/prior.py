"""
Classes that represent the priors of the RKHS model.
"""

import numpy as np
from scipy.stats import norm, uniform


class jeffreys_prior:
    """Jeffrey's prior P(alpha0, sigma2) ∝ 1/sigma2. It is equivalent to a flat prior
    on (alpha0, log_sigma)"""

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
        idx_valid = np.where(x[..., 1] > 0)  # identify samples where sigma2 > 0
        res[idx_valid] = -np.log(
            x[*idx_valid, 1]
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


class RKHSPriorSimple:
    """The prior on p (the number of components) is assumed to be uniform in [1, n_leaves_max]"""

    def __init__(self, grid, sd_beta, transform_sigma=False):
        # Indices in the coords array
        self.idx_beta = 0
        self.idx_tau = 1
        self.idx_alpha0 = 0
        self.idx_sigma2 = 1

        # Independent priors
        self.prior_beta = norm(0, sd_beta)
        self.prior_tau = uniform_dist(grid.min(), grid.max())
        self.prior_alpha0_sigma2 = flat_prior() if transform_sigma else jeffreys_prior()

        # Other information
        self.grid = grid
        self.sd_beta = sd_beta
        self.transform_sigma = transform_sigma

    def logpdf_components(self, coords_components, inds_components=None):
        # Get current values of theta
        beta = coords_components[..., self.idx_beta]
        tau = coords_components[..., self.idx_tau]

        if inds_components is not None:
            beta = beta[inds_components]
            tau = tau[inds_components]

        # Compute logpdf for beta and tau
        lp_beta = self.prior_beta.logpdf(beta)
        lp_tau = self.prior_tau.logpdf(tau)

        return lp_beta + lp_tau

    def logpdf(self, coords, inds):
        # Get current values of theta
        beta = coords["components"][..., self.idx_beta]
        tau = coords["components"][..., self.idx_tau]
        alpha0_sigma2 = coords["common"]

        # Compute logpdf for beta and tau
        lp_beta = self.prior_beta.logpdf(beta)  # (ntemps, nwalkers, nleaves_max)
        lp_tau = self.prior_tau.logpdf(tau)  # (ntemps, nwalkers, nleaves_max)

        # Compute logpdf for alpha0 and sigma2
        lp_alpha0_sigma2 = self.prior_alpha0_sigma2.logpdf(alpha0_sigma2)[
            ..., 0
        ]  # (ntemps, nwalkers)

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
        out[..., self.idx_beta] = self.prior_beta.rvs(size=size)
        out[..., self.idx_tau] = self.prior_tau.rvs(size=size)

        return out