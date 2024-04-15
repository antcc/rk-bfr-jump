###
# Helper functions to post-process the sampler chains
###

import warnings
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm

from ..parameters import LogSqrtTransform


def get_full_chain_at_T(
    sampler,
    theta_space,
    X_std_orig,
    Y_std_orig,
    T=0,
    discard=0,
    transform_sigma=False,
    relabel_strategy="auto",
):
    # Get chain from sampler
    chain = deepcopy(sampler.get_chain(discard=discard))

    # Undo change sigma2 --> log_sigma if needed
    if transform_sigma:
        chain["common"][:, T, ..., theta_space.idx_sigma2] = LogSqrtTransform.backward(
            chain["common"][:, T, ..., theta_space.idx_sigma2]
        )

    chain_components = chain["components"][:, T, ...]
    chain_common = chain["common"][:, T, ...].squeeze()

    # Revert components back to original scale
    tau_old = chain_components[..., theta_space.idx_tau]
    idx_tau_grid = theta_space.get_idx_tau_grid(tau_old)

    chain_components[..., theta_space.idx_beta] *= (
        Y_std_orig / X_std_orig[idx_tau_grid]
    )  # Revert beta
    chain_common[..., theta_space.idx_alpha0] *= Y_std_orig  # Revert alpha0
    chain_common[..., theta_space.idx_sigma2] *= Y_std_orig**2  # Revert sigma2

    # Get indices
    inds = sampler.get_inds(discard=discard).copy()
    inds_components = inds["components"][:, T, ...]
    inds_common = inds["common"][:, T, ...]
    idx_order = None

    # Order components
    if sampler.nleaves_max["components"] > 1:
        if relabel_strategy == "auto":  # Relabeling algorithm of Simola et al. (2021)
            beta = chain_components[..., theta_space.idx_beta]
            tau = chain_components[..., theta_space.idx_tau]
            beta_flat = np.sort(
                beta.reshape(-1, sampler.nleaves_max["components"]),
                axis=-1,
            )
            tau_flat = np.sort(
                tau.reshape(-1, sampler.nleaves_max["components"]),
                axis=-1,
            )

            # Rescale parameters to common units
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message="Mean of empty slice"
                )

                beta_scale = np.nanmean(
                    norm.cdf(
                        beta_flat, loc=np.nanmean(beta_flat), scale=np.nanstd(beta_flat)
                    ),
                    axis=0,
                )
                tau_scale = np.nanmean(
                    norm.cdf(
                        tau_flat, loc=np.nanmean(tau_flat), scale=np.nanstd(tau_flat)
                    ),
                    axis=0,
                )

            # Look for the maximum pairwise distance
            pdist_beta_max = np.max(pdist(beta_scale.reshape(-1, 1)))
            pdist_tau_max = np.max(pdist(tau_scale.reshape(-1, 1)))
            idx_order = (
                theta_space.idx_beta
                if pdist_beta_max > pdist_tau_max
                else theta_space.idx_tau
            )

        else:  # Manual relabeling
            idx_order = (
                theta_space.idx_beta
                if relabel_strategy == "beta"
                else theta_space.idx_tau
            )

        # Order the last dimension based on b or t, maintaining shape and the correspondence b_i <--> t_i
        sorted_indices = np.argsort(chain_components[..., idx_order], axis=-1)
        chain_components = np.take_along_axis(
            chain_components, sorted_indices[..., None], axis=-2
        )

        # Order indices (NaN's go at the end on each branch)
        inds_components = np.take_along_axis(inds_components, sorted_indices, axis=-1)

    return chain_components, chain_common, inds_components, inds_common, idx_order


def get_flat_chain_components(coords, theta_space, ndim):
    """Simple utility function to extract the flat chains for all the parameters"""
    coords_T_beta = coords[..., theta_space.idx_beta].flatten()
    coords_T_tau = coords[..., theta_space.idx_tau].flatten()
    valid_idx = ~np.isnan(coords_T_beta)
    samples_flat = np.zeros((np.sum(valid_idx), ndim))
    samples_flat[:, theta_space.idx_beta] = coords_T_beta[valid_idx]
    samples_flat[:, theta_space.idx_tau] = coords_T_tau[valid_idx]

    return samples_flat
