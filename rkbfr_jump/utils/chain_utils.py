###
# Helper functions to post-process the sampler chains
###

try:
    from IPython.display import display
except ImportError:
    display = print

import warnings
from copy import deepcopy

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import pdist
from scipy.stats import invgamma, norm

from ..parameters import LogSqrtTransform
from ..prior import check_valid_t, generate_valid_t


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


def fix_all_false_inds(inds):
    idx_all_false = np.where(
        np.sum(inds, axis=-1) == 0
    )  # Find branches with 0 active components
    inds[*idx_all_false, 0] = True  # Set the first component to True


def sample_initial_values(
    coords, theta_space, prior, y_std_orig, y_scaled_mean, y_mean_abs, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    size_components = coords["components"].shape[:-1]

    # sample initial values for components values (b) from prior
    coords["components"][..., theta_space.idx_beta] = prior.container["components"].rvs(
        size=size_components, keys=[theta_space.idx_beta]
    )[..., theta_space.idx_beta]

    # uniform initial values for component times (t), but constricted to be different
    coords["components"][..., theta_space.idx_tau] = generate_valid_t(
        size_components, theta_space.grid, prior.min_dist_tau, shuffle=True, seed=seed
    )
    assert np.all(
        check_valid_t(
            theta_space.get_idx_tau_grid(
                coords["components"][..., theta_space.idx_tau]
            ),
            min_dist=prior.min_dist_tau,
        )
    )

    # sample initial values for alpha0 from normal distribution
    coords["common"][:, :, 0, theta_space.idx_alpha0] = norm(
        0, 10 * np.abs(y_scaled_mean)
    ).rvs(size=size_components[:2])

    # sample initial values for sigma2 from inverse-gamma distribution
    sigma2_initial_values = invgamma(2, scale=y_mean_abs / (100 * y_std_orig**2)).rvs(
        size=size_components[:2]
    )
    coords["common"][:, :, 0, theta_space.idx_sigma2] = (
        LogSqrtTransform.forward(sigma2_initial_values)
        if theta_space.transform_sigma
        else sigma2_initial_values
    )


def print_sampling_information(
    full_chain_components,
    full_chain_common,
    theta_space,
    ensemble,
    idx_order,
    thin_by,
    display_notebook=False,
):
    components_last = full_chain_components[-1, 0, :]  # last sample of first walker
    common_last = full_chain_common[-1, 0, :]  # last sample of first walker
    print("* Last sample (T=0, W=0):")
    df_components = pd.DataFrame(
        {
            "$b$": components_last[:, theta_space.idx_beta],
            "$t$": components_last[:, theta_space.idx_tau],
        }
    )
    df_common = pd.DataFrame(
        {
            "$\\alpha_0$": [common_last[theta_space.idx_alpha0]],
            "$\\sigma^2$": [common_last[theta_space.idx_sigma2]],
        }
    )
    if display_notebook:
        display(df_components)
        display(df_common.style.hide(axis="index"))
    else:
        print(df_components)
        print(df_common)

    print("\n* Acceptance % (T=0)")
    print(
        f"[{ensemble.moves[0].moves[0].__class__.__name__}]",
        100 * ensemble.moves[0].acceptance_fraction_separate[0][0],
    )
    print(
        f"[{ensemble.moves[0].moves[1].__class__.__name__}]",
        100 * ensemble.moves[0].acceptance_fraction_separate[1][0],
    )
    print(
        f"[{ensemble.rj_moves[0].__class__.__name__}]",
        100 * ensemble.rj_acceptance_fraction[0],
    )
    print("\n* Temperature swaps accepted %:", 100 * ensemble.swap_acceptance_fraction)
    print("\n* Last values of a (parameter of the in-model moves):")
    print(f"[GroupMoveRKHS] a={ensemble.moves[0].moves[0].a:.2f}")
    print(f"[StretchMove] a={ensemble.moves[0].moves[1].a:.2f}")

    if full_chain_components.shape[2] > 1:
        print(
            f"\n* Chain ordered by: {'beta' if idx_order == theta_space.idx_beta else 'tau'}"
        )

    print("\n* ", end="")
    _ = ensemble.backend.get_gelman_rubin_convergence_diagnostic(thin=thin_by)


def pp_to_arviz_idata(pp, y_obs):
    """Convert posterior predictive arrays to InferenceData."""
    coords = {
        "draw": np.arange(pp.shape[0]),
        "chain": np.arange(pp.shape[1]),
        "prediction": np.arange(pp.shape[2]),
    }
    data_vars = {"y_star": (("draw", "chain", "prediction"), pp)}

    idata_pp = az.convert_to_inference_data(
        xr.Dataset(data_vars=data_vars, coords=coords),
        group="posterior_predictive",
    )

    idata_obs = az.convert_to_inference_data(
        xr.Dataset(data_vars={"y_obs": ("observation", y_obs)}, coords=coords),
        group="observed_data",
    )

    az.concat(idata_pp, idata_obs, inplace=True)

    return idata_pp