import numpy as np

from .utils.utility import IgnoreWarnings


def predict_pp(
    chain_components, chain_common, theta_space, X_test, aggregate_pp, noise=False
):
    # Replace NaN with 0.0 to "turn off" the corresponding coefficients of beta
    chain_components = np.nan_to_num(chain_components, nan=0.0)

    beta = chain_components[
        ..., theta_space.idx_beta, None
    ]  # (nsteps, nwalkers, nleaves_max, 1)
    tau = chain_components[..., theta_space.idx_tau, None]
    alpha0 = chain_common[..., theta_space.idx_alpha0, None]

    idx_tau_grid = theta_space.get_idx_tau_grid(
        tau, add_dimension=False
    )  # (nsteps, nwalkers, nleaves_max)
    X_idx = np.moveaxis(
        X_test[:, idx_tau_grid], 0, -2
    )  # X_idx must be (nsteps, nwalkers, nfuncs, nleaves_max)
    y_pred_all = (
        alpha0 + (X_idx @ beta).squeeze()
    )  # result is (nsteps, nwalkers, nfuncs)

    if noise:
        sigma2 = chain_common[..., theta_space.idx_sigma2, None]
        y_pred_all += np.sqrt(sigma2) * np.random.standard_normal(y_pred_all.shape)

    # Summary of pp
    y_pred_all = y_pred_all.reshape(
        -1, len(X_test)
    )  # bring together all (nsteps, nwalkers)
    y_pred_aggregate = aggregate_pp(y_pred_all, axis=0)

    return y_pred_aggregate


def predict_weighted(
    chain_components,
    chain_common,
    nleaves,
    predict_fn,
    kwargs_predict_fn,
    add_p_kwarg=False,
):
    nsteps, nwalkers, nleaves_max = chain_components.shape[:3]
    nsamples_total = nwalkers * nsteps

    preds_by_p = []
    weights_by_p = []
    for p in range(1, nleaves_max + 1):
        idx_p = np.where(nleaves == p)
        nsamples_by_p = len(idx_p[0])

        if nsamples_by_p == 0:
            continue

        chain_components_p = chain_components[*idx_p]
        chain_common_p = chain_common[*idx_p]

        if add_p_kwarg:
            kwargs_predict_fn["p"] = p

        preds_by_p.append(
            predict_fn(chain_components_p, chain_common_p, **kwargs_predict_fn)
        )
        weights_by_p.append(nsamples_by_p / nsamples_total)

    preds = np.average(preds_by_p, weights=weights_by_p, axis=0)

    return preds


def predict_weighted_pp(
    chain_components,
    chain_common,
    nleaves,
    theta_space,
    X_test,
    aggregate_pp,
    noise=False,
):
    kwargs_predict_pp = {
        "theta_space": theta_space,
        "X_test": X_test,
        "aggregate_pp": aggregate_pp,
        "noise": noise,
    }
    preds = predict_weighted(
        chain_components, chain_common, nleaves, predict_pp, kwargs_predict_pp
    )

    return preds


def predict_map_pp(
    chain_components,
    chain_common,
    nleaves,
    map_p,
    theta_space,
    X_test,
    aggregate_pp,
    noise=False,
):
    chain_components_p = chain_components[nleaves == map_p]
    chain_common_p = chain_common[nleaves == map_p]
    preds = predict_pp(
        chain_components_p,
        chain_common_p,
        theta_space,
        X_test,
        aggregate_pp,
        noise=noise,
    )

    return preds


def predict_pe(
    chain_components_p, chain_common_p, p, theta_space, X_test, summary_statistic
):  # predict point_estimate
    """chain_components_p is of shape (*, nleaves_max, 2), i.e.,
    already flattened on the (nsteps, nwalkers) dimension."""
    chain_components_without_nan = chain_components_p[
        :, :p, :
    ]  # remove NaN values before summarizing
    chain_components_summary = summary_statistic(chain_components_without_nan, axis=0)
    chain_common_summary = summary_statistic(chain_common_p, axis=0)

    beta = chain_components_summary[:, theta_space.idx_beta]  # beta is a vector (p,)
    tau = chain_components_summary[:, theta_space.idx_tau]  # tau is a vector (p)
    alpha0 = chain_common_summary[theta_space.idx_alpha0]  # alpha0 is scalar

    idx_tau_grid = theta_space.get_idx_tau_grid(tau)  # (p,)
    X_idx = X_test[:, idx_tau_grid]
    y_pred_all = alpha0 + X_idx @ beta  # result is (nfunc,)

    return y_pred_all


def predict_weighted_summary(
    chain_components,
    chain_common,
    nleaves,
    theta_space,
    X_test,
    summary_statistic,
):
    kwargs_predict_pe = {
        "theta_space": theta_space,
        "X_test": X_test,
        "summary_statistic": summary_statistic,
    }
    preds = predict_weighted(
        chain_components,
        chain_common,
        nleaves,
        predict_pe,
        kwargs_predict_pe,
        add_p_kwarg=True,
    )

    return preds


def predict_map_summary(
    chain_components,
    chain_common,
    nleaves,
    map_p,
    theta_space,
    X_test,
    summary_statistic,
):
    chain_components_p = chain_components[nleaves == map_p]
    chain_common_p = chain_common[nleaves == map_p]
    preds = predict_pe(
        chain_components_p,
        chain_common_p,
        map_p,
        theta_space,
        X_test,
        summary_statistic,
    )

    return preds


def predict_vs(
    chain_components_p,
    chain_common_p,
    p,
    theta_space,
    X_train,
    y_train,
    X_test,
    summary_statistic,
    reg,
):  # predict variable_selection
    """chain_components_p is of shape (*, nleaves_max, 2), i.e.,
    already flattened on the (nsteps, nwalkers) dimension."""
    chain_components_without_nan = chain_components_p[
        :, :p, :
    ]  # remove NaN values before summarizing
    chain_components_summary = summary_statistic(chain_components_without_nan, axis=0)
    tau = chain_components_summary[:, theta_space.idx_tau]  # tau is a vector (p,)
    idx_tau_grid = np.sort(
        np.unique(theta_space.get_idx_tau_grid(tau))
    )  # idx is at most (p,), sorted for convenience

    with IgnoreWarnings():
        reg.fit(X_train[:, idx_tau_grid], y_train)
    y_pred = reg.predict(X_test[:, idx_tau_grid])

    return y_pred


def predict_weighted_variable_selection(
    chain_components,
    chain_common,
    nleaves,
    theta_space,
    X_train,
    y_train,
    X_test,
    summary_statistic,
    reg,
):
    kwargs_predict_vs = {
        "theta_space": theta_space,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "summary_statistic": summary_statistic,
        "reg": reg,
    }
    preds = predict_weighted(
        chain_components,
        chain_common,
        nleaves,
        predict_vs,
        kwargs_predict_vs,
        add_p_kwarg=True,
    )

    return preds


def predict_map_variable_selection(
    chain_components,
    chain_common,
    nleaves,
    map_p,
    theta_space,
    X_train,
    y_train,
    X_test,
    summary_statistic,
    reg,
):
    chain_components_p = chain_components[nleaves == map_p]
    chain_common_p = chain_common[nleaves == map_p]
    preds = predict_vs(
        chain_components_p,
        chain_common_p,
        map_p,
        theta_space,
        X_train,
        y_train,
        X_test,
        summary_statistic,
        reg,
    )

    return preds