import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from arviz.labels import MapLabeller
from matplotlib.ticker import MaxNLocator
from scipy.stats import mode as mode_discrete

from ..prediction import predict_pp
from .chain_utils import pp_to_arviz_idata


def plot_dataset_regression(
    X, y, grid, plot_means=True, n_samples=None, figsize=(9, 4)
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    n, N = X.shape

    if n_samples is None:
        n_samples = n

    axs[0].set_title(r"Functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    axs[0].plot(grid, X.T[:, :n_samples], alpha=0.8)

    axs[1].set_yticks([])
    axs[1].set_title(r"Scalar values $Y_i$")
    az.plot_dist(y, ax=axs[1])
    axs[1].plot(y, np.zeros_like(y), "|", color="k", alpha=0.5)

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0), linewidth=3, color="k", label="Sample mean"
        )
        axs[0].legend(fontsize=10)

        axs[1].axvline(
            y.mean(), ls="--", lw=1.5, color="r", alpha=0.5, label="Sample mean"
        )
        axs[1].legend(fontsize=10)


def plot_dataset_classification(
    X, y, grid, plot_means=True, n_samples=None, figsize=(9, 4), ax=None
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    n, N = X.shape
    n_samples_0 = (y == 0).sum()
    n_samples_1 = (y == 1).sum()

    if n_samples is None:
        plot_n0 = n_samples_0
        plot_n1 = n_samples_1
    else:
        plot_n0 = int(n_samples_0 * n_samples / len(y))
        plot_n1 = int(n_samples_1 * n_samples / len(y))

    axs[0].set_title(r"Labeled functional regressors $X_i(t)$")
    axs[0].set_xlabel(r"$t$")
    if plot_n0 > 0:
        axs[0].plot(
            grid,
            X.T[:, y == 0][:, :plot_n0],
            alpha=0.5,
            color="blue",
            label=["Class 0"] + [""] * (plot_n0 - 1),
        )
    if plot_n1 > 0:
        axs[0].plot(
            grid,
            X.T[:, y == 1][:, :plot_n1],
            alpha=0.5,
            color="red",
            label=["Class 1"] + [""] * (plot_n1 - 1),
        )

    if plot_means:
        axs[0].plot(
            grid, np.mean(X, axis=0), linewidth=3, color="k", label="Sample mean"
        )

    axs[0].legend(fontsize=10)

    axs[1].set_title("Class distribution")
    axs[1].set_xlabel("Class")
    axs[1].set_xticks([0, 1])
    counts = [n_samples_0, n_samples_1]
    freq = counts / np.sum(counts)
    if counts[0] > 0:
        axs[1].bar(0, freq[0], color="blue", label="Class 0", width=0.3)
    if counts[1] > 0:
        axs[1].bar(1, freq[1], color="red", label="Class 1", width=0.3)
    axs[1].legend()


def plot_mode(samples, ax):
    arr, density = az.kde(samples)
    idx_mode = np.argmax(density)
    mode = arr[idx_mode]
    ax.text(
        mode,
        ax.get_ylim()[1] * 0.8,
        f"mode={mode:.3f}",
        horizontalalignment="center",
        fontsize=8,
    )
    ax.scatter(mode, density[idx_mode], color="red", s=10)


def plot_reference_values(
    values,
    ax,
    vertical=True,
    label="Train value",
    color="blue",
    legend=True,
    alpha=0.5,
    legend_kwargs={},
):
    handles = []
    values = np.atleast_1d(values)
    for i, value in enumerate(values):
        if vertical:
            h = ax.axvline(
                x=value,
                color=color,
                linestyle="--",
                alpha=alpha,
                label=label,
            )
        else:
            h = ax.axhline(
                y=value, color=color, linestyle="--", alpha=alpha, label=label
            )
        if i == 0:
            handles += [h]

    if legend:
        ax.legend(handles=handles, **legend_kwargs)

    return handles


def plot_trace(
    chain,
    var_name="x",
    true_value=None,
    train_value=None,
    mode=False,
    color="blue",
    az_kwargs={},
):
    # Get chain (nwalkers, nsamples)
    chain = np.atleast_2d(chain)
    chain = chain.swapaxes(0, 1)  # for compatibility with arviz

    # Plot marginal posterior and trace
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Your data appears to have a single value or no finite values",
        )
        ax = az.plot_trace(
            chain,
            combined=True,
            compact=False,
            chain_prop={"linestyle": ("solid", "dotted", "dashed", "dashdot")},
            labeller=MapLabeller({"x": var_name}),
            plot_kwargs={"color": color},
            trace_kwargs={"color": color, "alpha": 0.2},
            fill_kwargs={"alpha": 0.2},
            **az_kwargs,
        )
    ax[0, 0].grid(alpha=0.3, linestyle="--")

    # Plot reference values
    handles = []
    if true_value is not None:
        true_value = np.atleast_1d(true_value)
        handles += plot_reference_values(
            true_value, ax[0, 0], label="True value", legend=False, color="k"
        )
        plot_reference_values(
            true_value, ax[0, 1], vertical=False, legend=False, color="k"
        )

    if train_value is not None:
        train_value = np.atleast_1d(train_value)
        handles += plot_reference_values(
            train_value, ax[0, 0], label="Train value", legend=False, color=color
        )
        plot_reference_values(
            train_value, ax[0, 1], vertical=False, legend=False, color=color
        )

    if mode:
        plot_mode(chain, ax[0, 0])

    if true_value is not None or train_value is not None:
        ax[0, 0].legend(handles=handles, fontsize=9)


# Define a custom formatter function
def custom_format_xaxis(x, pos):
    format_str = f"{x:.3f}"
    format_str = format_str.rstrip("0").rstrip(
        "."
    )  # Remove trailing zeroes and decimal point
    return format_str.center(7)


def plot_trace_p(nleaves, nleaves_max, color="red", p_true=None):
    axs = az.plot_trace(
        nleaves.T,
        combined=True,
        compact=False,
        chain_prop={"linestyle": ("solid", "dotted", "dashed", "dashdot")},
        labeller=MapLabeller({"x": "p"}),
        trace_kwargs={"color": color, "alpha": 0.2},
        hist_kwargs={
            "color": color,
            "lw": 2,
            "alpha": 0.5,
            "histtype": "step",
            "hatch": "/////",
        },
    )
    axs[0, 1].set_yticks(np.arange(nleaves_max) + 1)

    if p_true is not None:
        axs[0, 0].axvline(
            x=p_true, color=color, linestyle="--", alpha=0.5, label="True value"
        )
        axs[0, 0].legend(fontsize=9)


def plot_tempered_posterior_p(ntemps, nleaves_all_T, nleaves_max, colors):
    fig, ax = plt.subplots(1, ntemps, figsize=(4 * ntemps, 3))
    bins = np.arange(1, nleaves_max + 2) - 0.5

    for temp, ax_t in enumerate(ax):
        color = colors(temp / ntemps)
        ax_t.set_title(f"T = {temp+1}")
        ax_t.set_xlabel("p")
        ax_t.set_xticks(np.arange(nleaves_max + 1))
        ax_t.hist(
            nleaves_all_T[:, temp].flatten(),
            density=True,
            bins=bins,
            color=color,
            alpha=0.8,
        )


def triangular_plot_components(
    chain_components,
    nleaves,
    idx_samples,
    var_name,
    cmap,
    color_p="red",
    figsize=(10, 7),
):
    fig = plt.figure(figsize=figsize)
    if var_name == "t":
        var_name_title = "$\\boldsymbol{\\tau}$"
    else:
        var_name_title = var_name
    fig.suptitle(rf"Posterior distribution of {var_name_title}", fontweight="semibold")

    # Get effective range of p
    min_p, max_p = np.min(nleaves), np.max(nleaves)

    # Row and column names
    row_names = [f"p={p}" for p in np.arange(min_p, max_p + 1)]
    column_names = [f"{var_name}{p+1}" for p in np.arange(max_p + 1)]

    # Colormap
    colors = cmap(np.linspace(0.75, 0, max_p))

    # Record MAP value of p
    map_p = mode_discrete(nleaves, axis=None).mode

    # Loop through each subplot
    for i in range(min_p, max_p + 1):
        chain_components_p = chain_components[nleaves == i]

        for j in range(i):
            ax_plot = fig.add_subplot(max_p, max_p, (i - min_p) * max_p + j + 1)

            # Catch warnings and delete the plot if the warning is raised
            with warnings.catch_warnings():
                message = "Your data appears to have a single value or no finite values"
                warnings.filterwarnings("error", category=UserWarning, message=message)
                try:
                    # Plot data
                    az.plot_dist(
                        chain_components_p[:, j, idx_samples],
                        ax=ax_plot,
                        color=colors[j],
                        fill_kwargs={"alpha": 0.2},
                        plot_kwargs={"linewidth": 1},
                    )
                except UserWarning as e:
                    if message in str(e):
                        # fig.delaxes(ax_plot)
                        ax_plot.axvline(
                            chain_components_p[0, j, idx_samples],
                            color=colors[j],
                            linewidth=1,
                        )

            ax_plot.set_yticklabels([])
            ax_plot.tick_params(axis="x", labelsize=8)
            ax_plot.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_plot.xaxis.set_major_formatter(custom_format_xaxis)

            if j == 0:
                ax_plot.set_ylabel(
                    row_names[i - min_p],
                    rotation=0,
                    ha="right",
                    verticalalignment="center",
                    fontsize=12,
                    color=color_p if i == map_p else "k",
                )
                ax_plot.yaxis.set_label_coords(-0.2, 0.5)

            if i == max_p:
                ax_plot.set_xlabel(
                    column_names[j],
                    fontsize=12,
                    labelpad=15,
                )


def posterior_plot_common(
    chain_common,
    nleaves,
    idx_samples,
    colors,
    color_p="red",
    figsize=(7, 7),
    plot_sigma2=True,
):
    fig = plt.figure(figsize=figsize)

    if plot_sigma2:
        fig.suptitle(
            r"Posterior distribution of $\boldsymbol{\alpha}_0$ and $\boldsymbol{\sigma}^2$",
            fontweight="semibold",
        )
    else:
        fig.suptitle(
            r"Posterior distribution of $\boldsymbol{\alpha}_0$",
            fontweight="semibold",
            horizontalalignment="right",
        )

    # Get effective range of p
    min_p, max_p = np.min(nleaves), np.max(nleaves)

    # Row and column names
    row_names = [f"p={p}" for p in np.arange(min_p, max_p + 1)]
    column_names = [r"$\alpha_0$", r"$\sigma^2$"] if plot_sigma2 else [r"$\alpha_0$"]

    # Record MAP value of p
    map_p = mode_discrete(nleaves, axis=None).mode

    # Loop through each subplot
    for i in range(min_p, max_p + 1):
        chain_common_p = chain_common[nleaves == i]

        for j, col_name in enumerate(column_names):
            ax_plot = fig.add_subplot(max_p, 2, (i - min_p) * 2 + j + 1)

            # Catch warnings and delete the plot if the warning is raised
            with warnings.catch_warnings():
                message = "Your data appears to have a single value or no finite values"
                warnings.filterwarnings("error", category=UserWarning, message=message)
                try:
                    # Plot data
                    az.plot_dist(
                        chain_common_p[:, j],
                        ax=ax_plot,
                        color=colors[j],
                        fill_kwargs={"alpha": 0.2},
                        plot_kwargs={"linewidth": 1},
                    )
                except UserWarning as e:
                    if message in str(e):
                        # fig.delaxes(ax_plot)
                        ax_plot.axvline(
                            chain_common_p[0, j],
                            color=colors[j],
                            linewidth=1,
                        )

            ax_plot.set_yticklabels([])
            ax_plot.tick_params(axis="x", labelsize=8)
            ax_plot.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_plot.xaxis.set_major_formatter(custom_format_xaxis)

            if j == 0:
                ax_plot.set_ylabel(
                    row_names[i - min_p],
                    rotation=0,
                    ha="right",
                    verticalalignment="center",
                    fontsize=12,
                    color=color_p if i == map_p else "k",
                )
                ax_plot.yaxis.set_label_coords(-0.1, 0.5)

            if i == max_p:
                ax_plot.set_xlabel(
                    col_name,
                    fontsize=12,
                    labelpad=15,
                )


def plot_flat_posterior(
    samples_vars, theta_space, colors, ref_values=None, plot_sigma2=True
):
    fig, axs = plt.subplots(2, 2, figsize=(9, 5))
    # plt.subplots_adjust(hspace=0.3)

    if not plot_sigma2:
        fig.delaxes(axs[1, 1])

    for i, (var, samples, color) in enumerate(
        zip(theta_space.names, samples_vars, colors)
    ):
        ax = axs[i // 2, i % 2]

        if var == "t":
            var = "$\\tau$"
        elif var == "alpha0":
            var = "$\\alpha_0$"
        elif var == "sigma2":
            var = "$\\sigma^2$"

        # Set title and ticks
        ax.set_title(rf"All samples of {var}", fontsize=10)
        ax.set_yticks([])  # Hide y-axis ticks
        ax.set_yticklabels([])  # Hide y-axis labels

        # Plot dist and reference values
        az.plot_dist(
            samples, ax=ax, color=color, fill_kwargs={"alpha": 0.2}, textsize=9
        )
        if ref_values is not None:
            plot_reference_values(
                ref_values[i],
                ax,
                label=f"Train {var}",
                color=color,
                legend_kwargs={"fontsize": 8},
            )

        if i // 2 > 0:
            plot_mode(samples, ax)


def plot_prediction_results(
    df_all_methods_one,
    df_all_methods_two,
    df_reference_one=None,
    df_reference_two=None,
    title="Prediction results",
    score="RMSE",
    kind="linear",
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))

    df_all_methods_one_noise = df_all_methods_one[df_all_methods_one["Noise"] == True]  # noqa: E712
    df_all_methods_one_noiseless = df_all_methods_one[
        ~(df_all_methods_one["Noise"] == True)  # noqa: E712
    ]

    axs[0].scatter(
        df_all_methods_one_noise[score],
        df_all_methods_one_noise["Estimator"],
        color="tab:green",
        label="RKHS methods with noise",
        s=20,
    )
    axs[0].scatter(
        df_all_methods_one_noiseless[score],
        df_all_methods_one_noiseless["Estimator"],
        color="tab:blue",
        label="RKHS methods without noise",
        s=20,
    )
    axs[1].scatter(
        df_all_methods_two[score],
        df_all_methods_two["Estimator"],
        color="tab:blue",
        s=20,
    )
    if df_reference_one is not None:
        standard_l2 = "flin" if kind == "linear" else "flog"
        # Separate the L^2 regression method to highlight it as a direct competitor
        df_reference_one_without_flin = df_reference_one[
            df_reference_one["Estimator"] != standard_l2
        ]
        df_reference_flin = df_reference_one[
            df_reference_one["Estimator"] == standard_l2
        ]
        axs[0].scatter(
            df_reference_one_without_flin[score],
            df_reference_one_without_flin["Estimator"],
            color="tab:orange",
            label="Reference methods",
            s=20,
        )
        label_freg = (
            r"Standard $L^2$ linear regression"
            if kind == "linear"
            else "Alternative functional logistic regression"
        )
        axs[0].scatter(
            df_reference_flin[score],
            df_reference_flin["Estimator"],
            color="tab:red",
            label=label_freg,
            s=20,
        )
        axs[0].axvline(
            np.mean(df_reference_one[score]),
            linestyle="--",
            color="tab:orange",
            label="Mean of reference methods",
            alpha=0.5,
        )
        axs[1].scatter(
            df_reference_two[score],
            df_reference_two["Estimator"],
            color="tab:orange",
            s=20,
        )
        axs[1].axvline(
            np.mean(df_reference_two[score]),
            linestyle="--",
            color="tab:orange",
            alpha=0.5,
        )

    for ax in axs:
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel(score, fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%0.3f"))
        ax.xaxis.set_major_formatter(custom_format_xaxis)

    axs[0].set_title("One-stage methods", fontsize=12)
    axs[1].set_title("Two-stage methods", fontsize=12)

    plt.suptitle(title, fontsize=13, fontweight="semibold")

    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=10,
        ncol=2,
    )

    return fig


def plot_ppc(
    chain_components,
    chain_common,
    theta_space,
    X,
    y,
    is_test_data,
    num_pp_samples=500,
    figsize=(6, 4),
    kind="linear",
):
    pp = predict_pp(
        chain_components, chain_common, theta_space, X, noise=True, kind=kind
    )
    idata_pp = pp_to_arviz_idata(pp, y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(
        f"Posterior predictive distribution for {'X_test' if is_test_data else 'X'}",
        fontsize=14,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        if kind == "linear":
            az.plot_ppc(
                idata_pp,
                data_pairs={"y_obs": "y_star"},
                num_pp_samples=num_pp_samples,
                ax=ax,
            )
        else:  # logistic
            n_success = pp.reshape(-1, pp.shape[-1]).sum(axis=-1)
            y_str = "Y" if not is_test_data else "Y_{test}"
            ax.set_yticks([])
            ax.tick_params(labelsize=8)
            ax.set_title("T = No. of successes (1's)")
            az.plot_dist(n_success, label=rf"$T({y_str}^*)$", ax=ax)

            ax.axvline(
                n_success.mean(),
                ls="--",
                color="orange",
                lw=2,
                label=rf"$Mean(T({y_str}^*))$",
            )
            ax.axvline(
                y.sum(),
                ls="--",
                color="red",
                lw=2,
                label=rf"$T({y_str})$",
            )
            ax.legend(fontsize=10)