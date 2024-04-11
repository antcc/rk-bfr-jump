import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from arviz.labels import MapLabeller


def plot_dataset_regression(X, y, plot_means=True, n_samples=None, figsize=(9, 4)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    n, N = X.shape
    grid = np.linspace(1.0 / N, 1.0, N)

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
    X, y, plot_means=True, n_samples=None, figsize=(9, 4), ax=None
):
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    n, N = X.shape
    grid = np.linspace(1.0 / N, 1.0, N)
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