# encoding: utf-8

import numpy as np
from scipy.integrate import trapz
from scipy.special import expit


def brownian_kernel(s, t, sigma=1.0):
    return sigma * np.minimum(s, t)


def fractional_brownian_kernel(s, t, H=0.8):
    return 0.5 * (s ** (2 * H) + t ** (2 * H) - np.abs(s - t) ** (2 * H))


def ornstein_uhlenbeck_kernel(s, t, v=1.0):
    return np.exp(-np.abs(s - t) / v)


def squared_exponential_kernel(s, t, v=0.2):
    return np.exp(-((s - t) ** 2) / (2 * v**2))


def grollemund_smooth(t):
    return (
        5 * np.exp(-20 * (t - 0.25) ** 2)
        - 2 * np.exp(-20 * (t - 0.5) ** 2)
        + 2 * np.exp(-20 * (t - 0.75) ** 2)
    )


def cholaquidis_scenario3(t):
    return np.log(1 + 4 * t)


def cov_matrix(kernel_fn, s, t):
    ss, tt = np.meshgrid(s, t, indexing="ij")

    # Evaluate the kernel over meshgrid (vectorized operation)
    K = kernel_fn(ss, tt)

    return K


def gp(grid, mean_vector, kernel_fn, n_samples, rng=None):
    grid = np.asarray(grid)

    if rng is None:
        rng = np.random.default_rng()
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))

    kernel_matrix = cov_matrix(kernel_fn, grid, grid)

    X = rng.multivariate_normal(mean_vector, kernel_matrix, size=n_samples)

    return X


def generate_l2_dataset(
    X,
    grid,
    beta_coef,
    alpha0,
    sigma2,
    rng=None,
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    beta = beta_coef(grid)
    y = alpha0 + trapz(y=X * beta, x=grid)

    if sigma2 > 0.0:
        y += np.sqrt(sigma2) * rng.standard_normal(size=len(y))

    return y


def generate_rkhs_dataset(
    X,
    grid,
    beta,
    tau,
    alpha0,
    sigma2,
    rng=None,
):
    """Generate dataset based on GP with a given kernel function."""
    if rng is None:
        rng = np.random.default_rng()

    X_0 = X - X.mean(axis=0)  # The RKHS model assumes centered variables

    y = generate_response_linear(
        np.asarray(beta), np.asarray(tau), alpha0, sigma2, X_0, grid, rng=rng
    )

    return y


def generate_mixture_dataset(
    grid,
    mean_vector,
    mean_vector2,
    kernel_fn,
    kernel_fn2,
    n_samples,
    random_noise=None,
    rng=None,
):
    """Generate dataset based on a known distribution on X|Y."""
    if rng is None:
        rng = np.random.default_rng()
    if mean_vector is None:
        mean_vector = np.zeros(len(grid))
    if mean_vector2 is None:
        mean_vector2 = np.zeros(len(grid))

    # Generate samples with p=1/2
    prob = rng.binomial(1, 0.5, size=n_samples)

    n1 = np.count_nonzero(prob)
    n2 = n_samples - n1

    X1 = gp(grid, mean_vector, kernel_fn, n1, rng)
    X2 = gp(grid, mean_vector2, kernel_fn2, n2, rng)
    X = np.vstack((X1, X2))

    # Generate responses
    y = (n1 <= np.arange(n_samples)).astype(int)

    # Shuffle data
    idx = rng.permutation(np.arange(n_samples))
    X = X[idx, :]
    y = y[idx]

    if random_noise is not None:
        y = apply_label_noise(y, random_noise, rng)

    return X, y


def generate_response_linear(beta, tau, alpha0, sigma2, X, grid, rng=None):
    """Generate a linear RKHS response Y given X and θ"""
    idx_tau_grid = np.abs(grid - tau[:, None]).argmin(axis=-1)
    y = alpha0 + X[:, idx_tau_grid] @ beta

    if sigma2 > 0.0:
        if rng is None:
            rng = np.random.default_rng()

        y += np.sqrt(sigma2) * rng.standard_normal(size=y.shape)

    return y


def generate_response_logistic(
    X, theta, theta_space, noise=True, return_prob=False, th=0.5, rng=None
):
    """Generate a logistic RKHS response Y given X and θ.

    Returns the response vector and (possibly) the probabilities associated.
    """
    y_lin = generate_response_linear(X, theta, theta_space, noise=False)

    if noise:
        y = probability_to_label(y_lin, rng=rng)
    else:
        if th == 0.5:
            # sigmoid(x) >= 0.5 iff x >= 0
            y = apply_threshold(y_lin, 0.0)
        else:
            y = apply_threshold(expit(y_lin), th)

    if return_prob:
        return expit(y_lin), y
    else:
        return y


def apply_threshold(y, th=0.5):
    """Convert probabilities to class labels."""
    y_th = np.copy(y).astype(int)
    y_th[..., y >= th] = 1
    y_th[..., y < th] = 0

    return y_th


def probability_to_label(y_lin, random_noise=None, rng=None):
    """Convert probabilities into class labels."""
    if rng is None:
        rng = np.random.default_rng()

    labels = rng.binomial(1, expit(y_lin))

    if random_noise is not None:
        labels = apply_label_noise(labels, random_noise, rng)

    return labels


def apply_label_noise(y, noise_frac=0.05, rng=None):
    """Apply a random noise to the labels."""
    if rng is None:
        rng = np.random.default_rng()

    y_noise = y.copy()
    n_noise = int(len(y) * noise_frac)

    idx_0 = rng.choice(np.where(y == 0)[0], size=n_noise)
    idx_1 = rng.choice(np.where(y == 1)[0], size=n_noise)

    y_noise[idx_0] = 1
    y_noise[idx_1] = 0

    return y_noise


# Normalize a grid to [0,1]
def normalize_grid(grid):
    g_min, g_max = np.min(grid), np.max(grid)
    return (grid - g_min) / (g_max - g_min)
