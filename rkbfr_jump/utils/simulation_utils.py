# encoding: utf-8

import warnings

import numpy as np
from scipy.integrate import trapz
from scipy.special import expit
from skfda.datasets import (
    fetch_cran,
    fetch_growth,
    fetch_medflies,
    fetch_phoneme,
    fetch_tecator,
)
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import (
    LinearSmootherGeneralizedCVScorer,
    SmoothingParameterSearch,
    akaike_information_criterion,
)
from skfda.representation import FDataGrid


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


def apply_threshold(y, th=0.5):
    """Convert probabilities to class labels."""
    y_th = np.copy(y).astype(int)
    y_th[..., y >= th] = 1
    y_th[..., y < th] = 0

    return y_th


def linear_component_to_label(y_lin, random_noise=None, seed=None):
    """Convert linear component into class labels."""
    return probability_to_label(expit(y_lin), random_noise, seed)


def probability_to_label(probs, random_noise=None, seed=None):
    """Convert probabilities into class labels."""
    labels = np.random.binomial(1, probs)

    if random_noise is not None:
        labels = apply_label_noise(labels, random_noise, seed=seed)

    return labels


def apply_label_noise(y, noise_frac=0.05, seed=None):
    """Apply a random noise to the labels."""
    rng = np.random.default_rng(seed)

    y_noise = y.copy()
    n_noise = int(len(y) * noise_frac)

    idx_0 = rng.choice(np.where(y == 0)[0], size=n_noise)
    idx_1 = rng.choice(np.where(y == 1)[0], size=n_noise)

    y_noise[idx_0] = 1
    y_noise[idx_1] = 0

    return y_noise


def normalize_grid(grid, n_min, n_max):
    """Normalize a grid to [n_min, n_max]"""
    g_min, g_max = np.min(grid), np.max(grid)
    return n_min + (grid - g_min) * (n_max - n_min) / (g_max - g_min)


def get_data_linear(
    is_simulated_data,
    regressor_type,
    model_type,
    n_samples=150,
    n_grid=100,
    mean_vector=None,
    kernel_fn=None,
    beta_coef_true=None,
    beta_tau_true=None,
    tau_range=(0, 1),
    return_y_noiseless=False,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        grid = np.linspace(tau_range[0] + 1.0 / n_grid, tau_range[1], n_grid)
        alpha0_true = 5.0
        sigma2_true = 0.5

        # Generate regressors
        if regressor_type.lower() == "gbm":
            x = np.exp(gp(grid, mean_vector, brownian_kernel, n_samples, rng))
        else:  # GP
            x = gp(grid, mean_vector, kernel_fn, n_samples, rng)

        # Generate response
        if model_type.lower() == "l2":
            if beta_coef_true is None:
                raise ValueError("Must provide a coefficient function.")

            y = generate_l2_dataset(
                x, grid, beta_coef_true, alpha0_true, sigma2_true, rng=rng
            )

        elif model_type.lower() == "rkhs":
            beta_true, tau_true = beta_tau_true

            y = generate_rkhs_dataset(
                x, grid, beta_true, tau_true, alpha0_true, sigma2_true, rng=rng
            )

            if return_y_noiseless:
                y_noiseless = generate_rkhs_dataset(
                    x, grid, beta_true, tau_true, alpha0_true, 0.0, rng=rng
                )

        else:
            raise ValueError("Invalid model generation strategy.")

        # Create FData object
        x_fd = FDataGrid(x, grid)

    else:  # Real data
        if model_type.lower() == "tecator":
            x_fd, y = fetch_tecator(return_X_y=True)
            data = x_fd.data_matrix[..., 0]
            u, idx = np.unique(data, axis=0, return_index=True)  # Find repeated
            x_fd = FDataGrid(data[idx], x_fd.grid_points[0])
            y = y[idx, 1]  # Fat level
        elif model_type.lower() == "moisture":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                data = fetch_cran("Moisturespectrum", "fds")["Moisturespectrum"]
                y = fetch_cran("Moisturevalues", "fds")["Moisturevalues"]
            x_fd = FDataGrid(data["y"].T[:, ::7], data["x"][::7])
        elif model_type.lower() == "sugar":
            data = np.load("data/sugar.npz")
            x_fd = FDataGrid(data["x"][:, ::5])
            y = data["y"]
        else:
            raise ValueError("Real data set must be 'tecator', 'moisture' or 'sugar'.")

        grid = normalize_grid(x_fd.grid_points[0], tau_range[0], tau_range[1])

        x_fd = FDataGrid(x_fd.data_matrix, grid)

    if return_y_noiseless:
        return x_fd, y, grid, y_noiseless

    return x_fd, y, grid


def get_data_logistic(
    is_simulated_data,
    model_type,
    n_samples=150,
    n_grid=100,
    mean_vector=None,
    kernel_fn=None,
    beta_coef_true=None,
    beta_tau_true=None,
    noise=0.05,
    tau_range=(0, 1),
    mean_vector2=None,
    kernel_fn2=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if is_simulated_data:
        grid = np.linspace(tau_range[0] + 1.0 / n_grid, tau_range[1], n_grid)
        alpha0_true = -0.5

        if model_type.lower() == "mixture":
            x, y = generate_mixture_dataset(
                grid,
                mean_vector,
                mean_vector2,
                kernel_fn,
                kernel_fn2,
                n_samples,
                noise,
                rng,
            )

        else:  # Logistic model (RKHS or L2)
            # Generate regressors
            x = gp(grid, mean_vector, kernel_fn, n_samples, rng)

            # Generate response
            if model_type.lower() == "l2":
                if beta_coef_true is None:
                    raise ValueError("Must provide a coefficient function.")

                y_lin = generate_l2_dataset(
                    x, grid, beta_coef_true, alpha0_true, sigma2=0.0, rng=rng
                )
            elif model_type.lower() == "rkhs":
                beta_true, tau_true = beta_tau_true
                y_lin = generate_rkhs_dataset(
                    x, grid, beta_true, tau_true, alpha0_true, sigma2=0.0, rng=rng
                )
            else:
                raise ValueError("Invalid model generation strategy.")

            # Transform linear response for logistic model
            y = linear_component_to_label(
                y_lin, random_noise=noise, seed=rng.integers(2**32)
            )

        # Create FData object
        x_fd = FDataGrid(x, grid)

    else:  # Real data
        if model_type.lower() == "medflies":
            x_fd, y = fetch_medflies(return_X_y=True)
        elif model_type.lower() == "growth":
            x_fd, y = fetch_growth(return_X_y=True)
        elif model_type.lower() == "phoneme":
            x_fd, y = fetch_phoneme(return_X_y=True)
            y_idx = np.where(y < 2)[0]  # Only 2 classes
            rand_idx = rng.choice(y_idx, size=200)  # Choose 200 random curves
            x_fd = FDataGrid(
                x_fd.data_matrix[rand_idx, ::2, 0],  # Half the grid resolution
                x_fd.grid_points[0][::2],
            )
            y = y[rand_idx]
        else:
            raise ValueError("Real data set must be 'medflies', 'growth' or 'phoneme'.")

        grid = normalize_grid(x_fd.grid_points[0], tau_range[0], tau_range[1])
        x_fd = FDataGrid(x_fd.data_matrix, grid)

    return x_fd, y, grid


def smooth_data(X, X_test, smoothing_params):
    # Nadaraya-Watson kernel smoothing
    nw = SmoothingParameterSearch(
        KernelSmoother(kernel_estimator=NadarayaWatsonHatMatrix()),
        smoothing_params,
        param_name="kernel_estimator__bandwidth",
        scoring=LinearSmootherGeneralizedCVScorer(akaike_information_criterion),
        n_jobs=-1,
    )
    nw.fit(X)
    X_nw = nw.transform(X)
    X_test_nw = nw.transform(X_test)

    return X_nw, X_test_nw