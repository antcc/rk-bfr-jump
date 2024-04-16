# encoding: utf-8

"""
Script to carry out comparison experiments with several linear/logistic
regression methods, functional and otherwise.

A cross-validation loop is included to select the best hyperparameters
for our Bayesian methods.

For more information, run `python results_cv.py -h`.

Example:

`python results_cv.py linear emcee rkhs --kernel fbm --p-range 1 5 --n-folds 5 --n-reps 5`
"""

import argparse
import sys
import time
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from eryn.ensemble import EnsembleSampler
from eryn.moves import CombineMove, GaussianMove, StretchMove
from eryn.prior import ProbDistContainer
from eryn.state import State
from scipy.stats import invgamma, norm, trim_mean
from scipy.stats import mode as mode_discrete
from skfda.exploratory.depth import IntegratedDepth, ModifiedBandDepth
from skfda.representation import FDataGrid
from skfda.representation.grid import FDataGrid
from sklearn.linear_model import LogisticRegressionCV, Ridge, RidgeCV
from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from utils.run_utils import (
    bayesian_variable_selection_predict,
    cv_sk,
    linear_regression_comparison_suite,
    logistic_regression_comparison_suite,
)
from utils.sklearn_utils import DataMatrix, PLSRegressionWrapper

from rkbfr_jump import chain_utils, plot_utils, prediction, run_utils, utility
from rkbfr_jump import simulation_utils as simulation
from rkbfr_jump.likelihood import RKHSLikelihood
from rkbfr_jump.moves import GroupMoveRKHS, MTRJMoveRKHS, RJMoveRKHS
from rkbfr_jump.parameters import LogSqrtTransform, ThetaSpace
from rkbfr_jump.prior import RKHSPriorSimple
from rkbfr_jump.update import AdjustStretchScaleCombineMove

###################################################################
# CONFIGURATION
###################################################################

# Ignore warnings
# os.environ["PYTHONWARNINGS"] = "ignore"
# np.seterr(over="ignore", divide="ignore")

# Floating point precision for display
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)
pd.set_option("display.max_columns", 80)
pd.set_option("styler.format.precision", 4)

# Script behavior
RUN_REF_ALGS = True
VERBOSE = True
PRINT_TO_FILE = False
SAVE_RESULTS = False
PRINT_PATH = "results/"
SAVE_PATH = PRINT_PATH + "out/"


###################################################################
# PARSING FUNCTION
###################################################################


def get_arg_parser():
    parser = argparse.ArgumentParser(
        "Bayesian RKHS-based Functional Regression with RJMCMC"
    )
    data_group = parser.add_mutually_exclusive_group(required=True)

    # Mandatory arguments
    parser.add_argument(
        "kind", help="type of problem to solve", choices=["linear", "logistic"]
    )
    parser.add_argument(
        "data", help="type of data to use", choices=["rkhs", "l2", "mixture", "real"]
    )
    data_group.add_argument(
        "--kernel",
        help="name of kernel to use in simulations",
        choices=["ou", "sqexp", "fbm", "bm", "gbm", "homoscedastic", "heteroscedastic"],
    )
    data_group.add_argument(
        "--data-name",
        help="name of data set to use as real data",
        choices=["tecator", "moisture", "sugar", "medflies", "growth", "phoneme"],
    )

    # Optional experiment arguments
    parser.add_argument(
        "-r",
        "--n-reps",
        type=int,
        default=1,
        help="number of random train/test splits for robustness",
    )

    # Optional dataset arguments
    parser.add_argument(
        "-n", "--n-samples", type=int, default=150, help="number of functional samples"
    )
    parser.add_argument(
        "-N",
        "--n-grid",
        type=int,
        default=100,
        help="number of grid points for functional regressors",
    )
    parser.add_argument(
        "--smoothing",
        action="store_true",
        help="smooth functional data as part of preprocessing",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=2 / 3,
        help="fraction of data used for training",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="fraction of noise for logistic synthetic data",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="do not force predictors and response with unit variance.",
    )

    # Optional sampler arguments
    parser.add_argument(
        "-p",
        "--nleaves-max",
        type=int,
        default=5,
        help="maximum value of p (number of components)",
    )
    parser.add_argument(
        "--n-walkers",
        type=int,
        default=32,
        help="number of independent chains in MCMC algorithm",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="number of iterations in MCMC algorithm",
    )
    parser.add_argument(
        "--n-burn",
        type=int,
        default=500,
        help="number of initial samples to discard in MCMC algorithm",
    )
    parser.add_argument(
        "--num-try",
        type=int,
        default=1,
        help="number of tries in Multiple Try RJ scheme (1 for no MT)",
    )

    # Optional misc. arguments
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        const=1,
        default=0,
        nargs="?",
        help="set verbosity level",
    )

    return parser


###################################################################
# MAIN FUNCTION
###################################################################


def main():
    """Bayesian RKHS-based Functional Regression with RJMCMC"""

    ##
    # SET PARAMETERS VALUES
    ##

    # Parse command-line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    # Randomness and reproducibility
    seed = args.seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Dataset generation parameters
    mean_vector = None
    tau_range = (0, 1)
    beta_coef_true = (
        simulation.cholaquidis_scenario3
    )  # True coefficient function for L2 model
    beta_true = [-5.0, 5.0, 10.0]  # True components for RKHS model
    tau_true = [0.1, 0.6, 0.8]  # True time instants for RKHS model
    smoothing_params = np.logspace(-2, 2, 50)
    if args.kind == "linear":
        cv_folds = KFold(args.n_folds, shuffle=True, random_state=seed)
    else:
        cv_folds = StratifiedKFold(args.n_folds, shuffle=True, random_state=seed)

    # MCMC parameters
    relabel_strategy = "auto"

    # Names
    theta_names = ["β", "τ", "α0", "σ2"]
    point_estimates = ["mean", "median", "mode"]
    if args.kind == "linear":
        score_column = "RMSE"
        all_estimates = ["posterior_mean"] + point_estimates
        columns_name = ["Estimator", "Mean RMSE", "SD RMSE", "Mean rRMSE", "SD rRMSE"]
    else:
        score_column = "Acc"
        all_estimates = ["posterior_mean", "posterior_vote"] + point_estimates
        columns_name = ["Estimator", "Mean Acc", "SD Acc"]

    ##
    # GET DATASET
    ##

    if VERBOSE:
        print("Getting data...\n")

    # Get dataset parameters
    is_simulated_data = not args.data == "real"
    if is_simulated_data:
        model_type = args.data
    else:
        model_type = args.data_name

    if args.kernel == "ou":
        kernel_fn = simulation.ornstein_uhlenbeck_kernel
    elif args.kernel == "sqexp":
        kernel_fn = simulation.squared_exponential_kernel
    elif args.kernel == "bm":
        kernel_fn = simulation.brownian_kernel
    elif args.kernel == "fbm":
        kernel_fn = simulation.fractional_brownian_kernel
    else:  # gbm or mixture or real data
        kernel_fn = None

    # Retrieve data
    if args.kind == "linear":
        regressor_type = "gbm" if args.kernel == "gbm" else "gp"
        x_fd, y, grid = simulation.get_data_linear(
            is_simulated_data,
            regressor_type,
            model_type,
            args.n_samples,
            args.n_grid,
            mean_vector=mean_vector,
            kernel_fn=kernel_fn,
            beta_coef_true=beta_coef_true,
            beta_tau_true=[beta_true, tau_true],
            tau_range=tau_range,
            rng=rng,
        )
    else:  # logistic
        if args.data == "mixture":
            kernel_fn = simulation.brownian_kernel
            if args.kernel == "homoscedastic":
                kernel_fn2 = kernel_fn
                half_n_grid = args.n_grid // 2
                mean_vector2 = np.concatenate(
                    (  # 0 until 0.5, then 0.75t
                        np.full(half_n_grid, 0),
                        0.75
                        * np.linspace(
                            tau_range[0], tau_range[1], args.n_grid - half_n_grid
                        ),
                    )
                )
            else:  # heteroscedastic
                mean_vector2 = None

                def kernel_fn2(s, t):
                    return simulation.brownian_kernel(s, t, 2.0)
        else:
            mean_vector2 = None
            kernel_fn2 = None

        X_fd, y, grid = simulation.get_data_logistic(
            is_simulated_data,
            model_type,
            args.n_samples,
            args.n_grid,
            kernel_fn=kernel_fn,
            beta_coef=beta_coef_true,
            noise=args.noise,
            initial_smoothing=args.smoothing,
            tau_range=tau_range,
            kernel_fn2=kernel_fn2,
            mean_vector2=mean_vector2,
            rng=rng,
        )

    ##
    # RANDOM TRAIN/TEST SPLITS LOOP
    ##

    score_ref_best = defaultdict(list)
    score_bayesian_best = defaultdict(list)
    score_var_sel_best = defaultdict(list)
    score_ref_all = []
    score_bayesian_all = []
    score_var_sel_all = []
    exec_times = np.zeros((args.n_reps, 2))  # (splits, (ref, bayesian))

    if args.kind == "linear":
        rrmse_ref_best = defaultdict(list)
        rrmse_bayesian_best = defaultdict(list)
        rrmse_var_sel_best = defaultdict(list)

    # Multiple-regression estimator for variable selection algorithm
    if args.kind == "linear":
        cv_folds = KFold(5, shuffle=True, random_state=seed)
        est_multiple = Pipeline(
            [
                ("data", DataMatrix()),
                (
                    "reg",
                    RidgeCV(
                        alphas=np.logspace(-4, 4, 20),
                        scoring="neg_mean_squared_error",
                        cv=cv_folds,
                        n_jobs=args.n_cores,
                    ),
                ),
            ]
        )
    else:
        cv_folds = StratifiedKFold(5, shuffle=True, random_state=seed)
        est_multiple = Pipeline(
            [
                ("data", DataMatrix()),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=np.logspace(-4, 4, 20),
                        scoring="accuracy",
                        cv=cv_folds,
                        n_jobs=args.n_cores,
                    ),
                ),
            ]
        )

    try:
        for rep in range(args.n_reps):
            # Train/test split
            if args.kind == "linear":
                X_train_fd, X_test_fd, y_train, y_test = train_test_split(
                    x_fd, y, train_size=args.train_size, random_state=seed + rep
                )
            else:
                X_train_fd, X_test_fd, y_train, y_test = train_test_split(
                    x_fd,
                    y,
                    train_size=args.train_size,
                    stratify=y,
                    random_state=seed + rep,
                )

            # Smooth data
            if args.smoothing:
                X_train_fd, X_test_fd = simulation.smooth_data(
                    X_train_fd, X_test_fd, smoothing_params
                )

            # We always assume that the regressors are centered
            X_m = X_train_fd.mean(axis=0)
            X_train_fd = X_train_fd - X_m
            X_test_fd = X_test_fd - X_m

            # Get data matrices
            X_train = X_train_fd.data_matrix.reshape(-1, len(grid))
            X_test = X_test_fd.data_matrix.reshape(-1, len(grid))

            # Scale training data for our methods so that SD=1
            X_std_orig = np.std(X_train, axis=0)
            X_train_scaled = X_train / X_std_orig
            y_std_orig = np.std(y_train)
            y_train_scaled = y_train / y_std_orig

            ##
            # RUN REFERENCE ALGORITHMS
            ##

            if RUN_REF_ALGS:
                start = time.time()

                # Get reference models
                if args.kind == "linear":
                    est_ref = run_utils.get_reference_models_linear(seed + rep)
                else:
                    est_ref = run_utils.get_reference_models_logistic(
                        X_train, y_train, seed + rep
                    )

                if VERBOSE:
                    print(
                        f"(It. {rep + 1}/{args.n_reps}) "
                        f"Running {len(est_ref)} reference models..."
                    )

                # Fit models (through CV+refitting) and predict on test set
                df_ref_split, est_cv = cv_sk(
                    est_ref,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cv_folds,
                    kind=args.kind,
                    n_jobs=args.n_cores,
                    sort_by=0,
                    verbose=False,
                )

                # Save CV scores
                score_ref_all.append(est_cv.cv_results_)

                # Save score of best models
                ref_models_score = df_ref_split[["Estimator", score_column]]
                for name, score in ref_models_score.values:
                    score_ref_best[name].append(score)
                    if args.kind == "linear":
                        rrmse_ref_best[name].append(score / np.std(y_test))

                exec_times[rep, 0] = time.time() - start

            ##
            # RUN BAYESIAN ALGORITHM
            ##

            start = time.time()

            if VERBOSE:
                print(f"(It. {rep + 1}/{args.n_reps})")

            exec_times[rep, 1] = time.time() - start

    except KeyboardInterrupt:
        print("\n[INFO] Process halted by user. Skipping...")
        rep = rep - 1

    ##
    # AVERAGE RESULTS ACROSS SPLITS
    ##

    mean_scores = []

    if args.kind == "linear":
        dict_results = [
            ("", "", score_ref_best, rrmse_ref_best),
            (args.method + "_", "", score_bayesian_best, rrmse_bayesian_best),
            (args.method + "_", "+ridge", score_var_sel_best, rrmse_var_sel_best),
        ]

        for prefix, suffix, d1, d2 in dict_results:
            # Average RMSE and relative RMSE
            mean_scores.append(
                [
                    (
                        prefix + k + suffix,
                        np.mean(d1[k]),
                        np.std(d1[k]),
                        np.mean(d2[k]),
                        np.std(d2[k]),
                    )
                    for k in d1.keys()
                ]
            )

    else:  # logistic
        # Average accuracy
        mean_scores.append(
            [(k, np.mean(v), np.std(v)) for k, v in score_ref_best.items()]
        )
        mean_scores.append(
            [
                (args.method + "_" + k, np.mean(v), np.std(v))
                for k, v in score_bayesian_best.items()
            ]
        )
        mean_scores.append(
            [
                (args.method + "_" + k + "+log", np.mean(v), np.std(v))
                for k, v in score_var_sel_best.items()
            ]
        )

    df_metrics_ref = pd.DataFrame(mean_scores[0], columns=columns_name).sort_values(
        "Mean " + score_column, ascending=args.kind == "linear"
    )

    df_metrics_bayesian_var_sel = pd.DataFrame(
        mean_scores[1] + mean_scores[2], columns=columns_name
    ).sort_values("Mean " + score_column, ascending=args.kind == "linear")

    ##
    # PRINT RESULTS
    ##

    # Get filename
    if is_simulated_data:
        if args.data == "mixture":
            data_name = "mixture_" + args.kernel
        elif args.kernel == "gbm":
            data_name = "gbm_" + args.data
        else:
            data_name = "gp_" + kernel_fn.__name__ + "_" + args.data
    else:
        data_name = args.data_name

    if args.smoothing:
        smoothing = "_smoothing"
    else:
        smoothing = ""

    if args.kind == "linear":
        prefix_kind = "reg"
    else:
        prefix_kind = "clf"

    filename = (
        prefix_kind
        + "_"
        + args.method
        + "_"
        + (args.moves if args.method == "emcee" else args.step)
        + "_"
        + data_name
        + "_"
        + str(len(X_fd))
        + smoothing
        + ("_std" if args.standardize else "")
        + "_nw_"
        + str(args.n_walkers)
        + "_ni_"
        + str(args.n_iters)
        + "_seed_"
        + str(seed)
    )

    if PRINT_TO_FILE:
        print(f"\nSaving results to file '{filename}.results'")
        f = open(PRINT_PATH + filename + ".results", "w")
        sys.stdout = f  # Change the standard output to the file we created

    print(
        f"\n*** Bayesian-RKHS Functional {args.kind.capitalize()} " "Regression ***\n"
    )

    # Print dataset information

    print("-- GENERAL INFORMATION --")
    print(f"Random seed: {seed}")
    print(f"N_cores: {args.n_cores}")
    print(f"Random train/test splits: {rep + 1}")
    print(f"CV folds: {args.n_folds}")
    print("N_reps MLE:", args.n_reps_mle)

    print("\n-- MODEL GENERATION --")
    print(f"Total samples: {len(X_fd)}")
    print(f"Grid size: {len(X_fd.grid_points[0])}")
    print(f"Train size: {len(X_train)}")
    if args.smoothing == "nw":
        print("Smoothing: Nadaraya-Watson")
    elif args.smoothing == "basis":
        print("Smoothing: BSpline(16)")
    else:
        print("Smoothing: None")

    if args.standardize:
        standardize_str = "Standardized predictors"
        if args.kind == "linear":
            standardize_str += " and response"
        print(standardize_str)

    if is_simulated_data:
        if args.data == "mixture":
            if args.kernel == "homoscedastic":
                print("Model type: BM(0, 1) + BM(m(t), 1)")
            else:
                print("Model type: BM(0, 1) + BM(0, 2)")
        else:
            if args.kernel == "gbm":
                print("X ~ GBM(0, 1)")
            else:
                print(f"X ~ GP(0, {kernel_fn.__name__})")
            print(f"Model type: {args.data.upper()}")

    else:
        print(f"Data name: {args.data_name}")

    if args.kind == "logistic":
        print(f"Noise: {2*int(100*args.noise)}%")

    print("\n-- BAYESIAN RKHS MODEL --")
    print("Number of components (p):", args.nleaves_max)

    if rep + 1 > 0:
        # Print MCMC method information
        if args.method == "emcee":
            print("\n-- EMCEE SAMPLER --")
            print(f"N_walkers: {args.n_walkers}")
            print(f"N_iters: {args.n_iters} + {args.n_tune}")
            print(f"Burn: {args.n_burn}")
            print(f"Frac_random: {args.frac_random}")
            print(f"Num try: {args.num_try}")
        else:
            print("\n-- PYMC SAMPLER --")
            print(f"N_walkers: {args.n_walkers}")
            print(f"N_iters: {args.n_iters} + {args.n_tune}")
            print("Step method: " + ("NUTS" if args.step == "nuts" else "Metropolis"))
            if args.step == "nuts":
                print(f"  Target accept: {args.target_accept}")

        # Print results

        if RUN_REF_ALGS:
            print("\n-- RESULTS REFERENCE METHODS --")
            print(
                "Mean split execution time: "
                f"{exec_times[:rep + 1, 0].mean():.3f}"
                f"±{exec_times[:rep + 1, 0].std():.3f} s"
            )
            print(
                "Total splits execution time: "
                f"{exec_times[:rep + 1, 0].sum()/60.:.3f} min\n"
            )
            print(df_metrics_ref.to_string(index=False, col_space=7))

        print(f"\n-- RESULTS {args.method.upper()} --")
        print(
            "Mean split execution time: "
            f"{exec_times[:rep + 1, 1].mean():.3f}"
            f"±{exec_times[:rep + 1, 1].std():.3f} s"
        )
        print(
            "Total splits execution time: "
            f"{exec_times[:rep + 1, 1].sum()/60.:.3f} min\n"
        )
        print(df_metrics_bayesian_var_sel.to_string(index=False, col_space=7))

    ##
    # SAVE RESULTS
    ##

    try:
        if SAVE_RESULTS and rep + 1 > 0:
            # Save all the results dataframe in one CSV file
            df_all = [df_metrics_bayesian_var_sel]
            if RUN_REF_ALGS:
                df_all += [df_metrics_ref]

            df = pd.concat(df_all, axis=0, ignore_index=True)
            df.to_csv(SAVE_PATH + filename + ".csv", index=False)

            # Save the CV results to disk
            np.savez(
                SAVE_PATH + filename + ".npz",
                score_ref_all=score_ref_all,
                score_bayesian_all=score_bayesian_all,
                score_var_sel_all=score_var_sel_all,
            )
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
