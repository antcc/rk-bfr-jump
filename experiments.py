# encoding: utf-8

"""
Script to carry out comparison experiments with several linear/logistic
regression methods, functional and otherwise.

Our Bayesian methods are fitted with all the hyperparameters specified,
without a cross-validation loop.

For more information, run `python experiments.py -h`.

Example:

`python experiments.py linear rkhs --kernel fbm --nleaves-max 5 --n-reps 5`
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)

from eryn.ensemble import EnsembleSampler
from eryn.moves import CombineMove, StretchMove
from eryn.state import State
from rkbfr_jump import chain_utils, run_utils, utility
from rkbfr_jump import simulation_utils as simulation
from rkbfr_jump.likelihood import RKHSLikelihoodLinear, RKHSLikelihoodLogistic
from rkbfr_jump.moves import GroupMoveRKHS, MTRJMoveRKHS, RJMoveRKHS
from rkbfr_jump.parameters import ThetaSpace
from rkbfr_jump.prior import RKHSPriorLinear, RKHSPriorLogistic
from rkbfr_jump.update import AdjustStretchScaleCombineMove

###################################################################
# GLOBAL CONFIGURATION
###################################################################

# Floating point precision for display
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)
pd.set_option("display.max_columns", 80)
pd.set_option("styler.format.precision", 4)

# I/O behavior
PRINT_TO_FILE = True
SAVE_RESULTS = True
PRINT_PATH = "results/scores/"
SAVE_PATH = PRINT_PATH

# Prediction algorithms
RUN_REF_ALGS = True
RUN_SUMMARY_ALGS = True

# Prediction parameters
NFOLDS_CV = 10
SCORE_NAME_LINEAR = "RMSE"

# Parameter space
TRANSFORM_SIGMA = True
MIN_DIST_TAU = 1

###################################################################
# CMD ARGUMENT PARSING FUNCTION
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
        "--nreps",
        type=int,
        default=1,
        help="number of random train/test splits for robustness",
    )

    # Optional dataset arguments
    parser.add_argument(
        "-n", "--nsamples", type=int, default=300, help="number of functional samples"
    )
    parser.add_argument(
        "-N",
        "--ngrid",
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

    # Optional sampler arguments
    parser.add_argument(
        "-p",
        "--nleaves-max",
        type=int,
        default=5,
        help="maximum value of p (number of components)",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=32,
        help="number of independent chains in MCMC algorithm",
    )
    parser.add_argument(
        "--ntemps",
        type=int,
        default=5,
        help="number of temperatures in MCMC algorithm",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=500,
        help="number of iterations in MCMC algorithm",
    )
    parser.add_argument(
        "--nburn",
        type=int,
        default=500,
        help="number of initial samples to discard in MCMC algorithm",
    )
    parser.add_argument(
        "--num-try",
        type=int,
        default=2,
        help="number of tries in Multiple Try RJ scheme (1 for no MT)",
    )
    parser.add_argument(
        "--scale-prior-beta",
        type=float,
        default=5,
        help="Scale for the vague prior on beta",
    )
    parser.add_argument(
        "--lambda-p",
        type=int,
        default=0,
        help="parameter lambda for the prior on p (0 means uniform prior)",
    )
    parser.add_argument(
        "--leaf-by-leaf",
        action="store_true",
        help="whether to sample in a leaf-by-leaf manner in the in-model moves of the components",
    )

    # Optional prediction arguments
    parser.add_argument(
        "--prediction-noise",
        action="store_true",
        help="whether to include noise in the predictions: add the value of sigma2"
        " in the linear case or sample from a Bernouilli in the logistic case",
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
    # SET PARAMETER VALUES
    ##

    # --- Parse command-line arguments

    parser = get_arg_parser()
    args = parser.parse_args()

    # --- Generic parameters

    # Randomness and reproducibility
    seed = args.seed
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Linear or logistic problem
    kind = args.kind

    # CV parameters
    if kind == "linear":
        cv_folds = KFold(NFOLDS_CV, shuffle=True, random_state=seed)
    else:
        cv_folds = StratifiedKFold(NFOLDS_CV, shuffle=True, random_state=seed)

    # Score and columns names for the results dataframes
    if kind == "linear":
        score_name = SCORE_NAME_LINEAR
        column_names_split = ["Estimator", "Features", "Noise", "RMSE", "rRMSE"]
    else:
        score_name = "Acc"
        column_names_split = ["Estimator", "Features", "Noise", "Acc"]

    column_names_mean = [
        "Estimator",
        "Mean features",
        "SD features",
        f"Mean {score_name}",
        f"SD {score_name}",
    ]
    sort_by_split = -2 if kind == "linear" else -1

    # --- Dataset generation parameters

    mean_vector = None
    tau_range = (0, 1)
    beta_coef_true = (
        simulation.cholaquidis_scenario3
    )  # True coefficient function for L2 model
    beta_true = [-5.0, 5.0, 10.0]  # True components for RKHS model
    tau_true = [0.1, 0.6, 0.8]  # True time instants for RKHS model
    smoothing_params = np.logspace(-2, 2, 50)

    # --- Bayesian model parameters

    relabel_strategy = "auto"
    df_prior_beta = 5
    scale_prior_beta = args.scale_prior_beta
    scale_prior_alpha0 = 10
    lambda_p = args.lambda_p if args.lambda_p > 0 else None  # 0 means uniform prior

    summary_statistics = [
        np.mean,
        lambda x, axis: trim_mean(x, 0.1),
        np.median,
        lambda x, axis: np.apply_along_axis(utility.mode_kde, axis=axis, arr=x),
    ]

    # --- Eryn sampler parameters

    branch_names = ["components", "common"]
    nleaves_max = {"components": args.nleaves_max, "common": 1}
    nleaves_min = {"components": 1, "common": 1}
    ndims = {"components": 2, "common": 2 if kind == "linear" else 1}
    nwalkers = args.nwalkers
    ntemps = args.ntemps
    nsteps = args.nsteps
    nburn = args.nburn
    thin_by = 1
    num_try = args.num_try  # Number of tries for MT RJMCMC
    group_move_leaf_by_leaf = args.leaf_by_leaf

    ##
    # GET DATASET
    ##

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

    ngrid = args.ngrid
    nsamples = args.nsamples

    # Retrieve data
    if kind == "linear":
        regressor_type = "gbm" if args.kernel == "gbm" else "gp"
        x_fd, y, grid = simulation.get_data_linear(
            is_simulated_data,
            regressor_type,
            model_type,
            nsamples,
            ngrid,
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
                half_ngrid = ngrid // 2
                mean_vector2 = np.concatenate(
                    (  # 0 until 0.5, then 0.75t
                        np.full(half_ngrid, 0),
                        0.75
                        * np.linspace(tau_range[0], tau_range[1], ngrid - half_ngrid),
                    )
                )
            else:  # heteroscedastic
                mean_vector2 = None

                def kernel_fn2(s, t):
                    return simulation.brownian_kernel(s, t, 2.0)
        else:
            mean_vector2 = None
            kernel_fn2 = None

        x_fd, y, grid = simulation.get_data_logistic(
            is_simulated_data,
            model_type,
            nsamples,
            ngrid,
            mean_vector=mean_vector,
            kernel_fn=kernel_fn,
            beta_coef_true=beta_coef_true,
            beta_tau_true=[beta_true, tau_true],
            noise=args.noise,
            tau_range=tau_range,
            mean_vector2=mean_vector2,
            kernel_fn2=kernel_fn2,
            rng=rng,
        )

    if not is_simulated_data:  # Update dataset parameters
        ngrid = len(grid)
        nsamples = len(x_fd)

    ##
    # GET PARAMETER SPACE
    ##

    if kind == "linear":
        theta_space = ThetaSpace(
            grid,
            names=["b", "t", "alpha0", "sigma2"],
            idx=[0, 1, 0, 1],
            transform_sigma=TRANSFORM_SIGMA,
        )
    else:  # logistic
        theta_space = ThetaSpace(
            grid,
            names=["b", "t", "alpha0"],
            idx=[0, 1, 0],
        )

    ##
    # RANDOM TRAIN/TEST SPLITS LOOP
    ##

    ref_scores = defaultdict(lambda: ([], []))  # ([nfeatures], [scores])
    eryn_scores = defaultdict(lambda: ([], []))
    exec_times = np.zeros((args.nreps, 2))  # (splits, (ref, eryn))

    try:
        for rep in range(args.nreps):
            # --- Train/test split and preprocess data

            X_train_fd, X_test_fd, y_train, y_test = train_test_split(
                x_fd,
                y,
                train_size=args.train_size,
                stratify=None if kind == "linear" else y,
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
            X_train = X_train_fd.data_matrix.reshape(-1, ngrid)
            X_test = X_test_fd.data_matrix.reshape(-1, ngrid)

            # Scale training data for our methods
            if kind == "linear":
                X_train_std_orig = np.std(X_train, axis=0)
                X_train_scaled = X_train / X_train_std_orig
                y_train_std_orig = np.std(y_train)
                y_train_scaled = y_train / y_train_std_orig
            else:  # logistic
                X_train_std_orig = np.std(X_train, axis=0)
                X_train_scaled = 0.5 * X_train / X_train_std_orig

            ##
            # RUN REFERENCE ALGORITHMS
            ##

            if RUN_REF_ALGS:
                start = time.time()

                # --- Get reference models

                if kind == "linear":
                    est_ref = run_utils.get_reference_models_linear(
                        args.nleaves_max, seed + rep
                    )
                else:
                    est_ref = run_utils.get_reference_models_logistic(
                        args.nleaves_max, seed + rep
                    )

                if args.verbose > 0:
                    print(f"(It. {rep + 1}/{args.nreps}) Running reference models...")

                # --- Fit models (through CV+refitting) and predict on test set

                df_ref_split, _ = run_utils.cv_sk(
                    est_ref,
                    X_train_fd,
                    y_train,
                    X_test_fd,
                    y_test,
                    cv_folds,
                    kind=kind,
                    n_jobs=-1,
                    column_names=[col for col in column_names_split if col != "Noise"],
                    sort_by=sort_by_split,
                    verbose=args.verbose > 1,
                )

                if args.verbose > 1:
                    print()
                    print(df_ref_split.to_string(index=False, col_space=9), "\n")

                # --- Save score of best models

                for name, features, score in df_ref_split[
                    ["Estimator", "Features", score_name]
                ].values:
                    ref_scores[name][0].append(features)
                    ref_scores[name][1].append(score)

                # --- Measure time

                exec_times[rep, 0] = time.time() - start

            ##
            # RUN ERYN ALGORITHMS
            ##

            start = time.time()

            if args.verbose > 0:
                print(f"(It. {rep + 1}/{args.nreps}) Running Bayesian RKHS models...")

            # --- Define prior distributions and likelihood

            if kind == "linear":
                priors = {
                    "all_models_together": RKHSPriorLinear(
                        theta_space,
                        sd_beta=scale_prior_beta,
                        lambda_p=lambda_p,
                        min_dist_tau=MIN_DIST_TAU,
                    )
                }
                ll = RKHSLikelihoodLinear(theta_space, X_train_scaled, y_train_scaled)
            else:
                priors = {
                    "all_models_together": RKHSPriorLogistic(
                        theta_space,
                        df_beta=df_prior_beta,
                        scale_beta=scale_prior_beta,
                        scale_alpha0=scale_prior_alpha0,
                        lambda_p=lambda_p,
                        min_dist_tau=MIN_DIST_TAU,
                    )
                }
                ll = RKHSLikelihoodLogistic(theta_space, X_train_scaled, y_train)

            # --- Setup initial values

            coords, inds = chain_utils.setup_initial_coords_and_inds(
                ntemps,
                nwalkers,
                nleaves_max,
                ndims,
                theta_space,
                priors["all_models_together"],
                y_train,  # not scaled
                y_train_std_orig if kind == "linear" else None,
                seed + rep,
                kind=kind,
            )

            # --- Setup moves and update functions

            # In-model move for alpha0 and sigma2
            move_stretch = StretchMove(gibbs_sampling_setup="common", a=2)

            # Sample all parameters leaf by leaf in the components branch
            gibbs_sampling_setup_group = [
                (
                    "components",
                    np.zeros(
                        (nleaves_max["components"], ndims["components"]), dtype=bool
                    ),
                )
                for _ in range(nleaves_max["components"])
            ]
            for i in range(nleaves_max["components"]):
                gibbs_sampling_setup_group[i][-1][i] = True

            # In-model move for b and t
            move_group = GroupMoveRKHS(
                theta_space,
                dist_measure="beta",
                nfriends=nwalkers,
                n_iter_update=100,
                gibbs_sampling_setup=gibbs_sampling_setup_group
                if group_move_leaf_by_leaf
                else "components",
                a=2,
            )

            # MT RJ move: generate from prior
            rjmoveMTRKHS = MTRJMoveRKHS(
                priors["all_models_together"],
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                rj=True,
                gibbs_sampling_setup="components",  # Do not specify this if using dependent prior on beta
                num_try=num_try,
            )

            # RJ move: generate from prior
            rjmoveRKHS = RJMoveRKHS(
                priors["all_models_together"],
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                rj=True,
                gibbs_sampling_setup="components",  # Do not specify this if using dependent prior on beta
            )

            # Update function for the parameter 'a' in the in-model moves
            update_fn_group = AdjustStretchScaleCombineMove(
                idx_moves=[0, 1],
                target_acceptance=0.3,
                max_factor=0.1,
                supression_factor=0.1,
                min_a=1.1,
            )
            update_iters = 100

            # --- Posterior sampling with Eryn (RJMCMC)

            ensemble = EnsembleSampler(
                nwalkers,
                ndims,
                ll.evaluate_vectorized,
                priors,
                vectorize=True,
                provide_groups=True,
                tempering_kwargs=dict(ntemps=ntemps),
                nbranches=len(branch_names),
                branch_names=branch_names,
                nleaves_max=nleaves_max,
                nleaves_min=nleaves_min,
                moves=CombineMove([move_group, move_stretch]),
                rj_moves=rjmoveMTRKHS
                if lambda_p is None and num_try > 1
                else rjmoveRKHS,
                update_fn=update_fn_group,
                update_iterations=update_iters,
            )

            # Setup starting state
            state = State(coords, inds=inds)

            # Run the sampler
            ensemble.run_mcmc(
                state, nsteps, burn=nburn, progress=args.verbose > 1, thin_by=thin_by
            )

            # --- Post-process the cold chain only

            # Get full chain, with shape (nsteps, nwalkers, nleaves_max, ndim) and corresponding indices
            (
                full_chain_components,
                full_chain_common,
                inds_components_post,
                inds_common_post,
                idx_order,
            ) = chain_utils.get_full_chain_at_T(
                ensemble,
                theta_space,
                X_train_std_orig,
                y_train_std_orig if kind == "linear" else None,
                T=0,
                relabel_strategy=relabel_strategy,
                kind=kind,
            )

            # Get values of leaves (number of components) accross the chain
            nleaves_all_T = ensemble.get_nleaves()["components"]
            nleaves = nleaves_all_T[:, 0, ...]  # T=0

            # --- Predict on test set

            df_eryn_split = run_utils.compute_eryn_predictions(
                full_chain_components,
                full_chain_common,
                nleaves,
                theta_space,
                X_test,
                y_test,
                X_train,
                y_train,
                summary_statistics,
                column_names_split,
                cv_folds,
                sort_by_split,
                RUN_SUMMARY_ALGS,
                args.prediction_noise,
                seed=seed + rep,
                kind=kind,
            )

            if args.verbose > 1:
                print()
                print(
                    df_eryn_split.to_string(index=False, col_space=9),
                    "\n",
                )

            # --- Save score of best models

            for name, features, score in df_eryn_split[
                ["Estimator", "Features", score_name]
            ].values:
                eryn_scores[name][0].append(features)
                eryn_scores[name][1].append(score)

            # --- Measure time

            exec_times[rep, 1] = time.time() - start

    except KeyboardInterrupt:
        print("\n[INFO] Process halted by user. Skipping...")
        rep = rep - 1

    ##
    # AVERAGE RESULTS ACROSS SPLITS
    ##

    mean_scores_ref = [
        (k, np.mean(v[0]), np.std(v[0]), np.mean(v[1]), np.std(v[1]))
        for k, v in ref_scores.items()
    ]
    df_metrics_ref = pd.DataFrame(
        mean_scores_ref, columns=column_names_mean
    ).sort_values("Mean " + score_name, ascending=kind == "linear")

    mean_scores_eryn = [
        (k, np.mean(v[0]), np.std(v[0]), np.mean(v[1]), np.std(v[1]))
        for k, v in eryn_scores.items()
    ]
    df_metrics_eryn = pd.DataFrame(
        mean_scores_eryn, columns=column_names_mean
    ).sort_values("Mean " + score_name, ascending=kind == "linear")

    ##
    # PRINT RESULTS
    ##

    # --- Compose filename

    if is_simulated_data:
        if args.data == "mixture":
            data_name = "mixture_" + args.kernel
        elif args.kernel == "gbm":
            data_name = "gbm_" + args.data
        else:
            data_name = "gp_" + kernel_fn.__name__ + "_" + args.data
    else:
        data_name = args.data_name

    if kind == "linear":
        prefix_kind = "reg"
    else:
        prefix_kind = "clf"

    filename = prefix_kind + "_" + data_name + "_s_" + str(seed)

    # --- Choose printing medium

    if PRINT_TO_FILE:
        filepath_print = PRINT_PATH + filename + ".out"
        print(f"\n* Output saved to file {filepath_print}")
        f = open(filepath_print, "w")
        original_stdout = sys.stdout
        sys.stdout = f  # Change the standard output to the file we created

    # --- Print model information

    if args.verbose == 1 and not PRINT_TO_FILE:
        print("\n")  # 2 newlines
    elif args.verbose > 1 and not PRINT_TO_FILE:
        print()  # newline

    print(
        f"*** Bayesian RKHS-based Functional {kind.capitalize()} Regression with RJMCMC ***\n"
    )

    print("-- GENERAL INFORMATION --")
    print(f"Random seed: {seed}")
    print(f"Train/test splits: {rep + 1}")

    print("\n-- DATASET GENERATION --")
    print(f"Total samples: {nsamples}")
    print(f"Train size: {len(X_train)}")
    print(f"Grid size: {ngrid}")
    if args.smoothing:
        print("Smoothing: Nadaraya-Watson")

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
        print(f"Real data name: {args.data_name}")

    if kind == "logistic":
        print(f"Noise: {2*int(100*args.noise)}%")

    print("\n-- BAYESIAN RKHS MODEL --")
    print("Max. number of components (p):", args.nleaves_max)
    print(
        "Prior p:",
        f"U[1, {args.nleaves_max}]" if lambda_p is None else f"Poisson({lambda_p})",
    )
    if kind == "linear":
        print(f"Prior beta: N(mu=0, sd={scale_prior_beta})")
    else:
        print(f"Prior beta: t(df={df_prior_beta}, scale={scale_prior_beta})")
    print("Transform sigma:", TRANSFORM_SIGMA)
    print("Min. distance tau:", MIN_DIST_TAU)
    print("Prediction noise:", args.prediction_noise)

    # --- Print MCMC method information and results

    if rep + 1 > 0:
        print("\n-- ERYN SAMPLER --")
        print(f"Walkers: {nwalkers}")
        print(f"Temps: {ntemps}")
        print(f"Burn: {nburn}")
        print(f"Steps: {nsteps}")
        print(f"Num try: {num_try if lambda_p is None else 1}")

        if RUN_REF_ALGS:
            print("\n-- RESULTS REFERENCE METHODS --")
            print(
                "Mean split execution time: "
                f"{exec_times[:rep + 1, 0].mean():.3f}"
                f"±{exec_times[:rep + 1, 0].std():.3f} s"
            )
            print(
                "Total execution time: "
                f"{exec_times[:rep + 1, 0].sum()/60.:.3f} min\n"
            )
            print(df_metrics_ref.to_string(index=False, col_space=7))

        print("\n-- RESULTS ERYN METHODS--")
        print(
            "Mean split execution time: "
            f"{exec_times[:rep + 1, 1].mean():.3f}"
            f"±{exec_times[:rep + 1, 1].std():.3f} s"
        )
        print("Total execution time: " f"{exec_times[:rep + 1, 1].sum()/60.:.3f} min\n")
        print(df_metrics_eryn.to_string(index=False, col_space=7))

    if PRINT_TO_FILE:
        f.close()
        sys.stdout = original_stdout

    ##
    # SAVE RESULTS
    ##

    try:
        if SAVE_RESULTS and rep + 1 > 0:
            # Save all the results dataframe in one CSV file
            filepath_save = SAVE_PATH + filename + ".csv"
            df_results_all = [df_metrics_eryn]
            if RUN_REF_ALGS:
                df_results_all += [df_metrics_ref]

            df_save = pd.concat(df_results_all, axis=0, ignore_index=True)
            df_save = df_save.sort_values(
                "Mean " + score_name, ascending=kind == "linear"
            )
            df_save.to_csv(filepath_save, index=False)

            if not PRINT_TO_FILE:
                print()  # newline
            print(f"* Numerical results saved to file {filepath_save}")
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
