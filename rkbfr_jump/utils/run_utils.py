import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import mode as mode_discrete
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization
from skfda.ml.classification import KNeighborsClassifier, MaximumDepthClassifier
from skfda.ml.classification import LogisticRegression as FLR
from skfda.ml.classification import NearestCentroid as FNC
from skfda.ml.regression import LinearRegression as FLinearRegression
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA
from skfda.preprocessing.dim_reduction.variable_selection import (
    RKHSVariableSelection as RKVS,
)
from skfda.representation.basis import BSplineBasis, FDataBasis, FourierBasis
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline

sys.path.append("../../")  # To import from the local module "reference_methods"

from reference_methods._flda import FLDA
from reference_methods._fpls import APLS, FPLS

from .. import prediction
from .sklearn_utils import Basis, DataMatrix, FeatureSelector, PLSRegressionWrapper
from .utility import IgnoreWarnings


def fill_df_scores(df, y_true, y_pred, name, features, noise):
    rmse = root_mean_squared_error(y_true, y_pred)

    df.loc[len(df)] = [
        name,
        features,
        noise,
        rmse,
        rmse / np.std(y_true),
    ]


def cv_sk(
    estimators,
    X,
    y,
    X_test,
    y_test,
    folds,
    kind="linear",
    n_jobs=1,
    df=None,
    column_names=None,
    sort_by=-2,
    verbose=False,
):
    if kind == "linear":
        scoring = "neg_mean_squared_error"
        est_name = "reg"
    else:
        scoring = "accuracy"
        est_name = "clf"

    for name, pipe, params in estimators:
        if verbose:
            print(f"  Fitting {name}...")

        est_cv = GridSearchCV(pipe, params, scoring=scoring, n_jobs=n_jobs, cv=folds)

        with IgnoreWarnings():
            est_cv.fit(X, y)

        if name == "fknn":
            K = est_cv.best_params_[f"{est_name}__n_neighbors"]
            n_features = f"K={K}"
        elif name == "mdc" or name == "fnc":
            n_features = X.data_matrix.shape[1]
        elif name == "flr":
            n_features = est_cv.best_estimator_[est_name].p
        elif "qda" in name or "+nc" in name or name == "lda":
            n_features = est_cv.best_estimator_[est_name].n_features_in_
        elif "pls1" in name or name == "flda":
            if kind == "linear":
                n_features = est_cv.best_estimator_[est_name].n_components
            else:
                n_features = est_cv.best_estimator_[
                    est_name
                ].base_regressor.n_components
        else:
            if isinstance(est_cv.best_estimator_[est_name].coef_[0], FDataBasis):
                coef = est_cv.best_estimator_[est_name].coef_[0].coefficients[0]
            elif "log" in name:
                coef = est_cv.best_estimator_[est_name].coef_[0]
            else:
                coef = est_cv.best_estimator_[est_name].coef_

            n_features = sum(~np.isclose(coef, 0))

        y_pred = est_cv.predict(X_test)

        if kind == "linear":
            df = linear_regression_metrics(
                y_test, y_pred, n_features, name, df, column_names, sort_by
            )
        else:
            df = logistic_regression_metrics(
                y_test, y_pred, n_features, name, df, column_names, sort_by
            )

    return df, est_cv


def linear_regression_metrics(
    y_true,
    y_pred,
    n_features,
    predictor_name,
    df=None,
    column_names=None,
    sort_by=-2,
):
    if df is None:
        if column_names is None:
            column_names = ["Estimator", "Features", "RMSE", "rRMSE"]
        df = pd.DataFrame(columns=column_names)

    # r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    rrmse = rmse / np.std(y_true)
    df.loc[len(df)] = [
        predictor_name,
        n_features,
        rmse,
        rrmse,
        # r2
    ]

    df.sort_values(df.columns[sort_by], inplace=True)

    return df


def logistic_regression_metrics(
    y_true,
    y_pred,
    n_features,
    predictor_name,
    df=None,
    column_names=None,
    sort_by=-1,
):
    if df is None:
        if column_names is None:
            column_names = ["Estimator", "Features", "Acc"]
        df = pd.DataFrame(columns=column_names)

    acc = accuracy_score(y_true, y_pred)
    df.loc[len(df)] = [predictor_name, n_features, acc]

    df.sort_values(df.columns[sort_by], inplace=True, ascending=False)

    return df


def get_reference_models_linear(max_n_components, seed):
    alphas = np.logspace(-4, 4, 20)
    n_components = np.arange(max_n_components) + 1
    n_basis_bsplines = n_components[
        n_components >= 4
    ]  # Cubic splines, so n_basis must be >= 4
    n_basis_fourier = n_components[n_components % 2 != 0]

    basis_bspline = [BSplineBasis(n_basis=p) for p in n_basis_bsplines]
    basis_fourier = [FourierBasis(n_basis=p) for p in n_basis_fourier]

    params_regularizer = {"reg__alpha": alphas}
    params_select = {"selector__p": n_components}
    params_pls = {"reg__n_components": n_components}
    params_dim_red = {"dim_red__n_components": n_components}
    params_basis = {"basis__basis": basis_bspline + basis_fourier}

    regressors = linear_regression_comparison_suite(
        params_regularizer,
        params_select,
        params_dim_red,
        params_basis,
        params_pls,
        random_state=seed,
    )

    return regressors


def linear_regression_comparison_suite(
    params_regularizer,
    params_select,
    params_dim_red,
    params_basis,
    params_pls,
    random_state=None,
):
    regressors = []

    """
    MULTIVARIATE MODELS
    """

    # Lasso
    regressors.append(
        (
            "lasso",
            Pipeline([("data_matrix", DataMatrix()), ("reg", Lasso())]),
            params_regularizer,
        )
    )

    # PLS1 regression
    regressors.append(
        (
            "pls1",
            Pipeline([("data_matrix", DataMatrix()), ("reg", PLSRegressionWrapper())]),
            params_pls,
        )
    )

    """
    VARIABLE SELECTION + MULTIVARIATE MODELS
    """

    # Manual+Ridge
    regressors.append(
        (
            "manual+ridge",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("selector", FeatureSelector()),
                    ("reg", Ridge(random_state=random_state)),
                ]
            ),
            {**params_regularizer, **params_select},
        )
    )

    # FPCA+Ridge
    regressors.append(
        (
            "fpca+ridge",
            Pipeline(
                [
                    ("dim_red", FPCA(n_components=3)),  # Retains scores only
                    ("reg", Ridge(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_regularizer},
        )
    )

    # FPLS (fixed basis)+Ridge
    regressors.append(
        (
            "fpls+ridge",
            Pipeline(
                [
                    ("basis", Basis()),
                    ("dim_red", FPLS()),
                    ("reg", Ridge(random_state=random_state)),
                ]
            ),
            {**params_basis, **params_dim_red, **params_regularizer},
        )
    )

    # PCA+Ridge
    regressors.append(
        (
            "pca+ridge",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PCA(random_state=random_state)),
                    ("reg", Ridge(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_regularizer},
        )
    )

    # PLS+Ridge
    regressors.append(
        (
            "pls+ridge",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PLSRegressionWrapper()),
                    ("reg", Ridge(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_regularizer},
        )
    )

    """
    FUNCTIONAL MODELS
    """

    regressors.append(("apls", Pipeline([("reg", APLS())]), params_pls))

    """NOTE: while not strictly necessary, the test data undergoes the
             same basis expansion process as the training data. This is more
             computationally efficient and seems to improve the performance."""

    # Fixed basis + Functional Linear Regression
    regressors.append(
        (
            "flin",
            Pipeline(
                [
                    ("basis", Basis()),
                    ("reg", FLinearRegression()),
                ]
            ),
            params_basis,
        )
    )

    # Fixed basis + FPLS1 regression
    regressors.append(
        (
            "fpls1",
            Pipeline([("basis", Basis()), ("reg", FPLS())]),
            {**params_basis, **params_pls},
        )
    )

    return regressors


def logistic_regression_comparison_suite(
    params_clf,
    params_base_regressors_pls,
    params_select,
    params_dim_red,
    params_var_sel,
    params_depth,
    params_knn,
    params_flr,
    random_state=None,
):
    classifiers = []

    """
    MULTIVARIATE MODELS
    """

    # LR
    classifiers.append(
        (
            "log",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            params_clf,
        )
    )

    # LDA
    classifiers.append(
        ("lda", Pipeline([("data_matrix", DataMatrix()), ("clf", LDA())]), {})
    )

    # QDA
    classifiers.append(
        ("qda", Pipeline([("data_matrix", DataMatrix()), ("clf", QDA())]), {})
    )

    """
    VARIABLE SELECTION + MULTIVARIATE MODELS
    """

    # Manual+LR
    classifiers.append(
        (
            "manual+log",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("selector", FeatureSelector()),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_clf, **params_select},
        )
    )

    # FPCA+LR
    classifiers.append(
        (
            "fpca+log",
            Pipeline(
                [
                    ("dim_red", FPCA(n_components=3)),  # Retains scores only
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_clf},
        )
    )

    # PCA+LR
    classifiers.append(
        (
            "pca+log",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PCA(random_state=random_state)),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_clf},
        )
    )

    # PLS+LR
    classifiers.append(
        (
            "pls+log",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PLSRegressionWrapper()),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_clf},
        )
    )

    # APLS+LR
    classifiers.append(
        (
            "apls+log",
            Pipeline(
                [
                    ("dim_red", APLS()),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_dim_red, **params_clf},
        )
    )

    # RKVS+LR
    classifiers.append(
        (
            "rkvs+log",
            Pipeline(
                [
                    ("var_sel", RKVS()),
                    ("clf", LogisticRegression(random_state=random_state)),
                ]
            ),
            {**params_var_sel, **params_clf},
        )
    )

    # RMH+LR
    """classifiers.append(("rmh+log",
                       Pipeline([
                           ("var_sel", RMH()),
                           ("clf", LogisticRegression(
                               random_state=random_state))]),
                       params_clf
                        ))"""

    # PCA+QDA (Galeano et al. 2015)
    classifiers.append(
        (
            "pca+qda",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PCA(random_state=random_state)),
                    ("clf", QDA()),
                ]
            ),
            params_dim_red,
        )
    )

    # PLS+Nearest centroid
    classifiers.append(
        (
            "pls+nc",
            Pipeline(
                [
                    ("data_matrix", DataMatrix()),
                    ("dim_red", PLSRegressionWrapper()),
                    ("clf", NearestCentroid()),
                ]
            ),
            params_dim_red,
        )
    )

    # APLS+Nearest centroid (Delaigle and Hall 2012)
    classifiers.append(
        (
            "apls+nc",
            Pipeline([("dim_red", APLS()), ("clf", NearestCentroid())]),
            params_dim_red,
        )
    )

    """
    FUNCTIONAL MODELS
    """

    # Functional logistic regression (Berrendero et al. 2021)
    classifiers.append(("flog", Pipeline([("clf", FLR())]), params_flr))

    # FLDA (based on PLS1 regression, see Preda and Saporta 2007)
    classifiers.append(
        (
            "flda",
            Pipeline([("data_matrix", DataMatrix()), ("clf", FLDA())]),
            params_base_regressors_pls,
        )
    )

    # Maximum Depth Classifier
    classifiers.append(
        ("mdc", Pipeline([("clf", MaximumDepthClassifier())]), params_depth)
    )

    # KNeighbors Functional Classification
    classifiers.append(
        ("fknn", Pipeline([("clf", KNeighborsClassifier())]), params_knn)
    )

    # Nearest Centroid Functional Classification
    classifiers.append(("fnc", Pipeline([("clf", FNC())]), {}))

    return classifiers


def get_reference_models_logistic(X, y, seed):
    Cs = np.logspace(-4, 4, 20)
    n_selected = [5, 10, 15, 20, 25, 50]
    n_components = [2, 3, 4, 5, 7, 10, 15, 20]
    n_neighbors = [3, 5, 7, 9, 11]

    pls_regressors = [PLSRegressionWrapper(n_components=p) for p in n_components]

    params_clf = {"clf__C": Cs}
    params_select = {"selector__p": n_selected}
    params_dim_red = {"dim_red__n_components": n_components}
    params_var_sel = {"var_sel__n_features_to_select": n_components}
    params_flr = {"clf__p": n_components}
    params_knn = {
        "clf__n_neighbors": n_neighbors,
        "clf__weights": ["uniform", "distance"],
    }
    params_depth = {"clf__depth_method": [ModifiedBandDepth(), IntegratedDepth()]}
    # params_mrmr = {"var_sel__method": ["MID", "MIQ"]}
    params_base_regressors_pls = {"clf__base_regressor": pls_regressors}

    classifiers = logistic_regression_comparison_suite(
        params_clf,
        params_base_regressors_pls,
        params_select,
        params_dim_red,
        params_var_sel,
        params_depth,
        params_knn,
        params_flr,
        random_state=seed,
    )

    return classifiers


def compute_eryn_linear_predictions(
    chain_components,
    chain_common,
    nleaves,
    theta_space,
    X_test,
    y_test,
    X_train,
    y_train,
    summary_statistics,
    column_names,
    cv_folds,
    sort_by=-1,
    include_summary_methods=True,
    noise="both",  # True, False or 'both'
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    max_used_p = np.max(np.unique(nleaves))

    if noise == "both":
        noises = [True, False]
    else:
        noises = [noise]
    df = pd.DataFrame(columns=column_names)

    # Method 1 (PP)
    names_pp = ["pp_mean", "pp_tmean", "pp_median", "pp_mode"]
    for noise, (name, aggregate_pp) in product(
        noises, zip(names_pp, summary_statistics)
    ):
        Y_pred = prediction.predict_pp(
            chain_components,
            chain_common,
            theta_space,
            X_test,
            aggregate_pp,
            noise=noise,
        )

        fill_df_scores(df, y_test, Y_pred, name, max_used_p, noise)

    # Method 2 (Weighted PP)
    names_w_pp = ["w_pp_mean", "w_pp_tmean", "w_pp_median", "w_pp_mode"]
    for noise, (name, aggregate_pp) in product(
        noises, zip(names_w_pp, summary_statistics)
    ):
        Y_pred = prediction.predict_weighted_pp(
            chain_components,
            chain_common,
            nleaves,
            theta_space,
            X_test,
            aggregate_pp,
            noise=noise,
        )

        fill_df_scores(df, y_test, Y_pred, name, max_used_p, noise)

    # Method 3 (MAP PP)
    names_map_pp = ["map_pp_mean", "map_pp_tmean", "map_pp_median", "map_pp_mode"]
    map_p = mode_discrete(nleaves, axis=None).mode

    for noise, (name, aggregate_pp) in product(
        noises, zip(names_map_pp, summary_statistics)
    ):
        Y_pred = prediction.predict_map_pp(
            chain_components,
            chain_common,
            nleaves,
            map_p,
            theta_space,
            X_test,
            aggregate_pp,
            noise=noise,
        )

        fill_df_scores(df, y_test, Y_pred, name, map_p, noise)

    if include_summary_methods:
        # Method 4 (Weighted summary)
        names_summary = ["mean", "tmean", "median", "mode"]
        for name, summary_statistic in zip(names_summary, summary_statistics):
            Y_pred = prediction.predict_weighted_summary(
                chain_components,
                chain_common,
                nleaves,
                theta_space,
                X_test,
                summary_statistic,
            )

            fill_df_scores(df, y_test, Y_pred, "w_summary_" + name, max_used_p, "N/A")

        # Method 5 (MAP summary)
        for name, summary_statistic in zip(names_summary, summary_statistics):
            Y_pred = prediction.predict_map_summary(
                chain_components,
                chain_common,
                nleaves,
                map_p,
                theta_space,
                X_test,
                summary_statistic,
            )

            fill_df_scores(df, y_test, Y_pred, "map_summary_" + name, map_p, "N/A")

    # Method 6 (Weighted variable selection)
    names_vs = ["mean", "tmean", "median", "mode"]
    params_regularizer = {"reg__alpha": np.logspace(-4, 4, 20)}
    regs = [
        (
            "ridge",
            GridSearchCV(
                Pipeline([("reg", Ridge(random_state=seed))]),
                params_regularizer,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                cv=cv_folds,
            ),
        )
    ]
    for (summary_name, summary_statistic), (reg_name, reg) in product(
        zip(names_vs, summary_statistics), regs
    ):
        Y_pred = prediction.predict_weighted_variable_selection(
            chain_components,
            chain_common,
            nleaves,
            theta_space,
            X_train,
            y_train,
            X_test,
            summary_statistic,
            reg,
        )

        fill_df_scores(
            df,
            y_test,
            Y_pred,
            "w_vs_" + summary_name + "+" + reg_name,
            max_used_p,
            "N/A",
        )

    # Method 7 (MAP variable selection)
    for (summary_name, summary_statistic), (reg_name, reg) in product(
        zip(names_vs, summary_statistics), regs
    ):
        Y_pred = prediction.predict_map_variable_selection(
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
        )

        fill_df_scores(
            df, y_test, Y_pred, "map_vs_" + summary_name + "+" + reg_name, map_p, "N/A"
        )

    df.sort_values(df.columns[sort_by], inplace=True)

    return df
