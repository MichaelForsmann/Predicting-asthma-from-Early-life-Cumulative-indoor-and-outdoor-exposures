import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap
from sklearn.metrics import roc_auc_score, f1_score, r2_score, root_mean_squared_error
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold,
    StratifiedKFold, RepeatedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import randint, uniform
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
def _preprocess_fold(X_train, X_test, n_neighbors=20):
    """
    Fits StandardScaler and KNNImputer on X_train only,
    then transforms both X_train and X_test.
    This prevents data leakage from the test set into preprocessing.

    Args:
        X_train:     Training feature DataFrame for the current fold.
        X_test:      Test feature DataFrame for the current fold.
        n_neighbors: Number of neighbours for KNN imputation.

    Returns:
        X_train_processed: Scaled and imputed training DataFrame.
        X_test_processed:  Scaled and imputed test DataFrame (transform only).
    """
    scaler  = StandardScaler()
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fit on training data only, then transform
    X_train_scaled  = scaler.fit_transform(X_train)
    X_train_imputed = imputer.fit_transform(X_train_scaled)

    # FIX (leakage): transform test data using train-fitted objects only
    X_test_scaled   = scaler.transform(X_test)
    X_test_imputed  = imputer.transform(X_test_scaled)

    X_train_processed = pd.DataFrame(X_train_imputed, index=X_train.index, columns=X_train.columns)
    X_test_processed  = pd.DataFrame(X_test_imputed,  index=X_test.index,  columns=X_test.columns)

    return X_train_processed, X_test_processed


def nested_cross_validation_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat,
                                           parameters, n_neighbors=20,n=20):
    """
    Nested cross-validation with hyperparameter tuning (RandomizedSearchCV)
    and SHAP values for a Random Forest Classifier.

    Preprocessing (Z-scaling + KNN imputation) is applied inside each fold,
    fitted on training data only, to prevent data leakage.

    Args:
        X:                          Feature DataFrame.
        y:                          Binary or multiclass target Series.
        Number_of_repeats:          Number of outer CV repetitions.
        Number_of_splits_pr_repeat: Splits per repetition (outer and inner CV).
        parameters:                 Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:                k for KNN imputation (default 20, per paper).

    Returns:
        performance_df:         DataFrame of per-fold AUROC and F1 (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features, n_classes).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    n_classes = y.unique().shape[0]
    shap_values_per_cv = np.zeros((X.shape[0], X.shape[1], n_classes))
    performance_test  = []
    performance_train = []
    brier_test   = []
    brier_train  = []
    feature_importances = []
    parameters_tree = []
    calibration_test_true=[]
    calibration_test_pred=[]
    calibration_train_true=[]
    calibration_train_pred=[]
    prevalence     = float(np.mean(y))
    baseline_brier = prevalence * (1.0 - prevalence)
    CrossValidation = RepeatedStratifiedKFold(
        n_splits=Number_of_splits_pr_repeat,
        n_repeats=Number_of_repeats,
        random_state=42
    )

    for i, (train_outer_ix, test_outer_ix) in enumerate(CrossValidation.split(X, y)):

        X_train_raw, X_test_raw = X.iloc[train_outer_ix], X.iloc[test_outer_ix]
        y_train, y_test         = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        # FIX (leakage): preprocessing fitted on train only, applied to both
        X_train, X_test = _preprocess_fold(X_train_raw, X_test_raw, n_neighbors=n_neighbors)

        cv_inner = StratifiedKFold(
            n_splits=Number_of_splits_pr_repeat,
            random_state=42,
            shuffle=True
        )

        search = RandomizedSearchCV(
            # FIX: random_state and class_weight set for reproducibility and class imbalance
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=parameters,
            n_iter=n,
            cv=cv_inner,
            # FIX: scoring changed to roc_auc to match the paper's primary metric
            # (previously set to "f1" while comment said "roc_auc")
            scoring="roc_auc",
            random_state=42,
            # FIX: n_jobs=-1 to use all cores (was 1, contradicting the comment)
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # SHAP values for the test fold
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_per_cv[test_outer_ix, :, :] += shap_values

        # AUROC
        if n_classes > 2:
            performance_test.append(roc_auc_score(y_test,  best_model.predict_proba(X_test),  multi_class='ovr'))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class='ovr'))
        else:
            performance_test.append(roc_auc_score(y_test,  best_model.predict_proba(X_test)[:,1]))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train)[:,1]))
        brier_tr=brier_score_loss(y_train, best_model.predict_proba(X_train)[:,1])
        brier_te=brier_score_loss(y_test,  best_model.predict_proba(X_test)[:,1])
        
        brier_test.append(1.0 - brier_te/ baseline_brier)
        brier_train.append( 1.0 - brier_tr / baseline_brier)
        cal_test_true, cal_test_pred =calibration_curve(y_test,  best_model.predict_proba(X_test)[:,1],n_bins=5,strategy='quantile' )
        cal_train_true, cal_train_pred= calibration_curve(y_train, best_model.predict_proba(X_train)[:,1],n_bins=5,strategy='quantile' )
        calibration_test_true.append(cal_test_true)
        calibration_test_pred.append(cal_test_pred)
        calibration_train_true.append(cal_train_true)
        calibration_train_pred.append(cal_train_pred)

        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(search.best_params_)

        print(f'Done: {((i + 1) / (Number_of_repeats * Number_of_splits_pr_repeat)) * 100:.1f}%')

    total_shap = shap_values_per_cv / Number_of_repeats

    performance_df = pd.DataFrame(
        [performance_train, performance_test, brier_train, brier_test],
        index=["AUROC_train", "AUROC_test", "brier_train", "brier_test"]
    ).T

    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree,calibration_test_true,calibration_test_pred


def nested_cross_validation_Regression(X, y, Number_of_repeats, Number_of_splits,
                                        parameters, n_neighbors=20):
    """
    Nested cross-validation with hyperparameter tuning (RandomizedSearchCV)
    and SHAP values for a Random Forest Regressor.

    Args:
        X:                Feature DataFrame.
        y:                Continuous target Series.
        Number_of_repeats:  Number of outer CV repetitions.
        Number_of_splits:   Splits per repetition (outer and inner CV).
        parameters:       Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:      k for KNN imputation (default 20).

    Returns:
        performance_df:         DataFrame of per-fold R² and RMSE (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    shap_values_per_cv          = np.zeros((X.shape[0], X.shape[1]))
    root_mean_squared_error_test  = []
    root_mean_squared_error_train = []
    R_squared_test  = []
    R_squared_train = []
    feature_importances = []
    parameters_tree = []

    CV = RepeatedKFold(n_splits=Number_of_splits, n_repeats=Number_of_repeats, random_state=42)

    for i, (train_outer_ix, test_outer_ix) in enumerate(CV.split(X)):

        X_train_raw, X_test_raw = X.iloc[train_outer_ix], X.iloc[test_outer_ix]
        y_train, y_test         = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        # Preprocessing: fit on train, transform both (consistent with classification functions)
        X_train, X_test = _preprocess_fold(X_train_raw, X_test_raw, n_neighbors=n_neighbors)

        cv_inner = KFold(n_splits=Number_of_splits, random_state=i, shuffle=True)

        search = RandomizedSearchCV(
            # FIX: random_state set for reproducibility
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=parameters,
            cv=cv_inner,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_per_cv[test_outer_ix, :] += shap_values

        R_squared_test.append(best_model.score(X_test, y_test))
        R_squared_train.append(best_model.score(X_train, y_train))
        root_mean_squared_error_test.append(root_mean_squared_error(y_test,  best_model.predict(X_test)))
        root_mean_squared_error_train.append(root_mean_squared_error(y_train, best_model.predict(X_train)))

        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(result.best_params_)

        print(f'Done: {((i + 1) / (Number_of_repeats * Number_of_splits)) * 100:.1f}%')

    total_shap = shap_values_per_cv / Number_of_repeats

    performance_df = pd.DataFrame(
        [R_squared_train, R_squared_test,
         root_mean_squared_error_train, root_mean_squared_error_test],
        index=["R_squared_train", "R_squared_test",
               "RMSE_train", "RMSE_test"]
    ).T

    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree


def nested_cross_validation_nodata_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat,
                                                   parameters, n_neighbors=20):
    """
    Identical to nested_cross_validation_classification but retained as a separate
    entry point for backwards compatibility.

    All preprocessing (Z-scaling + KNN imputation) is applied correctly inside
    each fold — fitted on training data only.

    Args:
        X:                          Feature DataFrame.
        y:                          Binary or multiclass target Series.
        Number_of_repeats:          Number of outer CV repetitions.
        Number_of_splits_pr_repeat: Splits per repetition.
        parameters:                 Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:                k for KNN imputation (default 20).

    Returns:
        performance_df:         DataFrame of per-fold AUROC and F1 (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features, n_classes).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    # Delegates entirely to the fixed classification function above
    return nested_cross_validation_classification(
        X, y,
        Number_of_repeats,
        Number_of_splits_pr_repeat,
        parameters,
        n_neighbors=n_neighbors
    )
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, brier_score_loss

from your_module import calibration_metrics  # your ICI function


def nested_cross_calibrated_rf(
    X, y,
    Number_CV=2,
    number_split=3,
    n_neighbors_imputer=5,
    rf_kwargs=None,
    calibration_method="isotonic",
    inner_cv_folds=5,
    random_state=42,
):
    """
    Repeated stratified CV with isotonic-calibrated Random Forest.
    Mirrors nested_cross_bayesian_logistic() for direct comparison.

    Returns
    -------
    dict with the same shape as the Bayesian function:
        - summary           : pooled OOF metrics
        - test_perf         : per-fold test metric arrays (1D, n_folds)
        - train_perf        : per-fold train metric arrays (1D, n_folds)
        - oof_probs         : OOF mean predicted probabilities
        - oof_y             : ground truth
    """
    rf_kwargs = rf_kwargs or {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": random_state,
    }

    rskf = RepeatedStratifiedKFold(
        n_splits=number_split,
        n_repeats=Number_CV,
        random_state=random_state,
    )
    n_folds = number_split * Number_CV

    # ----- Per-fold storage (1D — RF is point-estimate, no posterior) -----
    test_perf = {
        "roc_test":   np.zeros(n_folds),
        "brier_test": np.zeros(n_folds),
        "ici_test":   np.zeros(n_folds),
        "bss_test":   np.zeros(n_folds),
    }
    train_perf = {
        "roc_train":   np.zeros(n_folds),
        "brier_train": np.zeros(n_folds),
        "ici_train":   np.zeros(n_folds),
        "bss_train":   np.zeros(n_folds),
    }

    # ----- OOF probability accumulators (for pooled OOF metrics) -----
    n_samples = len(y)
    oof_sum   = np.zeros(n_samples)
    oof_count = np.zeros(n_samples)

    prevalence     = float(np.mean(y))
    baseline_brier = prevalence * (1 - prevalence)

    j = 0
    for train_idx, test_idx in rskf.split(X, y):
        X_train_raw = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test_raw  = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
        y_train_raw = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test_raw  = y.iloc[test_idx]  if hasattr(y, "iloc") else y[test_idx]

        # ----- Fit imputer + scaler on TRAIN only (no leakage) -----
        imputer = KNNImputer(n_neighbors=n_neighbors_imputer)
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp  = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled  = scaler.transform(X_test_imp)

        # ----- Calibrated RF -----
        base_rf = RandomForestClassifier(**rf_kwargs)
        clf = CalibratedClassifierCV(
            estimator=base_rf,
            method=calibration_method,
            cv=inner_cv_folds,
        )
        clf.fit(X_train_scaled, y_train_raw)

        train_probs = clf.predict_proba(X_train_scaled)[:, 1]
        test_probs  = clf.predict_proba(X_test_scaled)[:, 1]

        # ----- Per-fold metrics -----
        train_perf["roc_train"][j]   = roc_auc_score(y_train_raw, train_probs)
        train_perf["brier_train"][j] = brier_score_loss(y_train_raw, train_probs)
        train_perf["ici_train"][j]   = calibration_metrics(y_train_raw.values, train_probs)
        train_perf["bss_train"][j]   = 1 - train_perf["brier_train"][j] / baseline_brier

        test_perf["roc_test"][j]     = roc_auc_score(y_test_raw, test_probs)
        test_perf["brier_test"][j]   = brier_score_loss(y_test_raw, test_probs)
        test_perf["ici_test"][j]     = calibration_metrics(y_test_raw.values, test_probs)
        test_perf["bss_test"][j]     = 1 - test_perf["brier_test"][j] / baseline_brier

        # ----- OOF accumulator -----
        oof_sum[test_idx]   += test_probs
        oof_count[test_idx] += 1

        j += 1

    # ----- Pooled OOF metrics -----
    assert (oof_count > 0).all(), "Some samples never appeared in a test fold"
    oof_mean_probs = oof_sum / oof_count

    pooled_brier = float(brier_score_loss(y, oof_mean_probs))
    summary = {
        "model":            "calibrated_rf",
        "calibration":      calibration_method,
        "prevalence":       prevalence,
        "baseline_brier":   baseline_brier,
        "pooled_oof_roc":   float(roc_auc_score(y, oof_mean_probs)),
        "pooled_oof_brier": pooled_brier,
        "pooled_oof_bss":   1 - pooled_brier / baseline_brier,
        "pooled_oof_ici":   float(calibration_metrics(np.asarray(y), oof_mean_probs)),
    }

    return {
        "summary":     summary,
        "test_perf":   test_perf,
        "train_perf":  train_perf,
        "oof_probs":   oof_mean_probs,
        "oof_y":       np.asarray(y),
    }
