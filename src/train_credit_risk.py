"""Train and evaluate the Smart Banking Analytics logistic-regression model.

The data used by this project are synthetic. The output is a research prototype,
not a calibrated probability of default or an automated lending decision.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "X_candidate.csv"
TARGET_PATH = ROOT / "data" / "processed" / "y_target.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression_credit_risk.joblib"
METRICS_PATH = ROOT / "reports" / "logistic_regression_training_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "logistic_regression_test_predictions.csv"
THRESHOLD_PATH = ROOT / "reports" / "logistic_regression_threshold_sensitivity.csv"
PROFILE_PATH = ROOT / "models" / "logistic_regression_reference_profile.json"

TEST_SIZE = 0.20
RANDOM_STATE_SPLIT = 42
RANDOM_STATE_CV = 44
N_SPLITS = 5
BETA = 2


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    categorical = features.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE_SPLIT,
                ),
            ),
        ]
    )


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    candidates = []
    for threshold in np.arange(0.05, 0.801, 0.01):
        predictions = (scores >= threshold).astype("int8")
        candidates.append(
            {
                "threshold": float(threshold),
                "f2": float(fbeta_score(y_true, predictions, beta=BETA, zero_division=0)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
            }
        )
    return max(candidates, key=lambda item: (item["f2"], item["precision"]))


def threshold_sensitivity(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 0.801, 0.01):
        predictions = (scores >= threshold).astype("int8")
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f2": float(fbeta_score(y_true, predictions, beta=BETA, zero_division=0)),
                "alerts_count": int(predictions.sum()),
                "alert_rate": float(predictions.mean()),
            }
        )
    return pd.DataFrame(rows)


def capacity_thresholds(sensitivity: pd.DataFrame) -> dict[str, float]:
    profiles = {}
    for capacity in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        feasible = sensitivity[sensitivity["alert_rate"] <= capacity]
        if feasible.empty:
            continue
        selected = feasible.sort_values(
            ["recall", "precision"], ascending=[False, False]
        ).iloc[0]
        profiles[f"capacity_{int(capacity * 100)}"] = float(selected["threshold"])
    return profiles


def make_reference_profile(features: pd.DataFrame, oof_scores: np.ndarray) -> dict:
    profile = {"reference_rows": int(len(features)), "features": {}, "score": {}}
    categorical = set(features.select_dtypes(include=["object", "str"]).columns)

    for column in features.columns:
        series = features[column]
        if column in categorical:
            values = series.astype("string")
            categories = sorted(str(value) for value in values.dropna().unique())
            counts = [int((values == category).sum()) for category in categories]
            counts.append(int(values.isna().sum()))
            profile["features"][column] = {
                "type": "categorical",
                "categories": categories,
                "counts": counts,
                "missing_count": int(values.isna().sum()),
            }
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            cuts = np.unique(np.quantile(finite, np.arange(0.1, 1.0, 0.1))).tolist() if len(finite) else []
            counts = np.histogram(finite, bins=[-np.inf, *cuts, np.inf])[0].astype(int).tolist()
            counts.append(int(len(values) - len(finite)))
            profile["features"][column] = {
                "type": "numeric",
                "quantile_edges": [float(value) for value in cuts],
                "counts": counts,
                "missing_count": int(len(values) - len(finite)),
            }

    score_cuts = np.unique(np.quantile(oof_scores, np.arange(0.1, 1.0, 0.1))).tolist()
    score_counts = np.histogram(oof_scores, bins=[-np.inf, *score_cuts, np.inf])[0].astype(int).tolist()
    profile["score"] = {
        "quantile_edges": [float(value) for value in score_cuts],
        "counts": score_counts,
        "mean": float(np.mean(oof_scores)),
        "median": float(np.median(oof_scores)),
    }
    return profile


def main() -> None:
    features = pd.read_csv(FEATURES_PATH)
    target_frame = pd.read_csv(TARGET_PATH)
    if "TARGET" not in target_frame:
        raise ValueError(f"La colonne TARGET est absente de {TARGET_PATH}")
    target = target_frame["TARGET"].astype("int8")

    forbidden = {"TARGET", "TARGET_LABEL", "DEMO_RISK_SCORE", "RECOMMENDED_DECISION"}
    leaked = sorted(forbidden.intersection(features.columns))
    if leaked:
        raise ValueError(f"Des colonnes de cible ou de décision sont présentes dans X: {leaked}")
    if len(features) != len(target):
        raise ValueError("X_candidate.csv et y_target.csv n'ont pas le même nombre de lignes.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("TARGET doit contenir uniquement les classes 0 et 1.")

    row_indices = np.arange(len(target))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE_SPLIT,
    )
    x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]

    pipeline = build_pipeline(x_train)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE_CV)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__C": [0.1, 1, 10]},
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    # Compare calibration methods using training-only out-of-fold predictions.
    calibration_results = {}
    calibration_oof_scores = {}
    for method in ("sigmoid", "isotonic"):
        candidate = CalibratedClassifierCV(
            estimator=search.best_estimator_, method=method, cv=cv, ensemble=True
        )
        candidate_scores = cross_val_predict(
            candidate, x_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        calibration_oof_scores[method] = candidate_scores
        calibration_results[method] = {
            "brier_score": float(brier_score_loss(y_train, candidate_scores)),
            "log_loss": float(log_loss(y_train, candidate_scores)),
            "average_precision": float(average_precision_score(y_train, candidate_scores)),
        }
    selected_calibration = min(
        calibration_results,
        key=lambda method: (calibration_results[method]["brier_score"], calibration_results[method]["log_loss"]),
    )

    # Use cross-fitted calibrated scores to set a threshold without looking at test labels.
    oof_scores = calibration_oof_scores[selected_calibration]
    operating_point = choose_threshold(y_train, oof_scores)
    threshold = operating_point["threshold"]
    sensitivity = threshold_sensitivity(y_train, oof_scores)
    sensitivity.to_csv(THRESHOLD_PATH, index=False)
    profiles = {"f2_max": threshold, **capacity_thresholds(sensitivity)}
    reference_profile = make_reference_profile(x_train, oof_scores)

    model = CalibratedClassifierCV(
        estimator=search.best_estimator_,
        method=selected_calibration,
        cv=cv,
        ensemble=True,
    )
    model.fit(x_train, y_train)
    test_scores = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_scores >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()

    metrics = {
        "model": "LogisticRegression",
        "purpose": "prototype d'aide à l'analyse du risque de défaut",
        "synthetic_data": True,
        "dataset_rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "positive_rate": float(target.mean()),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE_SPLIT, "stratified": True},
        "cross_validation": {
            "folds": N_SPLITS,
            "random_state": RANDOM_STATE_CV,
            "selection_metric": "average_precision",
            "best_mean_average_precision": float(search.best_score_),
        },
        "best_parameters": search.best_params_,
        "calibration_selection_cv": {
            "selection_metric": "lowest out-of-fold Brier score, with log loss as tie-breaker",
            "results": calibration_results,
            "selected_method": selected_calibration,
        },
        "threshold_selection": {
            "method": f"out-of-fold F2 maximization after {selected_calibration} calibration on training rows",
            "beta": BETA,
            **operating_point,
        },
        "threshold_profiles": profiles,
        "reference_profile": str(PROFILE_PATH.relative_to(ROOT)),
        "test_metrics_at_selected_threshold": {
            "threshold": threshold,
            "threshold_profile": "f2_max",
            "accuracy": float(accuracy_score(y_test, test_predictions)),
            "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
            "f2": float(fbeta_score(y_test, test_predictions, beta=BETA, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "average_precision": float(average_precision_score(y_test, test_scores)),
            "brier_score": float(brier_score_loss(y_test, test_scores)),
            "log_loss": float(log_loss(y_test, test_scores)),
            "alerts_count": int(test_predictions.sum()),
            "alert_rate": float(test_predictions.mean()),
            "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
        },
        "score_warning": "Le score est calibré uniquement sur la distribution synthétique du projet. Il n'est pas validé comme probabilité de défaut pour une population bancaire réelle.",
        "limitations": [
            "Le jeu est synthétique et ne permet pas d'estimer la performance sur une population bancaire réelle.",
            "La classe prédite est une alerte analytique, pas une décision d'octroi ou de refus.",
            "La calibration, l'équité et la validation temporelle restent à étudier.",
        ],
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "base_pipeline": search.best_estimator_,
            "threshold": threshold,
            "threshold_profiles": profiles,
            "feature_names": features.columns.tolist(),
            "metrics_path": str(METRICS_PATH.relative_to(ROOT)),
            "synthetic_data": True,
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    PROFILE_PATH.write_text(json.dumps(reference_profile, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            "test_row_index": test_indices,
            "target_reel_synthetique": y_test.to_numpy(),
            "score_defaut_calibre_synthetique": test_scores,
            "seuil_alerte": threshold,
            "profil_seuil": "f2_max",
            "alerte_defaut": test_predictions,
        }
    ).sort_values("test_row_index").to_csv(PREDICTIONS_PATH, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    print(f"Métriques : {METRICS_PATH}")
    print(f"Prédictions test : {PREDICTIONS_PATH}")
    print(f"Seuils évalués sur l'entraînement : {THRESHOLD_PATH}")
    print(f"Profil de référence dérive : {PROFILE_PATH}")


if __name__ == "__main__":
    main()
