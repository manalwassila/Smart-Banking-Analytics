"""Create coefficient, calibration-band, and error-analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "logistic_regression_credit_risk.joblib"
FEATURES_PATH = ROOT / "data" / "processed" / "X_candidate.csv"
PREDICTIONS_PATH = ROOT / "reports" / "logistic_regression_test_predictions.csv"


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    base_pipeline = bundle["base_pipeline"]
    features = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    test_x = features.iloc[predictions["test_row_index"].astype(int)].reset_index(drop=True)

    preprocessor = base_pipeline.named_steps["preprocessing"]
    classifier = base_pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    coefficient_rows = []
    for name, coefficient in zip(names, coefficients):
        family, feature = name.split("__", 1)
        is_numeric = family == "numeric"
        coefficient_rows.append(
            {
                "transformed_feature": feature,
                "feature_family": family,
                "coefficient_log_odds": float(coefficient),
                "odds_multiplier": float(np.exp(np.clip(coefficient, -30, 30))),
                "direction": "higher score" if coefficient > 0 else "lower score",
                "interpretation_unit": "one training standard deviation" if is_numeric else "one-hot category indicator",
            }
        )
    coefficient_frame = pd.DataFrame(coefficient_rows)
    coefficient_frame["absolute_coefficient"] = coefficient_frame["coefficient_log_odds"].abs()
    coefficient_frame.sort_values("absolute_coefficient", ascending=False).drop(
        columns="absolute_coefficient"
    ).to_csv(ROOT / "reports" / "logistic_regression_coefficients.csv", index=False)

    labels = predictions["target_reel_synthetique"].astype(int)
    alert = predictions["alerte_defaut"].astype(int)
    score = predictions["score_defaut_calibre_synthetique"].astype(float)
    error_class = np.select(
        [labels.eq(1) & alert.eq(1), labels.eq(0) & alert.eq(1), labels.eq(1) & alert.eq(0)],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    analysis = pd.DataFrame(
        {
            "test_row_index": predictions["test_row_index"],
            "target": labels,
            "score": score,
            "alert": alert,
            "error_class": error_class,
        }
    )
    analysis = pd.concat([analysis, test_x], axis=1)
    analysis["score_band"] = pd.qcut(score, q=10, duplicates="drop")
    calibration_bins = (
        analysis.groupby("score_band", observed=True)
        .agg(
            dossiers=("target", "size"),
            score_moyen=("score", "mean"),
            taux_defaut_observe=("target", "mean"),
            alertes=("alert", "sum"),
        )
        .reset_index()
    )
    calibration_bins["score_band"] = calibration_bins["score_band"].astype(str)
    calibration_bins.to_csv(ROOT / "reports" / "logistic_regression_calibration_bins.csv", index=False)

    key_numeric = [
        "DEBT_TO_INCOME_RATIO",
        "CREDIT_TO_ANNUAL_INCOME",
        "PAYMENT_TO_INCOME_RATIO",
        "MONTHLY_INCOME_MAD",
        "DISPOSABLE_INCOME_MAD",
    ]
    profiles = {}
    for name, part in analysis.groupby("error_class", observed=True):
        profiles[name] = {
            "count": int(len(part)),
            "median_numeric_features": {
                column: float(part[column].median())
                for column in key_numeric
                if column in part.columns
            },
        }

    summaries = {
        "warning": (
            "Coefficients are associations learned from synthetic data, not causal effects. "
            "Correlated financial ratios can share or amplify coefficients. One-hot indicators "
            "are regularized category terms and do not define a causal reference group."
        ),
        "test_alert_threshold": float(bundle["threshold"]),
        "test_error_counts": {key: value["count"] for key, value in profiles.items()},
        "test_error_profiles": profiles,
        "calibration_band_note": (
            "Score bands summarize observed synthetic defaults. They do not establish calibration "
            "for real applicants or a future time period."
        ),
    }
    (ROOT / "reports" / "logistic_regression_error_analysis.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved coefficient ranking, score bands, and test error profiles under reports/.")


if __name__ == "__main__":
    main()
