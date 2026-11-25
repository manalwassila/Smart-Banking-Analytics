"""Translate training-only threshold sensitivity into review-capacity scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "data" / "processed" / "y_target.csv"
SENSITIVITY_PATH = ROOT / "reports" / "logistic_regression_threshold_sensitivity.csv"
OUTPUT_PATH = ROOT / "reports" / "threshold_capacity_scenarios.csv"


def main() -> None:
    target = pd.read_csv(TARGET_PATH)["TARGET"].astype(int)
    train_indices, _ = train_test_split(
        range(len(target)), test_size=0.20, stratify=target, random_state=42
    )
    prevalence = float(target.iloc[train_indices].mean())
    sensitivity = pd.read_csv(SENSITIVITY_PATH)

    scenarios = []
    for capacity in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        feasible = sensitivity[sensitivity["alert_rate"] <= capacity]
        if feasible.empty:
            scenarios.append(
                {
                    "review_capacity_rate": capacity,
                    "threshold": None,
                    "estimated_alerts_per_10000": None,
                    "precision": None,
                    "recall": None,
                    "f2": None,
                    "estimated_false_alerts_per_10000": None,
                    "estimated_missed_defaults_per_10000": None,
                }
            )
            continue

        # Maximize recall subject to the stated review capacity; precision breaks ties.
        selected = feasible.sort_values(
            ["recall", "precision"], ascending=[False, False]
        ).iloc[0]
        alerts_per_10000 = float(selected["alert_rate"] * 10_000)
        true_positives_per_10000 = float(selected["recall"] * prevalence * 10_000)
        scenarios.append(
            {
                "review_capacity_rate": capacity,
                "threshold": float(selected["threshold"]),
                "estimated_alerts_per_10000": round(alerts_per_10000),
                "precision": float(selected["precision"]),
                "recall": float(selected["recall"]),
                "f2": float(selected["f2"]),
                "estimated_false_alerts_per_10000": round(alerts_per_10000 - true_positives_per_10000),
                "estimated_missed_defaults_per_10000": round((1 - selected["recall"]) * prevalence * 10_000),
            }
        )

    result = pd.DataFrame(scenarios)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Écrit : {OUTPUT_PATH}")
    print(result.to_string(index=False))
    print("\nCes volumes sont estimés par validation croisée hors pli sur l'entraînement, pas sur le test.")


if __name__ == "__main__":
    main()
