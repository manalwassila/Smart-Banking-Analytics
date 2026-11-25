"""Compare an incoming scoring batch with the training reference profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "logistic_regression_credit_risk.joblib"
DEFAULT_PROFILE = ROOT / "models" / "logistic_regression_reference_profile.json"
FORBIDDEN_COLUMNS = {"TARGET", "TARGET_LABEL", "DEMO_RISK_SCORE", "RECOMMENDED_DECISION"}


def psi(reference_counts: list[int], batch_counts: list[int]) -> float:
    reference = np.asarray(reference_counts, dtype=float)
    batch = np.asarray(batch_counts, dtype=float)
    if len(reference) != len(batch):
        raise ValueError("Les distributions de référence et du lot ont des dimensions différentes.")
    epsilon = 1e-6
    reference_rate = (reference + epsilon) / (reference.sum() + epsilon * len(reference))
    batch_rate = (batch + epsilon) / (batch.sum() + epsilon * len(batch))
    return float(np.sum((batch_rate - reference_rate) * np.log(batch_rate / reference_rate)))


def numeric_distribution(series: pd.Series, edges: list[float]) -> tuple[list[int], int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    counts = np.histogram(values[finite_mask], bins=[-np.inf, *edges, np.inf])[0].astype(int).tolist()
    missing = int((~finite_mask).sum())
    return counts + [missing], missing


def categorical_distribution(series: pd.Series, categories: list[str]) -> tuple[list[int], int]:
    values = series.astype("string")
    counts = [int((values == category).sum()) for category in categories]
    missing = int(values.isna().sum())
    known = set(categories)
    unknown = int((values.notna() & ~values.isin(known)).sum())
    return counts + [missing, unknown], unknown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_csv", type=Path, help="Incoming CSV with model predictor columns only")
    parser.add_argument("output_json", type=Path, help="Path for the drift report")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--threshold-profile", default="f2_max")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    reference = json.loads(args.reference_profile.read_text(encoding="utf-8"))
    batch = pd.read_csv(args.batch_csv)

    leaked = sorted(FORBIDDEN_COLUMNS.intersection(batch.columns))
    if leaked:
        raise ValueError(f"Le fichier du lot contient des colonnes cible/décision interdites : {leaked}")
    expected = bundle["feature_names"]
    missing_columns = [column for column in expected if column not in batch.columns]
    extra_columns = [column for column in batch.columns if column not in expected]
    if missing_columns or extra_columns:
        raise ValueError(
            f"Schéma incorrect. Colonnes manquantes: {missing_columns}; colonnes inattendues: {extra_columns}"
        )
    if batch.empty:
        raise ValueError("Le lot à surveiller ne contient aucune ligne.")

    feature_reports = []
    for column in expected:
        saved = reference["features"][column]
        series = batch[column]
        if saved["type"] == "numeric":
            observed_counts, missing_count = numeric_distribution(series, saved["quantile_edges"])
            reference_counts = saved["counts"]
            unknown_rate = None
        else:
            observed_counts, unknown_count = categorical_distribution(series, saved["categories"])
            missing_count = int(series.isna().sum())
            reference_counts = saved["counts"] + [0]
            unknown_rate = unknown_count / len(batch)
        feature_reports.append(
            {
                "feature": column,
                "type": saved["type"],
                "psi": psi(reference_counts, observed_counts),
                "reference_missing_rate": saved["missing_count"] / reference["reference_rows"],
                "batch_missing_rate": missing_count / len(batch),
                "unknown_category_rate": unknown_rate,
            }
        )

    profile_name = args.threshold_profile
    threshold_profiles = bundle.get("threshold_profiles", {"f2_max": bundle["threshold"]})
    if profile_name not in threshold_profiles:
        raise ValueError(f"Profil inconnu. Profils disponibles: {sorted(threshold_profiles)}")
    scores = bundle["pipeline"].predict_proba(batch[expected])[:, 1]
    score_edges = reference["score"]["quantile_edges"]
    score_counts = np.histogram(scores, bins=[-np.inf, *score_edges, np.inf])[0].astype(int).tolist()
    score_psi = psi(reference["score"]["counts"], score_counts)
    threshold = float(threshold_profiles[profile_name])

    feature_reports.sort(key=lambda item: item["psi"], reverse=True)
    report = {
        "batch_rows": int(len(batch)),
        "reference_rows": int(reference["reference_rows"]),
        "threshold_profile": profile_name,
        "threshold": threshold,
        "score_distribution": {
            "reference_oof_mean": reference["score"]["mean"],
            "reference_oof_median": reference["score"]["median"],
            "batch_mean": float(np.mean(scores)),
            "batch_median": float(np.median(scores)),
            "batch_alert_rate": float((scores >= threshold).mean()),
            "psi": score_psi,
        },
        "features_by_psi_descending": feature_reports,
        "interpretation": (
            "PSI, missingness, and unknown-category rates are monitoring signals only. "
            "No automatic drift threshold or business action is applied. Review changes "
            "with the data owner and, when labels arrive, monitor predictive performance separately."
        ),
        "synthetic_reference": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rapport de dérive écrit : {args.output_json}")
    print(f"Lot : {len(batch)} lignes; PSI des scores : {score_psi:.4f}; taux d'alertes : {report['score_distribution']['batch_alert_rate']:.1%}")
    print("Les indicateurs de dérive ne déclenchent aucune décision automatique.")


if __name__ == "__main__":
    main()
