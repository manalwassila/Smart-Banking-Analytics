"""Summarize synthetic model alerts and errors across client groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "processed" / "applications_financial_base.csv"
CLIENTS_PATH = ROOT / "data" / "raw" / "clients.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "X_candidate.csv"
PREDICTIONS_PATH = ROOT / "reports" / "logistic_regression_test_predictions.csv"
MODEL_PATH = ROOT / "models" / "logistic_regression_credit_risk.joblib"
DEFAULT_OUTPUT = ROOT / "reports" / "synthetic_group_fairness.csv"

GROUP_COLUMNS = {
    "GENDER": "genre",
    "MARITAL_STATUS": "situation_matrimoniale",
    "EDUCATION_LEVEL": "niveau_etudes",
    "EMPLOYMENT_STATUS": "situation_professionnelle",
    "FAMILY_RESPONSIBILITY_LEVEL": "responsabilite_familiale",
}
MIN_GROUP_SIZE = 30


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold-profile",
        default="f2_max",
        help="Profil du bundle (f2_max ou capacity_10 à capacity_80)",
    )
    args = parser.parse_args()

    base = pd.read_csv(BASE_PATH)
    features = pd.read_csv(FEATURES_PATH)
    target = pd.read_csv(ROOT / "data" / "processed" / "y_target.csv")["TARGET"].astype(int)
    clients = pd.read_csv(CLIENTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    bundle = joblib.load(MODEL_PATH)

    if base["SK_ID_CURR"].duplicated().any() or clients["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR doit être unique dans la base de demandes et les profils clients.")
    if len(base) != len(features) or len(base) != len(target):
        raise ValueError("La base, les variables et la cible ne sont pas alignées en nombre de lignes.")
    expected_features = base.drop(
        columns=["SK_ID_CURR", "APPLICATION_DATE", "TARGET", "TARGET_LABEL"]
    )
    if list(features.columns) != list(expected_features.columns):
        raise ValueError("Les colonnes des variables ne correspondent pas à la base utilisée pour l'audit.")
    if not np.array_equal(base["TARGET"].to_numpy(dtype=int), target.to_numpy()):
        raise ValueError("La cible de la base et y_target.csv ne sont pas alignées.")

    threshold_profiles = bundle.get("threshold_profiles", {"f2_max": bundle["threshold"]})
    if args.threshold_profile not in threshold_profiles:
        raise ValueError(f"Profil inconnu. Profils disponibles: {sorted(threshold_profiles)}")
    threshold = float(threshold_profiles[args.threshold_profile])

    row_index = predictions["test_row_index"].astype(int)
    if row_index.duplicated().any() or (row_index < 0).any() or (row_index >= len(base)).any():
        raise ValueError("Les indices de test sont dupliqués ou hors des lignes disponibles.")
    test = base.iloc[row_index.to_numpy()][["SK_ID_CURR"]].reset_index(drop=True)
    test["target"] = predictions["target_reel_synthetique"].to_numpy(dtype=int)
    if not np.array_equal(test["target"].to_numpy(), target.iloc[row_index.to_numpy()].to_numpy()):
        raise ValueError("Les cibles des prédictions de test ne correspondent pas aux lignes source.")
    test["score"] = predictions["score_defaut_calibre_synthetique"].to_numpy(dtype=float)
    test["alert"] = (test["score"] >= threshold).astype("int8")

    age = pd.to_numeric(clients["AGE"], errors="coerce")
    clients = clients.copy()
    clients["TRANCHE_AGE"] = pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
        right=True,
    ).astype("string")
    groups = test.merge(clients, on="SK_ID_CURR", how="left", validate="one_to_one", indicator=True)
    if groups["_merge"].ne("both").any():
        raise ValueError("Certains dossiers de test n'ont pas de profil client correspondant.")
    groups = groups.drop(columns="_merge")
    group_columns = {**GROUP_COLUMNS, "TRANCHE_AGE": "tranche_age"}

    rows = []
    for column, group_name in group_columns.items():
        for value, part in groups.groupby(column, dropna=False, observed=True):
            n = len(part)
            actual_default = part["target"].eq(1)
            alert = part["alert"].eq(1)
            tp = int((actual_default & alert).sum())
            fp = int((~actual_default & alert).sum())
            fn = int((actual_default & ~alert).sum())
            tn = int((~actual_default & ~alert).sum())
            rows.append(
                {
                    "profil_seuil": args.threshold_profile,
                    "seuil_alerte": threshold,
                    "dimension": group_name,
                    "groupe": "Non renseigne" if pd.isna(value) else str(value),
                    "effectif": n,
                    "taux_defaut_synthetique": safe_rate(tp + fn, n),
                    "score_moyen_synthetique": float(part["score"].mean()),
                    "ecart_score_moyen_taux_defaut": float(
                        part["score"].mean() - actual_default.mean()
                    ),
                    "brier_score_synthetique": float(
                        np.mean((part["score"] - part["target"]) ** 2)
                    ),
                    "taux_alerte": safe_rate(tp + fp, n),
                    "precision_alerte": safe_rate(tp, tp + fp),
                    "rappel_defaut": safe_rate(tp, tp + fn),
                    "taux_fausses_alertes": safe_rate(fp, fp + tn),
                    "effectif_sous_minimum": n < MIN_GROUP_SIZE,
                }
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".json")
    metrics = {
        "taux_defaut_synthetique": "Taux de défaut observé",
        "score_moyen_synthetique": "Score moyen synthétique",
        "brier_score_synthetique": "Brier score synthétique",
        "taux_alerte": "Taux d'alerte",
        "precision_alerte": "Précision des alertes",
        "rappel_defaut": "Rappel des défauts synthétiques",
        "taux_fausses_alertes": "Taux de fausses alertes",
    }
    group_gaps = {}
    for dimension, part in output.groupby("dimension", observed=True):
        eligible = part.loc[~part["effectif_sous_minimum"]]
        dimension_gaps = {}
        for metric, label in metrics.items():
            observed = eligible.dropna(subset=[metric]).set_index("groupe")[metric]
            if observed.empty:
                dimension_gaps[metric] = {"label": label, "groups_compared": 0, "gap": None}
                continue
            min_group = observed.idxmin()
            max_group = observed.idxmax()
            dimension_gaps[metric] = {
                "label": label,
                "groups_compared": int(len(observed)),
                "minimum_group": str(min_group),
                "minimum_rate": float(observed.loc[min_group]),
                "maximum_group": str(max_group),
                "maximum_rate": float(observed.loc[max_group]),
                "absolute_gap_percentage_points": float((observed.loc[max_group] - observed.loc[min_group]) * 100),
            }
        calibration = eligible.dropna(subset=["ecart_score_moyen_taux_defaut"]).set_index("groupe")
        if calibration.empty:
            dimension_gaps["absolute_group_calibration_gap"] = None
        else:
            calibration_gap = calibration["ecart_score_moyen_taux_defaut"].abs()
            group = calibration_gap.idxmax()
            dimension_gaps["absolute_group_calibration_gap"] = {
                "label": "Plus grand écart absolu entre score moyen et taux de défaut observé",
                "group": str(group),
                "absolute_gap_percentage_points": float(calibration_gap.loc[group] * 100),
            }
        group_gaps[str(dimension)] = dimension_gaps

    metadata_path.write_text(
        json.dumps(
            {
                "synthetic_data": True,
                "test_rows": int(len(test)),
                "threshold_profile": args.threshold_profile,
                "threshold": threshold,
                "minimum_group_size_for_interpretation": MIN_GROUP_SIZE,
                "observed_group_rate_gaps": group_gaps,
                "warning": (
                    "Descriptive group comparisons on synthetic test data only. Gap summaries omit "
                    "groups with fewer than the stated minimum rows. They are not statistical "
                    "significance tests, legal fairness thresholds, or evidence of suitability "
                    "for credit decisions."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Analyse descriptive écrite : {args.output}")
    print(f"Métadonnées et limites : {metadata_path}")
    print("Les comparaisons sont descriptives et portent uniquement sur le jeu synthétique.")


if __name__ == "__main__":
    main()
