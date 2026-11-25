"""Score application rows with the trained synthetic-data prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "logistic_regression_credit_risk.joblib"
FORBIDDEN_COLUMNS = {"TARGET", "TARGET_LABEL", "DEMO_RISK_SCORE", "RECOMMENDED_DECISION"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="CSV contenant uniquement les variables explicatives attendues")
    parser.add_argument("output_csv", type=Path, help="Chemin du CSV de scores")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Bundle joblib entraîné")
    parser.add_argument(
        "--id-column",
        help="Colonne facultative identifiant les dossiers; elle est conservée en sortie, jamais utilisée par le modèle",
    )
    parser.add_argument(
        "--threshold-profile",
        default="f2_max",
        help="Profil de seuil (f2_max ou capacity_10 à capacity_80)",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    rows = pd.read_csv(args.input_csv)
    if rows.empty:
        raise ValueError("Le fichier d'entrée ne contient aucune ligne à scorer.")

    leaked = sorted(FORBIDDEN_COLUMNS.intersection(rows.columns))
    if leaked:
        raise ValueError(f"Le fichier d'entrée contient des colonnes de cible/décision interdites : {leaked}")

    expected = bundle["feature_names"]
    if args.id_column and args.id_column not in rows.columns:
        raise ValueError(f"La colonne identifiant '{args.id_column}' est absente du fichier d'entrée.")
    if args.id_column in expected:
        raise ValueError("La colonne identifiant ne peut pas être une variable explicative du modèle.")
    model_columns = [column for column in rows.columns if column != args.id_column]
    missing = [column for column in expected if column not in model_columns]
    extra = [column for column in model_columns if column not in expected]
    if missing or extra:
        raise ValueError(f"Schéma incorrect. Colonnes manquantes: {missing}; colonnes inattendues: {extra}")

    profiles = bundle.get("threshold_profiles", {"f2_max": bundle["threshold"]})
    if args.threshold_profile not in profiles:
        raise ValueError(f"Profil inconnu. Profils disponibles: {sorted(profiles)}")
    scores = bundle["pipeline"].predict_proba(rows[expected])[:, 1]
    threshold = float(profiles[args.threshold_profile])
    output = pd.DataFrame(
        {
            "ligne_source": range(len(rows)),
            "score_defaut_calibre_synthetique": scores,
            "seuil_alerte": threshold,
            "profil_seuil": args.threshold_profile,
            "alerte_defaut": (scores >= threshold).astype("int8"),
        }
    )
    if args.id_column:
        output.insert(0, args.id_column, rows[args.id_column].to_numpy())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"{len(output)} lignes scorées. Sortie : {args.output_csv}")
    print("Le score est calibré uniquement sur le jeu synthétique; il n'est pas validé pour une population bancaire réelle.")
    print("L'alerte ne constitue pas une décision de crédit.")


if __name__ == "__main__":
    main()
