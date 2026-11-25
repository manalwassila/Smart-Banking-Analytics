"""Interface Streamlit du prototype Smart Banking Analytics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "logistic_regression_credit_risk.joblib"
METRICS_PATH = ROOT / "reports" / "logistic_regression_training_metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "logistic_regression_test_predictions.csv"
GROUP_REPORT_PATH = ROOT / "reports" / "synthetic_group_fairness.csv"
GROUP_META_PATH = ROOT / "reports" / "synthetic_group_fairness.json"
FORBIDDEN_COLUMNS = {"TARGET", "TARGET_LABEL", "DEMO_RISK_SCORE", "RECOMMENDED_DECISION"}

st.set_page_config(
    page_title="Smart Banking Analytics",
    page_icon=":material/account_balance:",
    layout="wide",
)


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def render_overview(metrics: dict, predictions: pd.DataFrame, profile: str, threshold: float) -> None:
    st.header("Vue d’ensemble")
    st.caption("Indicateurs issus du jeu de test synthétique de 10 000 demandes.")

    target = predictions["target_reel_synthetique"].astype(int)
    scores = predictions["score_defaut_calibre_synthetique"].astype(float)
    alerts = (scores >= threshold).astype("int8")
    alert_rate = float(alerts.mean())
    precision = float(precision_score(target, alerts, zero_division=0))
    recall = float(recall_score(target, alerts, zero_division=0))
    f2 = float(fbeta_score(target, alerts, beta=2, zero_division=0))
    pr_auc = float(average_precision_score(target, scores))
    roc_auc = float(roc_auc_score(target, scores))

    with st.container(horizontal=True):
        st.metric("Demandes de test", f"{len(predictions):,}".replace(",", " "), border=True)
        st.metric("Défauts synthétiques", format_percent(float(target.mean())), border=True)
        st.metric("Alertes au seuil choisi", format_percent(alert_rate), border=True)
        st.metric("Rappel des défauts", format_percent(recall), border=True)
        st.metric("Précision des alertes", format_percent(precision), border=True)

    st.caption(
        f"Profil : `{profile}` · Seuil : {threshold:.2f} · F2 : {f2:.3f} · "
        f"Average precision : {pr_auc:.3f} · ROC-AUC : {roc_auc:.3f}"
    )

    chart_data = pd.DataFrame({"score": scores, "cible": target.map({0: "Sans défaut observé", 1: "Défaut observé"})})
    chart_data["bande"] = pd.cut(chart_data["score"], bins=np.linspace(0, 1, 21), include_lowest=True)
    distribution = (
        chart_data.groupby(["bande", "cible"], observed=True).size().unstack(fill_value=0)
        .reindex(columns=["Sans défaut observé", "Défaut observé"], fill_value=0)
    )
    distribution.index = [interval.mid for interval in distribution.index]
    st.subheader("Distribution des scores sur le test")
    st.line_chart(
        distribution,
        x_label="Score de défaut synthétique",
        y_label="Nombre de demandes",
        alt="Répartition des scores synthétiques par cible observée",
    )

    with st.expander("Résultats du protocole d’entraînement"):
        test_metrics = metrics["test_metrics_at_selected_threshold"]
        st.json(
            {
                "meilleurs_parametres": metrics["best_parameters"],
                "calibration_retenue": metrics["calibration_selection_cv"]["selected_method"],
                "seuil_f2_entraînement": metrics["threshold_selection"]["threshold"],
                "precision_test_au_seuil_f2": test_metrics["precision"],
                "rappel_test_au_seuil_f2": test_metrics["recall"],
                "brier_test": test_metrics["brier_score"],
                "alertes_test_au_seuil_f2": test_metrics["alerts_count"],
            }
        )


def render_group_audit(group_report: pd.DataFrame, group_metadata: dict) -> None:
    st.header("Audit descriptif par groupe")
    st.warning(
        "Ces comparaisons portent uniquement sur les données synthétiques du projet. "
        "Elles ne constituent pas une évaluation juridique ou statistique de l’équité.",
        title="Interprétation prudente",
    )
    available_profiles = group_report["profil_seuil"].drop_duplicates().tolist()
    audit_profile = st.selectbox("Profil évalué dans le rapport", available_profiles, index=0)
    profile_rows = group_report[group_report["profil_seuil"] == audit_profile]
    st.caption(f"Seuil audité : {float(profile_rows['seuil_alerte'].iloc[0]):.2f} · profil {audit_profile}")

    dimensions = profile_rows["dimension"].drop_duplicates().tolist()
    dimension = st.selectbox("Groupe à examiner", dimensions, format_func=lambda value: value.replace("_", " ").capitalize())
    selected = profile_rows[profile_rows["dimension"] == dimension].sort_values("effectif", ascending=False)
    selected = selected.rename(
        columns={
            "groupe": "Groupe",
            "effectif": "Effectif",
            "taux_defaut_synthetique": "Défaut observé",
            "score_moyen_synthetique": "Score moyen",
            "brier_score_synthetique": "Brier score",
            "taux_alerte": "Taux d’alerte",
            "precision_alerte": "Précision",
            "rappel_defaut": "Rappel défaut",
            "taux_fausses_alertes": "Fausses alertes",
            "effectif_sous_minimum": "Effectif < 30",
        }
    )
    chart = selected[["Groupe", "Taux d’alerte", "Fausses alertes"]].melt(
        id_vars="Groupe", var_name="Mesure", value_name="Part"
    )
    st.bar_chart(
        chart,
        x="Groupe",
        y="Part",
        color="Mesure",
        horizontal=True,
        x_label="Part du groupe",
        alt=f"Taux d’alerte et taux de fausses alertes par groupe : {dimension}",
    )
    display_columns = [
        "Groupe", "Effectif", "Défaut observé", "Score moyen", "Brier score",
        "Taux d’alerte", "Précision", "Rappel défaut", "Fausses alertes", "Effectif < 30",
    ]
    st.dataframe(
        selected[display_columns],
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="percent")
            for column in [
                "Défaut observé", "Score moyen", "Taux d’alerte", "Précision",
                "Rappel défaut", "Fausses alertes",
            ]
        },
        alt="Mesures synthétiques par groupe client",
    )

    summary = group_metadata.get("observed_group_rate_gaps", {}).get(dimension, {})
    if summary:
        with st.expander("Écarts observés entre groupes"):
            gap_rows = []
            for key, item in summary.items():
                if not isinstance(item, dict):
                    continue
                if "absolute_gap_percentage_points" in item and "maximum_group" in item:
                    gap_rows.append(
                        {
                            "Mesure": item["label"],
                            "Groupe bas": item["minimum_group"],
                            "Groupe haut": item["maximum_group"],
                            "Écart (points)": item["absolute_gap_percentage_points"],
                        }
                    )
                elif "absolute_gap_percentage_points" in item and "group" in item:
                    gap_rows.append(
                        {
                            "Mesure": item["label"],
                            "Groupe bas": "—",
                            "Groupe haut": item["group"],
                            "Écart (points)": item["absolute_gap_percentage_points"],
                        }
                    )
            if gap_rows:
                st.dataframe(pd.DataFrame(gap_rows), hide_index=True, alt="Écarts de taux entre groupes")
    st.caption("Les groupes de moins de 30 dossiers sont signalés et exclus des résumés d’écarts.")


def render_scoring(bundle: dict, threshold_profile: str, threshold: float) -> None:
    st.header("Scorer un nouveau lot")
    st.write("Chargez un CSV avec les 22 variables du modèle. Les lignes restent dans votre session et ne sont pas enregistrées par l’application.")
    uploaded = st.file_uploader("Fichier CSV à scorer", type=["csv"], max_upload_size=50)
    if uploaded is None:
        st.info("Le fichier doit contenir toutes les variables du modèle et ne doit pas inclure la cible ni les champs de décision.")
        return

    try:
        rows = pd.read_csv(uploaded)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"Lecture du CSV impossible : {exc}", title="Fichier invalide")
        return
    if rows.empty:
        st.error("Le CSV ne contient aucune ligne.", title="Fichier vide")
        return
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(rows.columns))
    if forbidden:
        st.error(f"Le fichier contient des colonnes interdites : {', '.join(forbidden)}", title="Cible ou décision détectée")
        return

    expected = bundle["feature_names"]
    missing = [column for column in expected if column not in rows.columns]
    extra = [column for column in rows.columns if column not in expected]
    id_choice = st.selectbox("Colonne d’identifiant à conserver (facultative)", ["Aucune", *extra])
    id_column = None if id_choice == "Aucune" else id_choice
    unexpected = [column for column in extra if column != id_column]
    if missing or unexpected:
        st.error(
            f"Schéma incompatible. Variables manquantes : {missing or 'aucune'}; "
            f"colonnes supplémentaires : {unexpected or 'aucune'}. Gardez une seule colonne d’identifiant facultative.",
            title="Vérifiez les colonnes du CSV",
        )
        return

    st.caption(f"{len(rows):,} lignes prêtes · seuil {threshold:.2f} ({threshold_profile})".replace(",", " "))
    if st.button("Calculer les alertes", type="primary", width="stretch"):
        try:
            scores = bundle["pipeline"].predict_proba(rows[expected])[:, 1]
        except (TypeError, ValueError) as exc:
            st.error(f"Le modèle ne peut pas traiter les valeurs du fichier : {exc}", title="Valeurs incompatibles")
            return
        result = pd.DataFrame(
            {
                "ligne_source": np.arange(len(rows)),
                "score_defaut_calibre_synthetique": scores,
                "seuil_alerte": threshold,
                "profil_seuil": threshold_profile,
                "alerte_defaut": (scores >= threshold).astype("int8"),
            }
        )
        if id_column:
            result.insert(0, id_column, rows[id_column].to_numpy())
        alert_rate = float(result["alerte_defaut"].mean())
        with st.container(horizontal=True):
            st.metric("Demandes scorées", f"{len(result):,}".replace(",", " "), border=True)
            st.metric("Alertes analytiques", f"{int(result['alerte_defaut'].sum()):,}".replace(",", " "), border=True)
            st.metric("Taux d’alerte", format_percent(alert_rate), border=True)
        st.dataframe(result.head(100), hide_index=True, alt="Aperçu des scores calculés")
        st.download_button(
            "Télécharger les résultats CSV",
            data=result.to_csv(index=False).encode("utf-8-sig"),
            file_name="smartbank_scores.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
            on_click="ignore",
        )
        st.warning(
            "Le score est calibré uniquement sur les données synthétiques du projet. "
            "Une alerte n’est pas une décision d’acceptation ou de refus.",
            title="Aide à l’analyse uniquement",
        )


metrics = load_json(str(METRICS_PATH))
predictions = load_csv(str(PREDICTIONS_PATH))
group_report = load_csv(str(GROUP_REPORT_PATH))
group_metadata = load_json(str(GROUP_META_PATH))
bundle = load_model()
profiles = bundle.get("threshold_profiles", {"f2_max": bundle["threshold"]})

with st.sidebar:
    st.title("SmartBank")
    st.caption("Analyse du risque de crédit")
    page = st.segmented_control(
        "Espace",
        ["Synthèse", "Audit groupes", "Scoring CSV"],
        default="Synthèse",
        width="stretch",
    )
    profile = st.selectbox(
        "Profil de seuil",
        list(profiles),
        index=list(profiles).index("f2_max") if "f2_max" in profiles else 0,
        help="Le profil modifie le volume d’alertes. Il ne représente pas une règle bancaire validée.",
    )
    threshold = float(profiles[profile])
    st.caption("Prototype pédagogique · données synthétiques")

st.title("Smart Banking Analytics")
st.caption("Prototype d’aide à l’analyse du risque de crédit · Maroc")
st.warning(
    "Données synthétiques uniquement. Le score est expérimental et ne doit pas servir à une décision automatique de crédit.",
    title="Environnement de démonstration",
)

if page == "Synthèse":
    render_overview(metrics, predictions, profile, threshold)
elif page == "Audit groupes":
    render_group_audit(group_report, group_metadata)
else:
    render_scoring(bundle, profile, threshold)
