# Smart Banking Analytics

Prototype pédagogique d'analyse du risque de crédit au Maroc, construit avec
des données entièrement synthétiques. Le modèle produit un score et une alerte
analytique; il ne prend pas de décision d'octroi ou de refus.

## Démarrage rapide

Le dépôt a été développé avec Python 3.13. Pour créer un environnement et
installer les versions utilisées par le projet :

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

L'interface s'ouvre dans le navigateur. Elle comprend une synthèse des
performances, un audit descriptif par groupe et un formulaire pour scorer puis
télécharger un lot CSV.

Les données synthétiques sont déjà présentes dans `data/raw/`. Pour recréer les
fichiers analytiques, puis entraîner le modèle et générer ses rapports :

```powershell
.\.venv\Scripts\python.exe notebooks\prepare_week3.py
.\.venv\Scripts\python.exe src\train_credit_risk.py
.\.venv\Scripts\python.exe src\analyze_credit_risk.py
.\.venv\Scripts\python.exe src\analyze_threshold_capacity.py
.\.venv\Scripts\python.exe src\monitor_credit_risk.py `
  data\processed\X_candidate.csv `
  reports\synthetic_holdout_drift_monitoring.json `
  --threshold-profile capacity_50
.\.venv\Scripts\python.exe src\analyze_group_fairness.py
```

La préparation écrit dans `data/processed/`; l'entraînement remplace le bundle
du modèle dans `models/` et régénère ses métriques dans `reports/`. Les fichiers
`reports/` sont des sorties reproductibles, pas des sources d'entrée manuelles.

## Scorer un lot

Le CSV doit contenir les 22 colonnes prédictives de
`data/processed/X_candidate.csv`. Une colonne d'identifiant facultative peut
être conservée en sortie et exclue du modèle :

```powershell
.\.venv\Scripts\python.exe src\predict_credit_risk.py `
  --id-column SK_ID_CURR `
  --threshold-profile capacity_50 `
  chemin\vers\lot.csv `
  reports\scored_applications.csv
```

Profils de seuil disponibles : `f2_max` et `capacity_10` à `capacity_80`.
Sans identifiant, la sortie inclut `ligne_source`, un index à base zéro. Les
colonnes de cible et les champs des scénarios de décision sont refusés.

## Structure

- `data/raw/` : tables synthétiques et documentation du schéma.
- `data/processed/` : base d'analyse, cible et variables candidates.
- `notebooks/` : exploration et préparation des données.
- `src/` : entraînement, scoring, analyses et monitoring.
- `models/` : bundle joblib, profil de référence et instructions détaillées.
- `reports/` : métriques, analyses d'erreurs, dérive et comparaisons par groupe.

Les descriptions de chaque commande et des sorties se trouvent dans
`models/README.md`. Les résultats du modèle courant sont résumés dans
`reports/fiche_modele_logistique.md`.

## Limites d'usage

- Toutes les données et étiquettes sont synthétiques; les métriques ne décrivent
  pas la performance sur des clients bancaires réels.
- Les seuils et profils de capacité servent à comparer des hypothèses, pas à
  définir une politique bancaire.
- Les scores calibrés le sont uniquement sur cette distribution synthétique.
- Les rapports par groupe sont descriptifs et ne constituent pas une évaluation
  statistique ou juridique de l'équité.
- Une décision de crédit exige une revue humaine. Ce projet ne doit pas être
  utilisé pour accepter ou refuser automatiquement une demande.
