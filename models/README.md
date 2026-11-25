# Credit risk prototype

The saved estimator is a logistic-regression prototype trained on the project's
synthetic data. It applies sigmoid or isotonic calibration selected by
training-only out-of-fold Brier score, then produces a risk score and analytical
default alert. Calibration applies only to this synthetic distribution; the
score is not validated as a real banking probability of default. The model does
not make a credit approval or rejection decision.

## Train

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe src\train_credit_risk.py
```

The training script reads `data/processed/X_candidate.csv` and
`data/processed/y_target.csv`. It uses a stratified 80/20 train/test split,
selects `C` by five-fold average precision on the training partition, compares
sigmoid and isotonic calibration by out-of-fold Brier score, sets the alert
threshold by out-of-fold F2, then measures performance on the held-out test
partition. Preprocessing is fitted inside each fold.

## Score a CSV

The input CSV must contain the 22 predictor columns in
`data/processed/X_candidate.csv`, in any order. To retain an application key
in the output, pass its column name with `--id-column`; that column is excluded
from the model and copied into the score file. Without an ID, the output still
contains `ligne_source`, a zero-based row index for matching scores to input.
The input must not contain TARGET, TARGET_LABEL, or synthetic decision-scenario
outputs.

```powershell
.\.venv\Scripts\python.exe src\predict_credit_risk.py `
  --id-column SK_ID_CURR `
  data\processed\X_candidate.csv `
  reports\scored_applications.csv
```

To use the illustrative 50% review-capacity profile instead of the default
F2-maximizing profile, pass the profile option explicitly:

```powershell
.\.venv\Scripts\python.exe src\predict_credit_risk.py `
  --threshold-profile capacity_50 `
  data\processed\X_candidate.csv `
  reports\scored_applications_capacity_50.csv
```

The output contains `score_defaut_calibre_synthetique`, `seuil_alerte`, and
`alerte_defaut`. The score is calibrated only to the project's synthetic data;
do not interpret it as a validated real-world probability of default. Review the
metrics and limitations in `reports/logistic_regression_training_metrics.json`
before interpreting any result. This prototype requires human review and is not
validated for real banking use.

For the current run's results, calibration diagnostics, and error analysis, see
`reports/fiche_modele_logistique.md`.

The current F2-selected threshold is 0.11 and raises alerts for about 70% of the
held-out synthetic test rows. That is a high review volume. Use
`reports/logistic_regression_threshold_sensitivity.csv` to compare other
precision/recall trade-offs; an acceptable threshold depends on analyst capacity
and the relative cost of missed defaults and false alerts.

## Error analysis and coefficients

```powershell
.\.venv\Scripts\python.exe src\analyze_credit_risk.py
.\.venv\Scripts\python.exe src\analyze_threshold_capacity.py
```

This writes coefficient rankings, test score bands, and an error-profile
summary under `reports/`. Coefficients describe associations in the synthetic
training set; correlated ratios and one-hot categories should not be read as
causal effects.

The capacity analysis estimates a threshold for review limits from 10% to 80%,
maximizing recall subject to each limit. Its output is based on training-only
out-of-fold scores and is an aid to choosing a policy assumption, not an
automated policy decision. Results are saved to
`reports/threshold_capacity_scenarios.csv`.

## Monitor an incoming batch

Training writes a reference profile to
`models/logistic_regression_reference_profile.json`. Compare a new feature-only
CSV against it with:

```powershell
.\.venv\Scripts\python.exe src\monitor_credit_risk.py `
  path\to\incoming_batch.csv `
  reports\incoming_batch_drift.json `
  --threshold-profile capacity_50
```

The report includes feature and score PSI, missingness, unknown categories, and
the expected alert rate under the chosen threshold profile. It reports signals
for review and does not automatically reject a batch or make a credit decision.
The synthetic holdout demonstration is saved to
`reports/synthetic_holdout_drift_monitoring.json`.

## Compare alerts across client groups

Run a descriptive group analysis on the held-out synthetic test rows:

```powershell
.\.venv\Scripts\python.exe src\analyze_group_fairness.py
```

The CSV report compares synthetic default rates, mean scores, Brier score, alert
rate, precision, default recall, and false-alert rate by gender, age band,
education, employment, marital status, and family responsibility. The JSON file
adds between-group rate ranges and the largest within-group difference between
mean score and observed synthetic default rate. These summaries are descriptive,
not statistical significance tests.
You can select a review-capacity profile with `--threshold-profile capacity_50`.
The results are descriptive checks on synthetic data, not a legal fairness
assessment or evidence that the model is appropriate for credit decisions.
