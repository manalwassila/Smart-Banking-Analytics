"""Smart Banking Analytics - Semaine 3
Nettoyage contrôlé, contrôles d'intégrité et feature engineering.
Exécuter depuis la racine du projet après extraction du ZIP dans data/raw/.
Aucune ligne n'est supprimée automatiquement pour les valeurs extrêmes.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

FILES = ['clients','financial_profiles','credit_applications','credit_history','repayments','bureau_monthly','assets','insurance','decision_scenarios']
tables = {}
for name in FILES:
    path = RAW / f'{name}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Fichier introuvable : {path}')
    df = pd.read_csv(path)
    # Suppression uniquement des doublons de lignes strictement identiques.
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f'{name}: {before:,} -> {len(df):,} lignes; doublons exacts retirés: {before-len(df)}')
    tables[name] = df

# Contrôle des identifiants principaux : SK_ID_CURR est unique dans ces tables.
for name in ['clients','financial_profiles','credit_applications','assets','insurance','decision_scenarios']:
    df = tables[name]
    if df['SK_ID_CURR'].duplicated().any():
        raise ValueError(f'Clé SK_ID_CURR non unique dans {name}')

# Contrôle des clés étrangères des tables enfant.
client_ids = set(tables['clients']['SK_ID_CURR'])
for name in ['financial_profiles','credit_applications','credit_history','repayments','bureau_monthly','assets','insurance','decision_scenarios']:
    orphan = ~tables[name]['SK_ID_CURR'].isin(client_ids)
    print(f'{name}: références client sans parent = {int(orphan.sum())}')

# Normalisation des dates connues.
for name, cols in {
    'credit_applications':['APPLICATION_DATE'],
    'credit_history':['CREDIT_START_DATE'],
    'repayments':['DUE_DATE','PAYMENT_DATE'],
    'bureau_monthly':['MONTH_DATE']
}.items():
    for col in cols:
        tables[name][col] = pd.to_datetime(tables[name][col], errors='coerce')

# Le dataset comporte 18 009 lignes avec au moins une valeur manquante dans
# decision_scenarios : ces absences sont conservées et documentées, pas imputées ici.

# Construction d'une base analytique de demande : variables disponibles à la date de demande.
apps = tables['credit_applications'].copy()
fin = tables['financial_profiles'].copy()
base = apps.merge(fin, on='SK_ID_CURR', how='left', validate='one_to_one')
# Champs d'identification/décision exclus des features. TARGET est la cible seulement.
excluded = {'SK_ID_CURR','APPLICATION_DATE','TARGET','TARGET_LABEL'}
feature_cols = [c for c in base.columns if c not in excluded]
X = base[feature_cols].copy()
y = base['TARGET'].astype('int8')

# Vérification temporelle : les tables de remboursements et de bureau mensuel
# ne sont PAS jointes directement. Elles doivent être agrégées en filtrant
# strictement les événements antérieurs ou égaux à APPLICATION_DATE.
base.to_csv(OUT / 'applications_financial_base.csv', index=False)
X.to_csv(OUT / 'X_candidate.csv', index=False)
y.to_frame('TARGET').to_csv(OUT / 'y_target.csv', index=False)

# Résumé qualité après contrôles.
summary = []
for name, df in tables.items():
    summary.append({'table':name,'rows':len(df),'columns':len(df.columns),
                    'missing_cells':int(df.isna().sum().sum()),
                    'exact_duplicate_rows':int(df.duplicated().sum())})
pd.DataFrame(summary).to_csv(OUT / 'quality_summary.csv', index=False)
print('\nExports créés dans', OUT)
print('Base candidate:', base.shape, '| X:', X.shape, '| y:', y.shape)
print('Taux TARGET=1:', round(y.mean()*100, 2), '%')
