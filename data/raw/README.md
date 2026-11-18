# Smart Banking Morocco — Dataset synthétique enrichi (v2.0)

## Présentation
Dataset **entièrement synthétique** destiné au PFA Smart Banking Analytics. Il représente un scénario pédagogique de risque de crédit au Maroc. Il ne contient aucune donnée bancaire réelle, aucun nom réel, CIN, adresse ou numéro de compte réel.

## Volume
- 50 000 clients et 50 000 demandes principales de crédit
- 82 500 lignes d'historique de crédit
- 600 000 lignes d'échéances/remboursements
- 300 000 observations bancaires mensuelles
- 50 000 profils d'actifs, 50 000 profils d'assurance et 50 000 scénarios de décision

## Fichiers
| Fichier | Contenu |
|---|---|
| `clients.csv` | Profil démographique/professionnel et composition familiale |
| `financial_profiles.csv` | Revenus, dépenses, engagements, épargne et indicateurs de capacité financière |
| `credit_applications.csv` | Demande principale et cible `TARGET` |
| `credit_history.csv` | Historique des crédits |
| `repayments.csv` | Échéances et remboursements |
| `bureau_monthly.csv` | Comportement bancaire mensuel |
| `assets.csv` | Patrimoine immobilier et autres actifs financiers |
| `insurance.csv` | Protections contractuelles vie et assurance liée au crédit |
| `decision_scenarios.csv` | Scénarios pédagogiques de décision : intégral, partiel ou rejet |
| `schema.sql` | Proposition de schéma PostgreSQL |
| `dataset_summary.json` | Résumé des volumes et enrichissements |

Toutes les tables sont reliées par `SK_ID_CURR`.

## Enrichissements ajoutés
### Famille
`NUMBER_OF_CHILDREN`, `DEPENDENT_CHILDREN`, `OTHER_DEPENDENTS`, `HAS_FINANCIAL_DEPENDENTS`, `FAMILY_RESPONSIBILITY_LEVEL`.

### Situation financière
`MONTHLY_FAMILY_EXPENSES_MAD`, `ADDITIONAL_MONTHLY_INCOME_MAD`, `HOUSEHOLD_MONTHLY_INCOME_MAD`, `TOTAL_MONTHLY_COMMITMENTS_MAD`, `DISPOSABLE_INCOME_MAD`, `HOUSEHOLD_EXPENSE_RATIO`, `SAVINGS_TO_MONTHLY_INCOME`.

### Patrimoine
`OWNS_PROPERTY`, `NUMBER_OF_PROPERTIES`, `PROPERTY_TYPE`, `TOTAL_PROPERTY_VALUE_MAD`, `PROPERTY_FULLY_PAID`, `OUTSTANDING_MORTGAGE_MAD`, `MONTHLY_RENTAL_INCOME_MAD`, `OTHER_FINANCIAL_ASSETS_MAD`, `NET_PROPERTY_VALUE_MAD`.

### Assurance
`HAS_LIFE_INSURANCE`, `HAS_LOAN_PROTECTION_INSURANCE`, montants de couverture et primes mensuelles. **Aucune donnée médicale, aucun diagnostic, ni état de santé n'est inclus.** La présence ou l'absence d'assurance ne doit pas être utilisée seule pour pénaliser un demandeur.

## Cible ML
`TARGET` dans `credit_applications.csv` :
- `0` : aucun défaut observé dans le scénario synthétique
- `1` : défaut observé dans le scénario synthétique

La cible reste binaire car elle représente l'événement historique de défaut. Cela ne signifie pas que la décision de financement doit être binaire. Le moteur de décision peut recommander une acceptation intégrale, une acceptation partielle ou un rejet.

## Scénarios de décision — attention
`decision_scenarios.csv` contient des **scénarios illustratifs générés par une règle synthétique** à partir de variables financières et de signaux de comportement. `DEMO_RISK_SCORE` n'est **pas** une probabilité de défaut issue d'un modèle ML entraîné et calibré. Ces champs servent à prototyper le moteur de décision et l'interface.

Le scénario applique une logique indicative :
- risque faible/modéré : financement intégral possible si la capacité financière le permet ;
- risque modéré/élevé : financement partiel possible, avec taux indicatif majoré lorsque cela reste abordable ;
- risque très élevé ou capacité de remboursement insuffisante : rejet recommandé.

Les seuils, taux, ratios et règles sont fictifs, non réglementaires et non validés par une banque. La capacité de remboursement doit primer : une hausse du taux ne doit jamais rendre une mensualité inabordable. Le statut `À valider par un analyste` rappelle que la décision finale est humaine.

## Bonnes pratiques ML et prévention des fuites de données
1. Pour prédire le risque au moment de la demande, n'utilisez pas les remboursements postérieurs à `APPLICATION_DATE`.
2. Agrégez l'historique et le bureau uniquement à partir des informations disponibles avant la date de demande.
3. N'utilisez pas `TARGET_LABEL`, `TARGET`, ni les colonnes de `decision_scenarios.csv` comme variables prédictives du modèle de défaut.
4. `decision_scenarios.csv` est une sortie de démonstration du moteur, pas une table de features ML.
5. Faites le split train/validation/test avant l'apprentissage des transformations. Imputation, encodage et éventuel rééchantillonnage doivent être ajustés sur le train uniquement.
6. Évaluez notamment Recall, Precision, F1, ROC-AUC, PR-AUC et matrice de confusion, en accordant une attention particulière au rappel de la classe défaut.
7. Contrôlez les biais et la pertinence des variables. L'âge, le genre, la situation matrimoniale, le nombre d'enfants et la propriété immobilière peuvent soulever des enjeux d'équité : ne les utilisez pas automatiquement pour décider sans justification, validation juridique et analyse de fairness.
8. Les données de santé sont exclues. Les assurances décrivent uniquement des protections contractuelles.

## Unités et limites
- Montants monétaires en dirhams marocains (MAD).
- Données générées artificiellement : les performances ML et les scénarios de décision ne prouvent pas la performance ou la conformité d'un système bancaire réel.
- Prototype d'aide à la décision uniquement, jamais d'acceptation/refus automatique en production.
