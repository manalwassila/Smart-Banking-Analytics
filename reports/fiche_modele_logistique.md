# Fiche de développement du modèle logistique

**Projet :** Smart Banking Analytics

**Version :** prototype entraîné le 25 novembre 2026
**Cible :** `TARGET` (défaut synthétique)

## Décision de modélisation

La régression logistique reste le modèle retenu pour le prototype. Le pipeline
applique une imputation médiane et une standardisation aux variables numériques,
une imputation modale et un encodage one-hot aux variables catégorielles, puis
une régression logistique pondérée (`class_weight="balanced"`). La recherche
sur cinq plis a retenu `C = 10` selon l'average precision.

Cette fiche décrit l'étape de développement après le compte rendu de la semaine
4. Le seuil 0,36 présenté dans ce compte rendu provenait de l'essai initial non
calibré; le pipeline développé ici applique une calibration sigmoïde et utilise
le nouveau seuil 0,11 choisi sur des scores hors pli.

Le prétraitement est ajusté dans le pipeline au sein de chaque pli. `TARGET`,
`TARGET_LABEL` et les champs des scénarios de décision sont interdits parmi les
variables explicatives. Les tables de remboursements et de bureau mensuel ne
sont pas jointes directement : elles demanderaient une agrégation strictement
antérieure à la date de demande.

## Calibration et seuil d'alerte

Une calibration sigmoïde a été retenue en comparant sigmoid et isotonic par
Brier score hors pli sur les données d'entraînement. Le score Brier obtenu est
de 0,136315 pour sigmoid et 0,136371 pour isotonic; l'écart est faible. Le seuil
0,11 maximise F2 sur des scores calibrés hors pli. Il favorise le rappel et
produit un volume élevé d'alertes. Il s'agit d'un réglage de prototype, pas d'un
seuil métier validé.

Pour comparer des capacités hypothétiques, le tableau ci-dessous retient le
seuil donnant le rappel le plus élevé sans dépasser le volume d'alertes indiqué.
Les estimations sont calculées sur les prédictions hors pli des 40 000 dossiers
d'entraînement et ramenées à 10 000 dossiers.

| Capacité d'examen | Seuil | Rappel | Précision | Alertes / 10 000 | Défauts manqués / 10 000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 % | 0,40 | 26,2 % | 49,5 % | 997 | 1 394 |
| 20 % | 0,27 | 44,0 % | 42,4 % | 1 958 | 1 057 |
| 30 % | 0,19 | 55,5 % | 35,7 % | 2 931 | 840 |
| 40 % | 0,16 | 63,5 % | 31,5 % | 3 802 | 689 |
| 50 % | 0,14 | 70,7 % | 28,2 % | 4 736 | 553 |
| 60 % | 0,13 | 75,5 % | 26,5 % | 5 385 | 463 |
| 70 % | 0,11 | 85,5 % | 23,1 % | 6 997 | 273 |
| 80 % | 0,10 | 90,7 % | 21,5 % | 7 959 | 175 |

Ces taux aident à cadrer la discussion avec l'équipe métier; ils ne constituent
pas une recommandation de capacité ou de seuil.

## Résultats observés

Le jeu comprend 50 000 demandes synthétiques, avec 18,88 % de défauts. Le
protocole a utilisé un partage stratifié de 40 000 dossiers d'entraînement et
10 000 dossiers de test. Le jeu de test a déjà été examiné dans les travaux
précédents de la semaine 4; les résultats ci-dessous sont donc exploratoires.
Une nouvelle période de test ou un jeu externe est requis avant de défendre une
estimation finale de généralisation.

| Mesure sur le test | Résultat |
| --- | ---: |
| Seuil d'alerte | 0,11 |
| Précision défaut | 23,3 % |
| Rappel défaut | 85,9 % |
| F1 défaut | 36,7 % |
| F2 défaut | 55,9 % |
| ROC-AUC | 0,713 |
| Average precision | 0,404 |
| Brier score | 0,136 |
| Log loss | 0,435 |
| Alertes | 6 961 / 10 000 (69,6 %) |

La matrice donne 1 622 vrais positifs, 266 faux négatifs, 5 339 faux positifs et
2 773 vrais négatifs. Le rappel élevé s'accompagne donc de nombreuses fausses
alertes. La sélection F2 ne prend pas en compte la capacité de traitement d'un
analyste ni le coût financier de chaque type d'erreur. Le tableau de sensibilité
des seuils permet de choisir un compromis opérationnel après définition de ces
contraintes.

La calibration a amélioré Brier et log loss sur ce jeu par rapport au score brut
de la logistique, mais cela ne valide pas la calibration hors de la distribution
synthétique. Le score ne doit pas être interprété comme une probabilité de défaut
réelle.

## Lecture des coefficients et erreurs

Les coefficients les plus élevés sont ceux du ratio dette/revenu et du ratio
crédit/revenu. Les deux sont positifs dans le modèle; leurs coefficients sont
respectivement 3,379 et 3,350 pour une hausse d'un écart-type d'entraînement.
Ces ratios sont corrélés et peuvent partager le signal appris. Les coefficients
indiquent des associations dans les données générées, pas des effets causaux.

Parmi les défauts manqués, les médianes du ratio dette/revenu (0,404) et du ratio
crédit/revenu (0,330) sont proches de celles des non-défauts correctement
classés (0,401 et 0,327). Le modèle a plus de difficulté lorsque les signaux
financiers se recouvrent. Les faux positifs ont une médiane dette/revenu de
0,663, ce qui illustre qu'un ratio élevé ne garantit pas qu'un défaut figure
dans la cible synthétique.

## Limites et usage

- Toutes les données sont synthétiques; la performance ne se transpose pas aux
  clients d'une banque sans validation externe.
- La calibration et le seuil ne sont valables que pour cette distribution
  générée.
- Les variables financières peuvent être corrélées, et les catégories peuvent
  véhiculer des relations propres au générateur de données.
- L'alerte est un indicateur d'analyse. Elle ne recommande pas une acceptation
  ou un refus automatique.
- Le modèle final de comparaison n'inclut pas XGBoost, car le paquet était
  indisponible et l'accès réseau empêchait son installation.

## Fichiers produits

- `models/logistic_regression_credit_risk.joblib` : pipeline calibré, colonnes
  attendues et seuil retenu.
- `reports/logistic_regression_training_metrics.json` : métriques, protocole,
  calibration et versions de bibliothèques.
- `reports/logistic_regression_test_predictions.csv` : scores et alertes sur le
  test synthétique.
- `reports/logistic_regression_threshold_sensitivity.csv` : compromis
  précision-rappel-seuil calculés sur les données d'entraînement hors pli.
- `reports/logistic_regression_coefficients.csv` : coefficients transformés.
- `reports/logistic_regression_calibration_bins.csv` : comparaison des scores
  moyens et taux de défaut observés par décile de score.
- `reports/logistic_regression_error_analysis.json` : profils synthétiques des
  faux positifs et faux négatifs.

Pour réentraîner ou régénérer les analyses, voir `models/README.md`.

## Démonstration du suivi de dérive

Le nouvel outil de suivi a comparé le lot de test synthétique (10 000 demandes)
au profil hors pli des données d'entraînement. Le PSI du score est de 0,00017;
le PSI le plus élevé parmi les variables est 0,00219 pour
`HOUSEHOLD_EXPENSE_RATIO`. Le lot ne contenait ni valeurs manquantes ni
catégories inconnues. Avec le profil de capacité `capacity_50`, le taux d'alerte
observé dans ce lot est de 47,18 %. Comme les deux lots proviennent du même
générateur synthétique, ce résultat démontre le fonctionnement du suivi sur un
lot comparable; il ne prouve pas la stabilité sur de nouvelles données réelles.

## Comparaison descriptive entre groupes

Un contrôle exploratoire supplémentaire compare les alertes et les erreurs du
jeu de test par groupe de clients. Pour le seuil `f2_max` (0,11), le taux de
fausses alertes varie de 44,1 % à 100 % entre les situations professionnelles
et de 57,8 % à 81,9 % entre les niveaux de responsabilité familiale. Ces écarts
sont calculés sur les données synthétiques du projet; ils ne constituent ni un
test statistique ni une mesure d'équité applicable à une banque réelle. Les
écarts de défauts générés par le dataset et les faibles effectifs de certaines
catégories influencent ces résultats.

Le tableau donne aussi le Brier score par groupe et la différence entre le score
moyen et le taux de défaut synthétique observé. Pour la situation professionnelle,
la plus grande différence absolue de score moyen est de 2,31 points de
pourcentage. Cette moyenne agrégée ne démontre pas que les scores sont calibrés
au niveau individuel ou sur une population future.

Le rapport détaille également les taux d'alerte, la précision et le rappel par
groupe. Les catégories de moins de 30 dossiers sont signalées et exclues des
résumés d'écarts. Avant tout usage décisionnel, ces résultats exigeraient des
données représentatives, une analyse métier et juridique, ainsi qu'une méthode
de validation adaptée; le prototype ne prend aucune décision d'octroi.
