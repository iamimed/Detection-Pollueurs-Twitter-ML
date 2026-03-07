#  Détection de Pollueurs sur Twitter par Apprentissage Automatique

Ce projet propose un pipeline complet de Machine Learning visant à classifier les utilisateurs de Twitter (humains légitimes vs bots/pollueurs) en se basant sur leurs comportements et les métadonnées de leurs profils.

##  Objectif du Projet
L'objectif est d'appliquer et de comparer les performances de 7 algorithmes de classification sur des données tabulaires, en gérant des scénarios de classes équilibrées et déséquilibrées (ratio de 5% de pollueurs).

##  Ingénierie des Caractéristiques (Feature Engineering)
En plus des métadonnées standards (nombre d'abonnés, ratio abonnements/abonnés, etc.), deux caractéristiques personnalisées ont été développées pour mieux capturer les comportements automatisés :
* **`hashtag_avg`** : Moyenne des hashtags utilisés par tweet.
* **`volatilite_abonnements`** : Variation entre le maximum et le minimum d'abonnements sur la période observée.

##  Modèles Évalués
1. Arbre de décision
2. Bagging
3. AdaBoost
4. Gradient Boosting (GBoost)
5. XGBoost
6. Forêts aléatoires (Random Forest)
7. Bayésien Naïf

##  Résultats Clés
Les modèles basés sur les méthodes d'ensembles (comme **XGBoost** et **Random Forest**) ont démontré une supériorité écrasante avec une **AUC dépassant 0.98**. 
Face à des données fortement déséquilibrées, ces mêmes algorithmes ont maintenu leur résilience face à la chute du taux de rappel (TP Rate).

##  Structure du Code
* `Features.py` : Logique d'extraction des caractéristiques brutes.
* `main.py` : Génération du dataset complet.
* `preprocessing.py` : Nettoyage, gestion des valeurs manquantes et normalisation (Z-score).
* `train.py` : Entraînement des modèles et génération des métriques sur classes équilibrées.
* `train_imbalanced.py` : Test de robustesse sur un échantillon à 5% de pollueurs.
* `run_pipeline.py` : Orchestrateur permettant de lancer tout le processus de bout en bout.

##  Comment exécuter le pipeline complet
```bash
python run_pipeline.py