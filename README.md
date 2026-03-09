#  Détection de Pollueurs sur Twitter par Apprentissage Automatique (TP1)

**Cours :** INF7370 - Apprentissage Automatique  
**Auteur :** Imed   

Ce projet propose un pipeline complet de **Machine Learning** visant à classifier les utilisateurs de Twitter (humains légitimes vs bots/pollueurs) en se basant sur leurs comportements et les métadonnées de leurs profils.

---

##  Objectif du Projet
L'objectif est d'appliquer et de comparer les performances de **7 algorithmes de classification** sur des données tabulaires, en gérant deux scénarios principaux :
1.  **Classes équilibrées.**
2.  **Classes déséquilibrées** (ratio de 5% de pollueurs) pour tester la robustesse des modèles face à la réalité du terrain.

---

## Ingénierie des Caractéristiques (Feature Engineering)
En plus des métadonnées standards (nombre d'abonnés, ratio abonnements/abonnés, etc.), deux caractéristiques personnalisées ont été développées pour mieux capturer les comportements automatisés :

* **`hashtag_avg`** : Moyenne des hashtags utilisés par tweet.
* **`volatilite_abonnements`** : Variation entre le maximum et le minimum d'abonnements sur la période observée.

---

##  Modèles Évalués
Le pipeline évalue et compare les algorithmes suivants :
* Arbre de décision
* Bagging
* AdaBoost
* Gradient Boosting (GBoost)
* **XGBoost**
* **Forêts aléatoires (Random Forest)**
* Bayésien Naïf

> **Résultats Clés :** Les modèles basés sur les méthodes d'ensembles (XGBoost et Random Forest) ont démontré une supériorité écrasante avec une **AUC dépassant 0.98**. Ils maintiennent une excellente résilience face à la chute du taux de rappel (TP Rate) sur les données déséquilibrées.

---

##  Structure du Projet
Voici l'arborescence requise pour le bon fonctionnement du pipeline :

```text
TP1_INF7370/
│
├── data/                          <-- CRÉEZ CE DOSSIER (CRUCIAL)
│   └── [Nom_de_votre_dataset].csv <-- PLACEZ VOTRE BASE DE DONNÉES ICI
│
├── Features.py                    <-- Logique d'extraction des caractéristiques brutes.
├── main.py                        <-- Génération du dataset complet.
├── preprocessing.py               <-- Nettoyage, gestion des valeurs manquantes et normalisation (Z-score).
├── train.py                       <-- Entraînement et métriques (classes équilibrées).
├── train_imbalanced.py            <-- Test de robustesse (5% de pollueurs).
├── run_pipeline.py                <-- Orchestrateur (lancement de bout en bout).
└── README.md                      <-- Documentation du projet.
```

---

##  Guide d'Exécution

### 1. Prérequis et installation
Assurez-vous d'avoir Python 3.x installé. Installez les bibliothèques requises avec la commande suivante :

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### 2. Préparation des données
**Attention :** La base de données brute **DOIT** absolument être placée dans un dossier nommé `data` à la racine de votre projet, comme illustré dans l'arborescence ci-dessus.

### 3. Exécution du code
Ouvrez votre terminal, naviguez jusqu'au dossier racine de votre projet, puis lancez le script orchestrateur :

```bash
python run_pipeline.py
```

---

## Résultats Attendus
Une fois le pipeline terminé avec succès, vous obtiendrez :

1.  **Fichiers de données (.csv) :** La génération de deux fichiers nettoyés et indexés, prêts pour l'entraînement et l'analyse (`dataset_complet.csv` et `dataset_final_pret.csv`).
2.  **Dans le terminal :** L'affichage des tableaux de performances (F-Mesure, AUC, etc.).
3.  **Fichiers images (.png) :**
    * Courbes ROC (comparaison équilibré vs déséquilibré).
    * Graphiques d'importance des caractéristiques (*Feature Importance*).
    * Matrices de confusion comparatives.


**Note :** Le code est aussi mise sur mon GitHub https://github.com/iamimed/Detection-Pollueurs-Twitter-ML
