import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

# Importation des 7 algorithmes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, RandomForestClassifier)
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def plot_f_measure(df_results):
    """Génère un graphique à barres comparant la F-Mesure de chaque modèle."""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_results)))
    plt.bar(df_results['Algorithme'], df_results['F-Mesure'], color=colors)
    plt.title('Comparaison de la F-Mesure par Algorithme (Classe Pollueurs)')
    plt.ylabel('Score F-Mesure')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("comparaison_f_mesure.png", dpi=300)
    print("Graphique F-Mesure sauvegardé sous 'comparaison_f_mesure.png'")

def plot_feature_importance(model, feature_names):
    """Génère un graphique montrant l'importance des 12 caractéristiques."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 7))
    plt.title("Importance des caractéristiques (Modèle Forêts aléatoires)")
    plt.bar(range(len(importances)), importances[indices], color='teal', align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel("Score d'importance")
    plt.tight_layout()
    plt.savefig("importance_features.png", dpi=300)
    print("Graphique de l'importance des caractéristiques sauvegardé sous 'importance_features.png'")

def run_comparative_analysis(filepath="dataset_final_pret.csv"):
    # 1. Préparation des données
    df = pd.read_csv(filepath)
    X = df.drop(columns=['UserID', 'CreatedAt', 'CollectedAt', 'Label'], errors='ignore')
    y = df['Label']

    # Séparation 80% apprentissage / 20% test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 2. Initialisation des modèles
    models = {
        "Arbre de décision": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "GBoost": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
        "Forêts aléatoires": RandomForestClassifier(random_state=42),
        "Bayésien Naïf": GaussianNB()
    }

    results = []
    plt.figure(figsize=(10, 8))

    # 3. Entraînement et évaluation
    for name, model in models.items():
        print(f"Entraînement de {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] 
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        tp_rate = tp / (tp + fn) 
        fp_rate = fp / (fp + tn) 
        f_measure = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Algorithme": name,
            "TP Rate (Pollueurs)": round(tp_rate, 4),
            "FP Rate (Pollueurs)": round(fp_rate, 4),
            "F-Mesure": round(f_measure, 4),
            "AUC": round(auc, 4)
        })
        
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr_curve, tpr_curve, label=f"{name} (AUC = {auc:.3f})")

    # 4. Affichage du Tableau et Génération des Graphiques
    df_results = pd.DataFrame(results)
    print("\n=== RÉSULTATS COMPARATIFS ===")
    try:
        print(df_results.to_markdown(index=False))
    except ImportError:
        print(df_results.to_string(index=False))
    
    # Graphique ROC
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
    plt.xlabel('Taux de Faux Positifs (FP Rate)')
    plt.ylabel('Taux de Vrais Positifs (TP Rate)')
    plt.title('Courbes ROC des différents algorithmes (Détection de Pollueurs)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig("courbe_roc_comparative.png", dpi=300, bbox_inches='tight')
    print("\nGraphique ROC sauvegardé sous 'courbe_roc_comparative.png'")

    # Graphique F-Mesure
    plot_f_measure(df_results)

    # Graphique Importance des Features (via Random Forest)
    plot_feature_importance(models["Forêts aléatoires"], X.columns)

if __name__ == "__main__":
    run_comparative_analysis()