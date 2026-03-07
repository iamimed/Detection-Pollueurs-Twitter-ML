import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Chargement du dataset qu'on a généré
df = pd.read_csv("dataset_complet.csv")

# --- SUPPRESSION DES DOUBLONS ---
# On supprime les lignes identiques basées sur l'identifiant unique 
df.drop_duplicates(subset=['UserID'], inplace=True)

# --- TRAITEMENT DES VALEURS MANQUANTES ---
# On sélectionne les colonnes numériques (on exclut les dates et le Label ET le UserID)
cols_to_fix = df.select_dtypes(include=['number']).columns.drop(['Label', 'UserID'])

# On remplace les NaN par la médiane de chaque colonne
for col in cols_to_fix:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# --- NORMALISATION (Z-SCORE) ---
# Le Z-score transforme les données pour avoir une moyenne de 0 et un écart-type de 1
scaler = StandardScaler()

# On applique la normalisation uniquement sur les caractéristiques d'entrée (X)
df[cols_to_fix] = scaler.fit_transform(df[cols_to_fix])
# --- SAUVEGARDE FINALE ---
df.to_csv("dataset_final_pret.csv", index=False)
print("Données nettoyées, normalisées et sauvegardées dans 'dataset_final_pret.csv'.")