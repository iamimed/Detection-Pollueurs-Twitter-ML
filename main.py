from Features import prepare_dataset
import pandas as pd

def main():
    data_folder = "data"
    
    # 1. Traitement des POLLUEURS (Spams)
    print("--- Analyse des Pollueurs ---")
    df_polluters = prepare_dataset(
        profile_path = f"{data_folder}/content_polluters.txt",
        tweets_path = f"{data_folder}/content_polluters_tweets.txt",
        followings_path = f"{data_folder}/content_polluters_followings.txt",
        label = 1
    )
    
    # 2. Traitement des LÉGITIMES (Humains)
    print("\n--- Analyse des Utilisateurs Légitimes ---")
    df_legit = prepare_dataset(
        profile_path = f"{data_folder}/legitimate_users.txt",
        tweets_path = f"{data_folder}/legitimate_users_tweets.txt",
        followings_path = f"{data_folder}/legitimate_users_followings.txt",
        label = 0
    )
    
    # 3. Fusion et Sauvegarde
    print("\nFusion des datasets...")
    dataset_final = pd.concat([df_polluters, df_legit], ignore_index=True)
    dataset_final.drop_duplicates(subset=['UserID'], inplace=True)
    dataset_final.to_csv("dataset_complet.csv", index=False)
    
    print(f"Terminé ! Fichier 'dataset_complet.csv' généré.")

    # --- Nettoyage final ---
# Supprimer les doublons sur l'ID utilisateur pour éviter de fausser l'IA
    dataset_final.drop_duplicates(subset=['UserID'], inplace=True)

# Réinitialiser l'index pour avoir une suite propre
    dataset_final.reset_index(drop=True, inplace=True)

# Configuration pour voir toutes les colonnes dans la console
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- APERÇU DU DATASET FINAL ---")
    print(dataset_final.head(10))

    print("\n--- STATISTIQUES DES CARACTÉRISTIQUES ---")
    print(dataset_final.describe())    

if __name__ == "__main__":
    main()


