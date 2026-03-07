import pandas as pd
import numpy as np
import re

def parse_profiles(filepath):
    """
    Lit le fichier de profil (ex: content_polluters.txt).
    Format : Tab-separated, sans header.
    """
    cols = [
        'UserID', 'CreatedAt', 'CollectedAt', 'NumberOfFollowings', 
        'NumberOfFollowers', 'NumberOfTweets', 'LengthOfScreenName', 
        'LengthOfDescriptionInUserProfile'
    ]
    
    print(f"Lecture du fichier profils : {filepath}")
    df = pd.read_csv(filepath, sep='\t', names=cols, header=None, on_bad_lines='skip', quoting=3)
    
    # Conversion des dates
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], errors='coerce')
    df['CollectedAt'] = pd.to_datetime(df['CollectedAt'], errors='coerce')
    
    return df

def compute_tweet_features(filepath):
    """
    Lit le fichier des tweets et calcule les statistiques textuelles et temporelles.
    """
    cols = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']
    print(f"--> Lecture des tweets (Patientez...) : {filepath}")
    
    df = pd.read_csv(filepath, sep='\t', names=cols, header=None, on_bad_lines='skip', quoting=3)
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'], errors='coerce')
    df['Tweet'] = df['Tweet'].astype(str)
    
    # --- PRÉ-CALCUL DES CARACTÉRISTIQUES DE BASE ---
    df['has_url'] = df['Tweet'].str.contains(r'http[s]?://', regex=True)
    df['has_mention'] = df['Tweet'].str.contains(r'@\w+', regex=True)
    
    # =========================================================================
    # AJOUT #1 : PREMIÈRE CARACTÉRISTIQUE SUPPLÉMENTAIRE (HASHTAGS)
    # On compte le nombre exact de hashtags par tweet pour faire une moyenne
    # =========================================================================
    df['hashtag_count'] = df['Tweet'].str.count(r'#\w+')

    # Agrégation par utilisateur
    print("    Calcul des moyennes (URL/Mentions/Hashtags)...")
    grouped = df.groupby('UserID')
    
    features = pd.DataFrame()
    features['url_ratio'] = grouped['has_url'].mean()
    features['mention_ratio'] = grouped['has_mention'].mean()
    
    # Sauvegarde de notre Ajout #1 dans le DataFrame
    features['hashtag_avg'] = grouped['hashtag_count'].mean()
    
    # --- Calcul des écarts de temps (Time Gaps) ---
    print("    Calcul des écarts de temps (Peut être long)...")
    def get_time_gaps(group):
        if len(group) < 2:
            return pd.Series([0.0, 0.0], index=['mean_gap', 'max_gap'])
        
        sorted_dates = group.sort_values().diff().dropna()
        gaps = sorted_dates.dt.total_seconds() / 60  # en minutes
        
        if len(gaps) == 0:
            return pd.Series([0.0, 0.0], index=['mean_gap', 'max_gap'])
        return pd.Series([gaps.mean(), gaps.max()], index=['mean_gap', 'max_gap'])
    
    time_stats = grouped['CreatedAt'].apply(get_time_gaps).unstack()
    features = features.join(time_stats)
    
    return features

def compute_followings_features(filepath):
    """
    Lit le fichier d'historique des abonnements (followings).
    """
    print(f"--> Lecture de l'historique des abonnements : {filepath}")
    df_followings = pd.read_csv(filepath, sep='\t', header=None, names=['UserID', 'SeriesOfFollowings'])
    
    # =========================================================================
    # AJOUT #2 : DEUXIÈME CARACTÉRISTIQUE SUPPLÉMENTAIRE (VOLATILITÉ)
    # Calcul de l'écart entre le maximum et le minimum d'abonnements 
    # =========================================================================
    def calculer_volatilite(serie_texte):
        if pd.isna(serie_texte):
            return 0
        try:
            liste_nombres = [int(x) for x in str(serie_texte).split(',')]
            return max(liste_nombres) - min(liste_nombres)
        except ValueError:
            return 0
            
    df_followings['volatilite_abonnements'] = df_followings['SeriesOfFollowings'].apply(calculer_volatilite)
    
    # On retourne uniquement l'ID et notre nouvelle caractéristique
    return df_followings[['UserID', 'volatilite_abonnements']].set_index('UserID')

def prepare_dataset(profile_path, tweets_path, followings_path, label):
    """
    Fusionne les profils, les stats de tweets, et les stats d'abonnements pour créer les 12 features.
    """
    # 1. Charger les 3 sources de données
    df_profiles = parse_profiles(profile_path)
    df_tweet_stats = compute_tweet_features(tweets_path)
    df_followings_stats = compute_followings_features(followings_path)
    
    # 2. Fusion principale
    print(f"--> Fusion finale des données pour la classe '{label}'...")
    merged = pd.merge(df_profiles, df_tweet_stats, on='UserID', how='left')
    merged = pd.merge(merged, df_followings_stats, on='UserID', how='left')
    
    # Remplissage des valeurs manquantes par 0
    merged.fillna(0, inplace=True)
    
    # --- CALCUL DES DERNIÈRES CARACTÉRISTIQUES DE BASE ---
    
    # Durée de vie du compte (en jours)
    merged['AccountAge_Days'] = (merged['CollectedAt'] - merged['CreatedAt']).dt.days
    merged.loc[merged['AccountAge_Days'] < 1, 'AccountAge_Days'] = 1 # Éviter division par 0
    
    # Rapport following/followers
    merged['Following_Followers_Ratio'] = merged['NumberOfFollowings'] / (merged['NumberOfFollowers'] + 1)
    
    # Nombre moyen de tweets par jour
    merged['Tweets_Per_Day'] = merged['NumberOfTweets'] / merged['AccountAge_Days']
    
    # Ajout du label (ex: 1 pour pollueur, 0 pour légitime)
    merged['Label'] = label
    
    print(" Jeu de données prêt !")
    return merged