import subprocess
import sys

def run_script(script_name):
    """Exécute un script Python et vérifie s'il y a des erreurs."""
    print(f"\n{'='*50}")
    print(f" Lancement de l'étape : {script_name}")
    print(f"{'='*50}\n")
    
    # sys.executable s'assure qu'on utilise le bon environnement Python (celui de ton VS Code)
    result = subprocess.run([sys.executable, script_name])
    
    # Si le script plante (code d'erreur différent de 0), on arrête tout
    if result.returncode != 0:
        print(f"\n ERREUR : Le script {script_name} a échoué. Arrêt du pipeline.")
        sys.exit(1)
        
    print(f"\n {script_name} terminé avec succès !")

if __name__ == "__main__":
    print("DÉMARRAGE DU PIPELINE DE MACHINE LEARNING...")
    
    # Étape 1 : Extraction des données (main.py utilise Features.py en interne)
    run_script("main.py")
    
    # Étape 2 : Nettoyage et Normalisation
    run_script("preprocessing.py")
    
    # Étape 3 : Entraînement des modèles et Graphiques
    run_script("train.py")
        
    # Étape 4 : Entraînement des modèles (Déséquilibré 5%)
    run_script("train_imbalanced.py") 
    
    print("\n PIPELINE COMPLET ! Tous les CSV et graphiques sont à jour.")