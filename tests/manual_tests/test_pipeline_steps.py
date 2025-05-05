import pandas as pd
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH pour permettre les imports relatifs
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Importer les fonctions des différents modules
# Assumer que clean_data existe dans data_preparation.py
try:
    from utils.data_preparation import clean_data
except ImportError:
    print("ERROR: Could not import clean_data from utils.data_preparation.py. Ensure the file and function exist.")
    # Créer une fonction placeholder si elle manque pour permettre au script de continuer (mais échouera probablement)
    def clean_data(df):
        print("WARNING: Using placeholder clean_data function.")
        return df

from utils.feature_engineering import apply_feature_pipeline
from utils.labeling import build_labels # Utilise le placeholder créé

def run_manual_pipeline_test(input_csv_path: str = "data/raw/eth_raw.csv"):
    """
    Exécute manuellement les étapes du pipeline de transformation sur un fichier CSV
    et effectue des vérifications basiques.
    """
    print(f"--- Starting Manual Pipeline Test ---")
    input_path = Path(input_csv_path)

    # --- 1. Charger les données ---
    print(f"\n[Step 1] Loading raw data from: {input_path}")
    if not input_path.exists():
        print(f"ERROR: Input file not found at {input_path}. Please generate it first (e.g., using api_manager.py).")
        return
    try:
        df = pd.read_csv(input_path, parse_dates=['timestamp'], index_col='timestamp')
        print(f"Loaded DataFrame shape: {df.shape}")
        assert len(df) > 0, "Loaded DataFrame is empty."
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return

    # --- 2. Nettoyer les données ---
    print("\n[Step 2] Cleaning data...")
    try:
        df_cleaned = clean_data(df.copy()) # Utiliser une copie
        print(f"Cleaned DataFrame shape: {df_cleaned.shape}")
        # Vérification du nombre de lignes (ajuster le seuil si nécessaire)
        min_rows_threshold = 1000
        assert len(df_cleaned) > min_rows_threshold, f"Expected > {min_rows_threshold} rows after cleaning, found {len(df_cleaned)}"
        print(f"Assertion Passed: Rows > {min_rows_threshold} ({len(df_cleaned)})")
    except Exception as e:
        print(f"ERROR during clean_data: {e}")
        return

    # --- 3. Appliquer Feature Engineering (sans LLM) ---
    print("\n[Step 3] Applying feature engineering (no LLM)...")
    try:
        df_features = apply_feature_pipeline(df_cleaned.copy(), include_llm=False)
        print(f"Features DataFrame shape: {df_features.shape}")
        assert len(df_features) > 0, "DataFrame became empty after feature engineering."
        # La vérification > 1000 lignes est plus pertinente ici car des NaNs sont supprimés
        assert len(df_features) > min_rows_threshold, f"Expected > {min_rows_threshold} rows after feature engineering, found {len(df_features)}"
        print(f"Assertion Passed: Rows > {min_rows_threshold} ({len(df_features)})")
        print(f"Columns after features: {df_features.columns.tolist()}")
    except Exception as e:
        print(f"ERROR during apply_feature_pipeline: {e}")
        return

    # --- 4. Construire les Labels (avec placeholder) ---
    print("\n[Step 4] Building labels (using placeholder)...")
    try:
        df_final = build_labels(df_features.copy()) # Utilise le placeholder
        final_shape = df_final.shape
        final_columns = df_final.columns.tolist()
        print(f"Final DataFrame shape: {final_shape}")
        print(f"Final Columns ({len(final_columns)}): {final_columns}")

        # Vérification du nombre de lignes final
        assert len(df_final) > min_rows_threshold, f"Expected > {min_rows_threshold} rows after labeling, found {len(df_final)}"
        print(f"Assertion Passed: Rows > {min_rows_threshold} ({len(df_final)})")

        # Vérification du nombre de colonnes final (cible = 38)
        expected_cols = 38
        assert len(final_columns) == expected_cols, f"Expected exactly {expected_cols} columns, found {len(final_columns)}"
        print(f"Assertion Passed: Columns == {expected_cols} ({len(final_columns)})")

    except Exception as e:
        print(f"ERROR during build_labels: {e}")
        return

    print("\n--- Manual Pipeline Test Completed Successfully ---")
    # Optionnel: Sauvegarder le résultat pour inspection
    # output_path = "data/processed/manual_test_output.parquet"
    # print(f"Saving final DataFrame for inspection to {output_path}")
    # df_final.to_parquet(output_path)

if __name__ == "__main__":
    # Créer le dossier s'il n'existe pas
    manual_test_dir = Path("tests/manual_tests")
    manual_test_dir.mkdir(parents=True, exist_ok=True)
    # Exécuter le test
    run_manual_pipeline_test()
