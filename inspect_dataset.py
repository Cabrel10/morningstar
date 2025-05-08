import pandas as pd

# Chemin vers ton dataset
file_path = './dataset/crypto_dataset_complet.parquet'

# Chargement
df = pd.read_parquet(file_path)

# Affichage des colonnes
print("Colonnes disponibles :")
for col in df.columns:
    print(" -", col)

# Affichage des premières lignes
print("\nAperçu des 5 premières lignes :")
print(df.head(5))
