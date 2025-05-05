#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# Charger le dataset enrichi
df = pd.read_parquet('data/enriched/enriched_dataset.parquet')

# Afficher les informations sur le dataset
print(f"Nombre de lignes: {len(df)}")
print(f"Nombre de colonnes: {len(df.columns)}")
print("\nListe des colonnes:")
for col in df.columns:
    print(f"- {col}")

# Afficher quelques statistiques de base
print("\nStatistiques de base:")
print(df.describe())

# Afficher les premières lignes
print("\nPremières lignes:")
print(df.head())
