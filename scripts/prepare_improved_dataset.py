#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour pru00e9parer un dataset amélioré avec normalisation robuste et u00e9quilibrage des classes.
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_improved_dataset(input_path, output_path=None, test_size=0.2, random_state=42):
    """
    Pru00e9pare un dataset amélioré avec normalisation robuste et u00e9quilibrage des classes.
    
    Args:
        input_path: Chemin vers le dataset d'entru00e9e
        output_path: Chemin vers le dataset de sortie (par du00e9faut: input_path + '_normalized.csv')
        test_size: Proportion des donnu00e9es u00e0 utiliser pour le test
        random_state: Graine alu00e9atoire pour la reproductibilitu00e9
    """
    # Valider le chemin de sortie
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('')) + '_normalized.csv'
    
    # Charger le dataset
    logger.info(f"Chargement du dataset depuis {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Convertir les colonnes de type datetime en string pour u00e9viter les problu00e8mes de conversion
    for col in df.select_dtypes(include=['datetime64']).columns:
        df[col] = df[col].astype(str)
    
    # 1. Identifier les diffu00e9rents types de colonnes
    logger.info("Identification des diffu00e9rents types de colonnes")
    
    # Colonnes de prix et volume (u00e0 normaliser avec RobustScaler)
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Colonnes techniques (u00e0 normaliser avec StandardScaler)
    technical_cols = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'volume', 'market_regime', 'level_sl', 'level_tp', 
        'instrument_type', 'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
    ] and not col.startswith('llm_') and not col.startswith('mcp_')]
    
    # Colonnes LLM (du00e9ju00e0 normalisées)
    llm_cols = [col for col in df.columns if col.startswith('llm_')]
    
    # Colonnes MCP (u00e0 normaliser avec StandardScaler)
    mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
    
    # Colonnes HMM (probabilitu00e9s du00e9ju00e0 normalisées, ru00e9gime u00e0 garder tel quel)
    hmm_prob_cols = ['hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    hmm_regime_cols = ['hmm_regime']
    
    # Colonnes cibles
    target_cols = ['market_regime', 'level_sl', 'level_tp']
    
    # Autres colonnes
    other_cols = ['instrument_type']
    
    # 2. Normalisation des donnu00e9es
    logger.info("Normalisation des donnu00e9es")
    
    # Sauvegarder l'index pour le restaurer plus tard
    index_df = df.index
    
    # Exclure les colonnes de type datetime et object de la normalisation
    exclude_cols = ['timestamp', 'symbol', 'split']
    
    # Filtrer les colonnes techniques pour exclure les colonnes non numériques
    technical_cols = [col for col in technical_cols if col not in exclude_cols and df[col].dtype != 'object' and not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Normaliser les colonnes de prix et volume avec RobustScaler
    price_scaler = RobustScaler()
    if len(price_cols) > 0 and all(col in df.columns for col in price_cols):
        df[price_cols] = price_scaler.fit_transform(df[price_cols])
        logger.info(f"Colonnes de prix et volume normalisées avec RobustScaler: {price_cols}")
    
    # Normaliser les colonnes techniques avec StandardScaler
    tech_scaler = StandardScaler()
    if len(technical_cols) > 0:
        df[technical_cols] = tech_scaler.fit_transform(df[technical_cols])
        logger.info(f"Colonnes techniques normalisées avec StandardScaler: {len(technical_cols)} colonnes")
    
    # Normaliser les colonnes MCP avec StandardScaler
    mcp_scaler = StandardScaler()
    if len(mcp_cols) > 0:
        # Filtrer les colonnes MCP pour exclure les colonnes non numériques
        mcp_cols = [col for col in mcp_cols if col not in exclude_cols and df[col].dtype != 'object' and not pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(mcp_cols) > 0:
            df[mcp_cols] = mcp_scaler.fit_transform(df[mcp_cols])
            logger.info(f"Colonnes MCP normalisées avec StandardScaler: {len(mcp_cols)} colonnes")
    
    # 3. Normalisation des cibles SL/TP
    logger.info("Normalisation des cibles SL/TP")
    sl_tp_cols = ['level_sl', 'level_tp']
    sl_tp_scaler = RobustScaler()
    if all(col in df.columns for col in sl_tp_cols):
        # Sauvegarder les scalers pour la conversion inverse lors de l'u00e9valuation
        sl_tp_values = df[sl_tp_cols].values
        df[sl_tp_cols] = sl_tp_scaler.fit_transform(sl_tp_values)
        logger.info(f"Colonnes SL/TP normalisées avec RobustScaler: {sl_tp_cols}")
    
    # 4. Conversion de instrument_type en entier
    logger.info("Conversion de instrument_type en entier")
    if 'instrument_type' in df.columns:
        instrument_mapping = {'ETH': 0, 'BTC': 1, 'USDT': 2, 'eth': 0, 'btc': 1, 'usdt': 2}
        df['instrument_type'] = df['instrument_type'].map(lambda x: instrument_mapping.get(x, 0))
        logger.info("Colonne instrument_type convertie en entier")
    
    # 5. Diviser en ensembles d'entraînement et de test
    logger.info("Division en ensembles d'entraînement et de test")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    logger.info(f"Ensemble d'entraînement: {len(train_df)} lignes, Ensemble de test: {len(test_df)} lignes")
    
    # 6. u00c9quilibrer les classes pour market_regime (uniquement sur l'ensemble d'entraînement)
    logger.info("u00c9quilibrage des classes pour market_regime")
    if 'market_regime' in train_df.columns:
        # Compter les classes avant u00e9quilibrage
        class_counts_before = train_df['market_regime'].value_counts()
        logger.info(f"Distribution des classes avant u00e9quilibrage: {class_counts_before.to_dict()}")
        
        # Identifier les features et la cible pour SMOTE
        X_train = train_df.drop(columns=['market_regime'])
        y_train = train_df['market_regime']
        
        # Appliquer SMOTE pour u00e9quilibrer les classes
        try:
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            # Reconstruire le DataFrame u00e9quilibru00e9
            train_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
            train_df['market_regime'] = y_train_resampled
            
            # Compter les classes apru00e8s u00e9quilibrage
            class_counts_after = train_df['market_regime'].value_counts()
            logger.info(f"Distribution des classes apru00e8s u00e9quilibrage: {class_counts_after.to_dict()}")
        except Exception as e:
            logger.warning(f"Erreur lors de l'u00e9quilibrage des classes avec SMOTE: {e}")
            logger.warning("Poursuite sans u00e9quilibrage des classes")
    
    # 7. Recombiner les ensembles d'entraînement et de test
    logger.info("Recombinaison des ensembles d'entraînement et de test")
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    combined_df = pd.concat([train_df, test_df], axis=0)
    logger.info(f"Dataset combiné: {len(combined_df)} lignes")
    
    # 8. Sauvegarder les scalers pour une utilisation ultu00e9rieure
    logger.info("Sauvegarde des scalers")
    scalers = {
        'price_scaler': price_scaler,
        'tech_scaler': tech_scaler,
        'mcp_scaler': mcp_scaler,
        'sl_tp_scaler': sl_tp_scaler
    }
    
    scalers_dir = Path(output_path).parent / 'scalers'
    scalers_dir.mkdir(parents=True, exist_ok=True)
    scalers_path = scalers_dir / 'scalers.npz'
    
    np.savez(scalers_path, **{k: v for k, v in scalers.items()})
    logger.info(f"Scalers sauvegardés dans {scalers_path}")
    
    # 9. Sauvegarder le dataset normalisé
    logger.info(f"Sauvegarde du dataset normalisé dans {output_path}")
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Dataset normalisé sauvegardé avec {len(combined_df)} lignes et {len(combined_df.columns)} colonnes")
    
    # 10. Sauvegarder les métadonnées
    metadata = {
        'price_cols': price_cols,
        'technical_cols': technical_cols,
        'llm_cols': llm_cols,
        'mcp_cols': mcp_cols,
        'hmm_prob_cols': hmm_prob_cols,
        'hmm_regime_cols': hmm_regime_cols,
        'target_cols': target_cols,
        'other_cols': other_cols
    }
    
    metadata_path = Path(output_path).parent / 'metadata.npz'
    np.savez(metadata_path, **{k: np.array(v) for k, v in metadata.items()})
    logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
    
    return combined_df, scalers, metadata

def main():
    parser = argparse.ArgumentParser(description='Pru00e9pare un dataset amélioré avec normalisation robuste et u00e9quilibrage des classes.')
    parser.add_argument('--input', type=str, required=True, help='Chemin vers le dataset d\'entru00e9e')
    parser.add_argument('--output', type=str, help='Chemin vers le dataset de sortie')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proportion des donnu00e9es u00e0 utiliser pour le test')
    parser.add_argument('--random-state', type=int, default=42, help='Graine alu00e9atoire pour la reproductibilitu00e9')
    
    args = parser.parse_args()
    
    prepare_improved_dataset(args.input, args.output, args.test_size, args.random_state)

if __name__ == "__main__":
    main()
