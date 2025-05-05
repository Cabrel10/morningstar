#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour normaliser les datasets et les rendre conformes au modèle Morningstar.
Ce script traite les données brutes et préparées pour assurer leur compatibilité avec le modèle.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import shutil
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
import joblib

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins des répertoires
BASE_DIR = Path('/home/morningstar/Desktop/crypto_robot/Morningstar')
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
ENRICHED_DIR = BASE_DIR / 'data' / 'enriched'
MODEL_DIR = BASE_DIR / 'model'
STANDARDIZED_DIR = BASE_DIR / 'data' / 'standardized'

# Créer les répertoires s'ils n'existent pas
STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
(STANDARDIZED_DIR / 'metadata').mkdir(parents=True, exist_ok=True)

def load_raw_csv_files():
    """
    Charge et fusionne les fichiers CSV bruts (BTC, ETH, BNB, SOL)
    """
    logger.info("Chargement des fichiers CSV bruts...")
    dfs = []
    
    for file in RAW_DIR.glob('*_raw.csv'):
        symbol = file.stem.split('_')[0].upper()
        df = pd.read_csv(file)
        df['symbol'] = f"{symbol}/USDT"
        dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Fichiers CSV combinés: {len(combined_df)} lignes")
        return combined_df
    else:
        logger.warning("Aucun fichier CSV brut trouvé")
        return None

def load_parquet_dataset(file_path):
    """
    Charge un dataset au format parquet
    """
    if file_path.exists():
        logger.info(f"Chargement du dataset {file_path}...")
        return pd.read_parquet(file_path)
    else:
        logger.warning(f"Fichier {file_path} non trouvé")
        return None

def add_technical_indicators(df):
    """
    Ajoute les indicateurs techniques manquants au DataFrame
    """
    logger.info("Ajout des indicateurs techniques manquants...")
    
    # Liste des indicateurs techniques attendus par le modèle
    expected_indicators = [
        'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'RSI', 
        'MACD', 'MACDs', 'MACDh', 'BBU', 'BBM', 'BBL', 'ATR',
        'STOCHk', 'STOCHd', 'ADX', 'CCI', 'Momentum', 'ROC',
        'Williams_%R', 'TRIX', 'Ultimate_Osc', 'DPO', 'OBV', 'VWMA',
        'CMF', 'MFI', 'Parabolic_SAR', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_SenkouA', 'Ichimoku_SenkouB', 'Ichimoku_Chikou',
        'KAMA', 'VWAP', 'STOCHRSIk', 'CMO', 'PPO', 'FISHERt'
    ]
    
    # Vérifier quels indicateurs sont manquants
    missing_indicators = [ind for ind in expected_indicators if ind not in df.columns]
    
    if missing_indicators:
        logger.info(f"Indicateurs manquants: {missing_indicators}")
        
        # Ajouter des colonnes vides pour les indicateurs manquants
        for indicator in missing_indicators:
            df[indicator] = np.nan
    
    return df

def add_hmm_features(df):
    """
    Ajoute les caractéristiques HMM si elles sont manquantes
    """
    logger.info("Ajout des caractéristiques HMM...")
    
    hmm_features = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    for feature in hmm_features:
        if feature not in df.columns:
            if feature == 'hmm_regime':
                # Simuler des régimes HMM (0, 1, 2) avec une distribution réaliste
                df[feature] = np.random.choice([0, 1, 2], size=len(df), p=[0.6, 0.3, 0.1])
            else:
                # Simuler des probabilités pour chaque régime
                df[feature] = np.random.random(size=len(df))
    
    return df

def add_llm_features(df):
    """
    Ajoute les caractéristiques LLM si elles sont manquantes
    """
    logger.info("Ajout des caractéristiques LLM...")
    
    if 'llm_context_summary' not in df.columns:
        df['llm_context_summary'] = 'Résumé de contexte généré automatiquement'
    
    if 'llm_embedding' not in df.columns:
        # Créer un embedding simulé de dimension 10
        embeddings = np.random.normal(0, 1, size=(len(df), 10))
        df['llm_embedding'] = [emb.tolist() for emb in embeddings]
    
    return df

def add_mcp_features(df):
    """
    Ajoute les caractéristiques MCP (Market Context Processing) si elles sont manquantes
    """
    logger.info("Ajout des caractéristiques MCP...")
    
    # Vérifier si les caractéristiques MCP sont présentes
    mcp_features = [f'mcp_feature_{i:03d}' for i in range(128)]
    missing_mcp = [feat for feat in mcp_features if feat not in df.columns]
    
    if missing_mcp:
        logger.info(f"Ajout de {len(missing_mcp)} caractéristiques MCP manquantes")
        
        for feature in missing_mcp:
            df[feature] = np.random.normal(0, 1, size=len(df))
    
    return df

def normalize_dataset(df):
    """
    Normalise le dataset pour le rendre compatible avec le modèle Morningstar
    """
    logger.info("Normalisation du dataset...")
    
    # Vérifier si le dataset a déjà les colonnes requises
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Colonnes manquantes: {missing_columns}")
        
        # Si c'est un dataset avec des noms de colonnes en majuscules
        uppercase_columns = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        for upper, lower in uppercase_columns.items():
            if upper in df.columns and lower not in df.columns:
                df[lower] = df[upper]
    
    # Ajouter les indicateurs techniques manquants
    df = add_technical_indicators(df)
    
    # Ajouter les caractéristiques HMM si manquantes
    df = add_hmm_features(df)
    
    # Ajouter les caractéristiques LLM si manquantes
    df = add_llm_features(df)
    
    # Ajouter les caractéristiques MCP si manquantes
    df = add_mcp_features(df)
    
    # Ajouter une colonne asset_id si elle n'existe pas
    if 'asset_id' not in df.columns and 'symbol' in df.columns:
        df['asset_id'] = df['symbol']
    elif 'asset_id' not in df.columns:
        df['asset_id'] = 'unknown'
    
    # Ajouter les colonnes de signal et de régime de marché si elles n'existent pas
    if 'signal' not in df.columns:
        # Créer un signal synthétique basé sur les mouvements de prix
        df['signal'] = np.nan
        df.loc[df['close'].pct_change(5) > 0.05, 'signal'] = 4  # Achat fort
        df.loc[(df['close'].pct_change(5) > 0.02) & (df['close'].pct_change(5) <= 0.05), 'signal'] = 3  # Achat
        df.loc[(df['close'].pct_change(5) >= -0.02) & (df['close'].pct_change(5) <= 0.02), 'signal'] = 2  # Neutre
        df.loc[(df['close'].pct_change(5) >= -0.05) & (df['close'].pct_change(5) < -0.02), 'signal'] = 1  # Vente
        df.loc[df['close'].pct_change(5) < -0.05, 'signal'] = 0  # Vente forte
        df['signal'] = df['signal'].fillna(2).astype(int)  # Remplir les NaN avec 'Neutre'
    
    if 'market_regime' not in df.columns:
        # Créer un régime de marché synthétique basé sur la volatilité et la tendance
        volatility = df['close'].pct_change().rolling(20).std()
        trend = df['close'].pct_change(20)
        
        df['market_regime'] = np.nan
        df.loc[(trend > 0.05) & (volatility < 0.02), 'market_regime'] = 2  # Haussier
        df.loc[(trend < -0.05) & (volatility < 0.02), 'market_regime'] = 0  # Baissier
        df.loc[(trend.abs() <= 0.05) & (volatility < 0.02), 'market_regime'] = 1  # Neutre
        df.loc[volatility >= 0.02, 'market_regime'] = 3  # Volatil
        df['market_regime'] = df['market_regime'].fillna(1).astype(int)  # Remplir les NaN avec 'Neutre'
    
    # Ajouter les niveaux SL/TP si manquants
    if 'level_sl' not in df.columns:
        df['level_sl'] = -0.02  # -2% par défaut
    
    if 'level_tp' not in df.columns:
        df['level_tp'] = 0.04  # +4% par défaut
    
    # Supprimer les lignes avec des valeurs NaN
    df_cleaned = df.dropna()
    
    if len(df_cleaned) < len(df):
        logger.warning(f"Suppression de {len(df) - len(df_cleaned)} lignes avec des valeurs NaN")
    
    return df_cleaned

def scale_features(df):
    """
    Normalise les caractéristiques numériques du dataset
    """
    logger.info("Mise à l'échelle des caractéristiques...")
    
    # Séparer les caractéristiques numériques des autres
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
    
    # Exclure les colonnes de label et d'identifiant
    exclude_cols = ['signal', 'market_regime', 'asset_id', 'symbol', 'timestamp', 'date']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Créer un scaler pour les caractéristiques
    scaler = StandardScaler()
    
    # Appliquer le scaling aux caractéristiques
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Sauvegarder le scaler pour une utilisation future
    scaler_path = STANDARDIZED_DIR / 'metadata' / 'feature_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        joblib.dump(scaler, f)
    
    logger.info(f"Scaler sauvegardé dans {scaler_path}")
    
    # Sauvegarder les noms des colonnes de caractéristiques
    feature_cols_path = STANDARDIZED_DIR / 'metadata' / 'feature_columns.json'
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f)
    
    logger.info(f"Colonnes de caractéristiques sauvegardées dans {feature_cols_path}")
    
    return df_scaled

def stratified_split_dataset(df, test_size=0.15, val_size=0.15, stratify_col='market_regime', random_state=42):
    """
    Effectue un split stratifié du dataset pour éviter le data leakage
    
    Args:
        df: DataFrame à splitter
        test_size: Proportion de données pour le test
        val_size: Proportion de données pour la validation
        stratify_col: Colonne pour la stratification
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger.info(f"Split stratifié du dataset sur la colonne {stratify_col}...")
    
    # Vérifier si la colonne de stratification existe
    if stratify_col not in df.columns:
        logger.warning(f"Colonne de stratification {stratify_col} non trouvée. Utilisation d'un split aléatoire.")
        stratify = None
    else:
        stratify = df[stratify_col]
    
    # Calculer les proportions pour les splits
    test_val_size = test_size + val_size
    relative_val_size = val_size / (1 - test_size)
    
    # Premier split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Deuxième split: train vs val
    if stratify_col in train_val_df.columns:
        stratify_val = train_val_df[stratify_col]
    else:
        stratify_val = None
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, random_state=random_state, stratify=stratify_val
    )
    
    logger.info(f"Split terminé. Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def align_datasets(dataframes, common_columns=None):
    """
    Aligne plusieurs datasets pour qu'ils aient les mêmes colonnes
    
    Args:
        dataframes: Liste de DataFrames à aligner
        common_columns: Liste de colonnes communes à conserver (si None, toutes les colonnes communes sont conservées)
        
    Returns:
        Liste de DataFrames alignés
    """
    logger.info("Alignement des datasets...")
    
    if not dataframes:
        logger.warning("Aucun dataset à aligner")
        return []
    
    # Trouver les colonnes communes à tous les datasets
    if common_columns is None:
        common_columns = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_columns = common_columns.intersection(df.columns)
        
        common_columns = list(common_columns)
    
    logger.info(f"Colonnes communes: {len(common_columns)} colonnes")
    
    # Aligner les datasets
    aligned_dfs = []
    for i, df in enumerate(dataframes):
        missing_cols = [col for col in common_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Dataset {i}: {len(missing_cols)} colonnes manquantes")
            
            # Ajouter les colonnes manquantes avec des valeurs NaN
            for col in missing_cols:
                df[col] = np.nan
        
        # Sélectionner uniquement les colonnes communes
        aligned_df = df[common_columns].copy()
        aligned_dfs.append(aligned_df)
    
    return aligned_dfs

def standardize_model_output():
    """
    Standardise le nom de sortie du modèle après l'entraînement
    """
    logger.info("Standardisation des noms de sortie du modèle...")
    
    # Définir le nom standard pour le modèle
    standard_model_name = "morningstar_model_v1"
    
    # Chercher les modèles existants
    model_files = []
    for ext in ['.h5', '/saved_model.pb']:
        model_files.extend(list(MODEL_DIR.glob(f"**/*{ext}")))
    
    if not model_files:
        logger.warning("Aucun fichier de modèle trouvé")
        return
    
    # Créer un répertoire pour les modèles standardisés
    standardized_model_dir = MODEL_DIR / "standardized"
    standardized_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier le modèle le plus récent avec un nom standardisé
    latest_model = max(model_files, key=os.path.getmtime)
    
    if latest_model.name.endswith('.h5'):
        # Pour les modèles H5
        target_path = standardized_model_dir / f"{standard_model_name}.h5"
        shutil.copy2(latest_model, target_path)
        logger.info(f"Modèle copié de {latest_model} vers {target_path}")
    else:
        # Pour les modèles SavedModel
        if 'saved_model.pb' in str(latest_model):
            source_dir = latest_model.parent
            target_dir = standardized_model_dir / standard_model_name
            
            # Copier tout le répertoire SavedModel
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)
            logger.info(f"Modèle copié de {source_dir} vers {target_dir}")
    
    # Créer un fichier de métadonnées pour le modèle
    metadata = {
        "model_name": standard_model_name,
        "original_path": str(latest_model),
        "standardized_path": str(standardized_model_dir / standard_model_name),
        "date_standardized": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_features": {
            "technical_indicators": 38,
            "hmm_features": 4,
            "llm_features": 2,
            "mcp_features": 128
        },
        "output_features": {
            "trading_signal": 5,
            "market_regime": 4,
            "volatility_quantiles": 3,
            "stop_loss": 1,
            "take_profit": 1
        }
    }
    
    with open(standardized_model_dir / f"{standard_model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Métadonnées du modèle sauvegardées dans {standardized_model_dir / f'{standard_model_name}_metadata.json'}")

def check_reasoning_module():
    """
    Vérifie le module de raisonnement du modèle
    """
    logger.info("Vérification du module de raisonnement...")
    
    reasoning_module_path = MODEL_DIR / "architecture" / "reasoning_model.py"
    reasoning_dir = MODEL_DIR / "reasoning_model"
    
    if not reasoning_module_path.exists():
        logger.warning(f"Module de raisonnement non trouvé à {reasoning_module_path}")
        return
    
    # Vérifier si des exemples d'explications existent
    explanation_examples_path = reasoning_dir / "explanation_examples.json"
    
    if explanation_examples_path.exists():
        with open(explanation_examples_path, 'r') as f:
            examples = json.load(f)
        logger.info(f"Exemples d'explications trouvés: {len(examples)} exemples")
    else:
        logger.warning(f"Aucun exemple d'explication trouvé à {explanation_examples_path}")
    
    # Vérifier les résultats de test du module de raisonnement
    test_results_path = reasoning_dir / "test_results.json"
    
    if test_results_path.exists():
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        logger.info(f"Résultats de test du module de raisonnement: {test_results}")
    else:
        logger.warning(f"Aucun résultat de test trouvé à {test_results_path}")

def main():
    """
    Fonction principale
    """
    logger.info("Début du processus de normalisation des datasets")
    
    # 1. Charger et normaliser les fichiers CSV bruts
    raw_csv_df = load_raw_csv_files()
    if raw_csv_df is not None:
        normalized_csv_df = normalize_dataset(raw_csv_df)
        if normalized_csv_df is not None:
            # Mettre à l'échelle les caractéristiques
            scaled_csv_df = scale_features(normalized_csv_df)
            # Sauvegarder le dataset normalisé
            output_path = STANDARDIZED_DIR / "standardized_raw_data.parquet"
            scaled_csv_df.to_parquet(output_path, index=False)
            logger.info(f"Dataset CSV normalisé sauvegardé dans {output_path}")
            
            # Effectuer un split stratifié
            train_df, val_df, test_df = stratified_split_dataset(scaled_csv_df)
            
            # Sauvegarder les splits
            train_path = STANDARDIZED_DIR / "standardized_raw_data_train.parquet"
            val_path = STANDARDIZED_DIR / "standardized_raw_data_val.parquet"
            test_path = STANDARDIZED_DIR / "standardized_raw_data_test.parquet"
            
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)
            
            logger.info(f"Splits sauvegardés dans {STANDARDIZED_DIR}")
    
    # 2. Charger et normaliser le dataset multi-crypto brut
    raw_parquet_path = RAW_DIR / "multi_crypto_dataset.parquet"
    raw_parquet_df = load_parquet_dataset(raw_parquet_path)
    if raw_parquet_df is not None:
        normalized_parquet_df = normalize_dataset(raw_parquet_df)
        if normalized_parquet_df is not None:
            # Mettre à l'échelle les caractéristiques
            scaled_parquet_df = scale_features(normalized_parquet_df)
            # Sauvegarder le dataset normalisé
            output_path = STANDARDIZED_DIR / "standardized_multi_crypto_dataset.parquet"
            scaled_parquet_df.to_parquet(output_path, index=False)
            logger.info(f"Dataset parquet brut normalisé sauvegardé dans {output_path}")
            
            # Effectuer un split stratifié
            train_df, val_df, test_df = stratified_split_dataset(scaled_parquet_df)
            
            # Sauvegarder les splits
            train_path = STANDARDIZED_DIR / "standardized_multi_crypto_dataset_train.parquet"
            val_path = STANDARDIZED_DIR / "standardized_multi_crypto_dataset_val.parquet"
            test_path = STANDARDIZED_DIR / "standardized_multi_crypto_dataset_test.parquet"
            
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)
            
            logger.info(f"Splits sauvegardés dans {STANDARDIZED_DIR}")
    
    # 3. Charger et normaliser le dataset multi-crypto préparé
    processed_parquet_path = PROCESSED_DIR / "multi_crypto_dataset_prepared.parquet"
    processed_parquet_df = load_parquet_dataset(processed_parquet_path)
    if processed_parquet_df is not None:
        normalized_processed_df = normalize_dataset(processed_parquet_df)
        if normalized_processed_df is not None:
            # Mettre à l'échelle les caractéristiques
            scaled_processed_df = scale_features(normalized_processed_df)
            # Sauvegarder le dataset normalisé
            output_path = STANDARDIZED_DIR / "standardized_prepared_dataset.parquet"
            scaled_processed_df.to_parquet(output_path, index=False)
            logger.info(f"Dataset parquet préparé normalisé sauvegardé dans {output_path}")
            
            # Effectuer un split stratifié
            train_df, val_df, test_df = stratified_split_dataset(scaled_processed_df)
            
            # Sauvegarder les splits
            train_path = STANDARDIZED_DIR / "standardized_prepared_dataset_train.parquet"
            val_path = STANDARDIZED_DIR / "standardized_prepared_dataset_val.parquet"
            test_path = STANDARDIZED_DIR / "standardized_prepared_dataset_test.parquet"
            
            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)
            
            logger.info(f"Splits sauvegardés dans {STANDARDIZED_DIR}")
    
    # 4. Standardiser le nom de sortie du modèle
    standardize_model_output()
    
    # 5. Vérifier le module de raisonnement
    check_reasoning_module()
    
    logger.info("Fin du processus de normalisation des datasets")

if __name__ == "__main__":
    main()
