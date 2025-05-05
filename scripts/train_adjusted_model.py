#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour entrau00eener le modu00e8le hybride ajustu00e9 aux dimensions du dataset enrichi.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

# Importer nos modules personnalisu00e9s
from model.architecture.enhanced_hybrid_model import build_enhanced_hybrid_model

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        Arguments parsu00e9s
    """
    parser = argparse.ArgumentParser(description="Entrau00eenement du modu00e8le hybride ajustu00e9")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/enriched/enriched_dataset.parquet",
        help="Chemin vers le dataset enrichi"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model/trained",
        help="Ru00e9pertoire de sortie pour le modu00e8le entrau00eenu00e9"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'u00e9poques d'entrau00eenement"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille du batch d'entrau00eenement"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Proportion de donnu00e9es utilisu00e9es pour la validation"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Utiliser l'early stopping"
    )
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    Charge le dataset enrichi.
    
    Args:
        dataset_path: Chemin vers le dataset enrichi
    
    Returns:
        DataFrame avec le dataset enrichi
    """
    logger.info(f"Chargement du dataset depuis {dataset_path}")
    try:
        df = pd.read_parquet(dataset_path)
        logger.info(f"Dataset chargu00e9 avec succu00e8s: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        sys.exit(1)

def preprocess_data(df):
    """
    Pru00e9traite les donnu00e9es pour l'entrau00eenement.
    
    Args:
        df: DataFrame avec le dataset enrichi
    
    Returns:
        Donnu00e9es pru00e9traitu00e9es pour l'entrau00eenement
    """
    logger.info("Pru00e9traitement des donnu00e9es pour l'entrau00eenement")
    
    # Identifier les colonnes techniques (indicateurs techniques)
    technical_columns = [
        col for col in df.columns if col.startswith(('rsi_', 'macd_', 'bbands_', 'ema_', 'sma_', 'atr_', 'adx_')) or 
        col in ['open', 'high', 'low', 'close', 'volume', 'returns']
    ]
    
    # Identifier les colonnes CryptoBERT
    cryptobert_columns = [col for col in df.columns if col.startswith('cryptobert_dim_')]
    
    # Identifier les colonnes de sentiment
    sentiment_columns = [col for col in df.columns if col == 'sentiment_score']
    
    # Identifier les colonnes HMM
    hmm_columns = [col for col in df.columns if col == 'hmm_regime']
    
    # Identifier les colonnes de mu00e9triques de marchu00e9
    market_columns = [col for col in df.columns if col.startswith('market_')]
    
    # Identifier les colonnes globales
    global_columns = [col for col in df.columns if col.startswith('global_')]
    
    # Combiner les colonnes de mu00e9triques de marchu00e9 et globales pour former MCP
    mcp_columns = market_columns + global_columns + sentiment_columns  # Inclure le sentiment dans MCP
    
    # Cru00e9er des donnu00e9es synthu00e9tiques pour les sorties (pour l'exemple)
    # Dans un cas ru00e9el, ces donnu00e9es seraient du00e9ju00e0 pru00e9sentes dans le dataset
    df['signal_class'] = np.random.randint(0, 5, size=len(df))  # 5 classes pour le signal de trading
    df['market_regime_class'] = np.random.randint(0, 4, size=len(df))  # 4 classes pour le ru00e9gime de marchu00e9
    df['volatility_quantile'] = np.random.random(size=len(df)) * 3  # 3 quantiles de volatilitu00e9
    df['stop_loss'] = np.random.random(size=len(df)) * 0.1  # Stop loss entre 0 et 10%
    df['take_profit'] = np.random.random(size=len(df)) * 0.2  # Take profit entre 0 et 20%
    
    # Normaliser les donnu00e9es numu00e9riques
    for col in technical_columns + mcp_columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0  # Si l'u00e9cart-type est 0, mettre u00e0 0
    
    # Convertir les donnu00e9es catu00e9gorielles en one-hot encoding
    df = pd.get_dummies(df, columns=['symbol'], prefix='symbol')
    
    # Ajouter une colonne instrument_type (pour l'exemple, tous sont de type 0 = spot)
    df['instrument_type'] = 0
    
    # Pru00e9parer les entru00e9es pour le modu00e8le
    X = {
        'technical_input': df[technical_columns].values,
        'llm_input': df[cryptobert_columns].values,
        'mcp_input': df[mcp_columns].values,
        'hmm_input': df[hmm_columns].values if hmm_columns else np.zeros((len(df), 1)),
        'instrument_input': df['instrument_type'].values.reshape(-1, 1)
    }
    
    # Pru00e9parer les sorties pour le modu00e8le
    y = {
        'signal': tf.keras.utils.to_categorical(df['signal_class'].values, num_classes=5),
        'market_regime': tf.keras.utils.to_categorical(df['market_regime_class'].values, num_classes=4),
        'volatility_quantiles': df[['volatility_quantile']].values,
        'sl_tp': df[['stop_loss', 'take_profit']].values
    }
    
    # Dimensions des entru00e9es
    input_dims = {
        'tech_dim': len(technical_columns),
        'llm_dim': len(cryptobert_columns),
        'mcp_dim': len(mcp_columns),
        'hmm_dim': len(hmm_columns) if hmm_columns else 1
    }
    
    logger.info(f"Donnu00e9es pru00e9traitu00e9es avec succu00e8s. Dimensions des entru00e9es: {input_dims}")
    
    return X, y, input_dims

def build_adjusted_model(input_dims):
    """
    Construit le modu00e8le ajustu00e9 aux dimensions du dataset.
    
    Args:
        input_dims: Dictionnaire avec les dimensions des entru00e9es
    
    Returns:
        Modu00e8le ajustu00e9
    """
    logger.info("Construction du modu00e8le ajustu00e9")
    
    # Construire le modu00e8le avec les dimensions ajustu00e9es
    model = build_enhanced_hybrid_model(
        tech_input_shape=(input_dims['tech_dim'],),
        llm_embedding_dim=input_dims['llm_dim'],
        mcp_input_dim=input_dims['mcp_dim'],
        hmm_input_dim=input_dims['hmm_dim'],
        instrument_vocab_size=10,
        instrument_embedding_dim=8,
        num_trading_classes=5,
        num_market_regime_classes=4,
        num_volatility_quantiles=3,
        num_sl_tp_outputs=2
    )
    
    # Compiler le modu00e8le
    model.compile(
        optimizer='adam',
        loss={
            'signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy',
            'volatility_quantiles': 'mse',
            'sl_tp': 'mse'
        },
        metrics={
            'signal': ['accuracy'],
            'market_regime': ['accuracy'],
            'volatility_quantiles': ['mae'],
            'sl_tp': ['mae']
        }
    )
    
    logger.info("Modu00e8le ajustu00e9 construit avec succu00e8s")
    return model

def train_model(model, X, y, args):
    """
    Entrau00eene le modu00e8le sur les donnu00e9es pru00e9traitu00e9es.
    
    Args:
        model: Modu00e8le u00e0 entrau00eener
        X: Donnu00e9es d'entru00e9e
        y: Donnu00e9es de sortie
        args: Arguments de la ligne de commande
    
    Returns:
        Historique d'entrau00eenement
    """
    logger.info(f"Entrau00eenement du modu00e8le sur {len(X['technical_input'])} u00e9chantillons")
    
    # Callbacks
    callbacks = []
    
    # Early stopping si demandu00e9
    if args.early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # TensorBoard callback
    log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard_callback)
    
    # Entrau00eener le modu00e8le
    history = model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Entrau00eenement terminu00e9 avec succu00e8s")
    return history

def save_model(model, output_dir, input_dims):
    """
    Sauvegarde le modu00e8le entrau00eenu00e9.
    
    Args:
        model: Modu00e8le entrau00eenu00e9
        output_dir: Ru00e9pertoire de sortie
        input_dims: Dimensions des entru00e9es
    """
    logger.info(f"Sauvegarde du modu00e8le dans {output_dir}")
    
    # Cru00e9er le ru00e9pertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modu00e8le
    model_path = output_path / "adjusted_model"
    model.save(model_path)
    
    # Sauvegarder les mu00e9tadonnu00e9es du modu00e8le
    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dimensions": input_dims,
        "model_summary": str(model.summary())
    }
    
    metadata_file = output_path / "model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Modu00e8le sauvegardu00e9 avec succu00e8s dans {model_path}")
    logger.info(f"Mu00e9tadonnu00e9es du modu00e8le sauvegardu00e9es dans {metadata_file}")

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Charger le dataset
    df = load_dataset(args.dataset_path)
    
    # Pru00e9traiter les donnu00e9es
    X, y, input_dims = preprocess_data(df)
    
    # Construire le modu00e8le ajustu00e9
    model = build_adjusted_model(input_dims)
    
    # Afficher le ru00e9sumu00e9 du modu00e8le
    model.summary()
    
    # Entrau00eener le modu00e8le
    history = train_model(model, X, y, args)
    
    # Sauvegarder le modu00e8le
    save_model(model, args.output_dir, input_dims)
    
    logger.info("Entrau00eenement et sauvegarde du modu00e8le terminu00e9s avec succu00e8s!")

if __name__ == "__main__":
    main()
