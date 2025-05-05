#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour entraîner un modèle hybride flexible qui s'adapte aux dimensions du dataset.
Ce script détecte automatiquement les dimensions des features et ajuste le modèle en conséquence.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins des répertoires
BASE_DIR = Path('/home/morningstar/Desktop/crypto_robot/Morningstar')
MODEL_DIR = BASE_DIR / 'model'

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Entraîne un modèle hybride flexible sur un dataset normalisé.")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Chemin vers le dataset normalisé (format parquet)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODEL_DIR / "trained"),
        help="Répertoire de sortie pour le modèle entraîné"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'époques d'entraînement"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille du batch pour l'entraînement"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Proportion des données à utiliser pour la validation"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Utiliser l'early stopping pour éviter le surapprentissage"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage pour l'optimiseur"
    )
    
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    Charge le dataset depuis un fichier parquet.
    """
    logger.info(f"Chargement du dataset depuis {dataset_path}")
    
    try:
        df = pd.read_parquet(dataset_path)
        logger.info(f"Dataset chargé avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        sys.exit(1)

def detect_feature_dimensions(df):
    """
    Détecte automatiquement les dimensions des différentes catégories de features.
    """
    logger.info("Détection des dimensions des features")
    
    # Identifier les colonnes par catégorie
    tech_columns = [
        'open', 'high', 'low', 'close', 'volume', 'SMA_short', 'SMA_long', 
        'EMA_short', 'EMA_long', 'RSI', 'MACD', 'MACDs', 'MACDh', 'BBU', 'BBM', 
        'BBL', 'ATR', 'STOCHk', 'STOCHd', 'ADX', 'CCI', 'Momentum', 'ROC',
        'Williams_%R', 'TRIX', 'Ultimate_Osc', 'DPO', 'OBV', 'VWMA',
        'CMF', 'MFI', 'Parabolic_SAR', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
        'Ichimoku_SenkouA', 'Ichimoku_SenkouB', 'Ichimoku_Chikou',
        'KAMA', 'VWAP', 'STOCHRSIk', 'CMO', 'PPO', 'FISHERt'
    ]
    
    hmm_columns = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Colonnes LLM et MCP
    llm_columns = [col for col in df.columns if 'llm_' in col]
    mcp_columns = [col for col in df.columns if 'mcp_feature_' in col]
    
    # Filtrer pour ne garder que les colonnes qui existent réellement dans le DataFrame
    tech_columns = [col for col in tech_columns if col in df.columns]
    hmm_columns = [col for col in hmm_columns if col in df.columns]
    
    # Obtenir les dimensions
    tech_dim = len(tech_columns)
    llm_dim = len(llm_columns)
    mcp_dim = len(mcp_columns)
    hmm_dim = len(hmm_columns)
    
    dimensions = {
        'tech_dim': tech_dim,
        'llm_dim': llm_dim,
        'mcp_dim': mcp_dim,
        'hmm_dim': hmm_dim,
        'tech_columns': tech_columns,
        'llm_columns': llm_columns,
        'mcp_columns': mcp_columns,
        'hmm_columns': hmm_columns
    }
    
    logger.info(f"Dimensions détectées: tech={tech_dim}, llm={llm_dim}, mcp={mcp_dim}, hmm={hmm_dim}")
    return dimensions

def preprocess_data(df, dimensions):
    """
    Prétraite les données pour l'entraînement.
    """
    logger.info("Prétraitement des données pour l'entraînement")
    
    # Extraire les features par catégorie
    X_tech = df[dimensions['tech_columns']].values if dimensions['tech_columns'] else np.zeros((len(df), 0))
    X_llm = df[dimensions['llm_columns']].values if dimensions['llm_columns'] else np.zeros((len(df), 0))
    X_mcp = df[dimensions['mcp_columns']].values if dimensions['mcp_columns'] else np.zeros((len(df), 0))
    X_hmm = df[dimensions['hmm_columns']].values if dimensions['hmm_columns'] else np.zeros((len(df), 0))
    
    # Générer des cibles simulées si elles n'existent pas dans le dataset
    if 'target_signal' not in df.columns:
        logger.info("Génération de cibles simulées pour l'entraînement")
        y_signal = np.random.randint(0, 5, size=(len(df),))  # 5 classes de signal
        y_regime = np.random.randint(0, 4, size=(len(df),))  # 4 classes de régime
        y_sl_tp = np.random.normal(0, 1, size=(len(df), 2))  # Stop Loss et Take Profit
        y_vol = np.random.normal(0, 1, size=(len(df), 3))    # Quantiles de volatilité
    else:
        # Utiliser les cibles du dataset si elles existent
        y_signal = df['target_signal'].values
        y_regime = df['target_regime'].values
        y_sl_tp = df[['target_sl', 'target_tp']].values
        y_vol = df[['target_vol_low', 'target_vol_med', 'target_vol_high']].values
    
    # Convertir les cibles catégorielles en one-hot
    y_signal_onehot = tf.keras.utils.to_categorical(y_signal, num_classes=5)
    y_regime_onehot = tf.keras.utils.to_categorical(y_regime, num_classes=4)
    
    # Préparer les entrées et sorties
    X = {
        'technical_input': X_tech,
        'llm_input': X_llm,
        'mcp_input': X_mcp,
        'hmm_input': X_hmm
    }
    
    y = {
        'signal': y_signal_onehot,
        'market_regime': y_regime_onehot,
        'sl_tp': y_sl_tp,
        'volatility_quantiles': y_vol
    }
    
    logger.info("Données prétraitées avec succès")
    return X, y

def build_flexible_model(dimensions):
    """
    Construit un modèle hybride flexible qui s'adapte aux dimensions des features.
    """
    logger.info("Construction du modèle flexible")
    
    # Définir les entrées
    technical_input = tf.keras.layers.Input(shape=(dimensions['tech_dim'],), name='technical_input')
    llm_input = tf.keras.layers.Input(shape=(dimensions['llm_dim'],), name='llm_input')
    mcp_input = tf.keras.layers.Input(shape=(dimensions['mcp_dim'],), name='mcp_input')
    hmm_input = tf.keras.layers.Input(shape=(dimensions['hmm_dim'],), name='hmm_input')
    
    # Encodeurs spécifiques à chaque type d'entrée
    # Encodeur technique
    if dimensions['tech_dim'] > 0:
        tech_dense1 = tf.keras.layers.Dense(64, activation='relu', name='tech_dense1')(technical_input)
        tech_bn1 = tf.keras.layers.BatchNormalization(name='tech_bn1')(tech_dense1)
        tech_dropout1 = tf.keras.layers.Dropout(0.3, name='tech_dropout1')(tech_bn1)
        tech_dense2 = tf.keras.layers.Dense(32, activation='relu', name='tech_dense2')(tech_dropout1)
        tech_bn2 = tf.keras.layers.BatchNormalization(name='tech_bn2')(tech_dense2)
        tech_output = tf.keras.layers.Dropout(0.3, name='tech_dropout2')(tech_bn2)
    else:
        tech_output = tf.keras.layers.Lambda(lambda x: x, name='tech_dummy')(technical_input)
    
    # Encodeur LLM
    if dimensions['llm_dim'] > 0:
        llm_dense1 = tf.keras.layers.Dense(128, activation='relu', name='llm_dense1')(llm_input)
        llm_bn1 = tf.keras.layers.BatchNormalization(name='llm_bn1')(llm_dense1)
        llm_dropout1 = tf.keras.layers.Dropout(0.3, name='llm_dropout1')(llm_bn1)
        llm_dense2 = tf.keras.layers.Dense(64, activation='relu', name='llm_dense2')(llm_dropout1)
        llm_bn2 = tf.keras.layers.BatchNormalization(name='llm_bn2')(llm_dense2)
        llm_output = tf.keras.layers.Dropout(0.3, name='llm_dropout2')(llm_bn2)
    else:
        llm_output = tf.keras.layers.Lambda(lambda x: x, name='llm_dummy')(llm_input)
    
    # Encodeur MCP
    if dimensions['mcp_dim'] > 0:
        mcp_dense1 = tf.keras.layers.Dense(64, activation='relu', name='mcp_dense1')(mcp_input)
        mcp_bn1 = tf.keras.layers.BatchNormalization(name='mcp_bn1')(mcp_dense1)
        mcp_dropout1 = tf.keras.layers.Dropout(0.3, name='mcp_dropout1')(mcp_bn1)
        mcp_dense2 = tf.keras.layers.Dense(32, activation='relu', name='mcp_dense2')(mcp_dropout1)
        mcp_bn2 = tf.keras.layers.BatchNormalization(name='mcp_bn2')(mcp_dense2)
        mcp_output = tf.keras.layers.Dropout(0.3, name='mcp_dropout2')(mcp_bn2)
    else:
        mcp_output = tf.keras.layers.Lambda(lambda x: x, name='mcp_dummy')(mcp_input)
    
    # Encodeur HMM
    if dimensions['hmm_dim'] > 0:
        hmm_dense1 = tf.keras.layers.Dense(32, activation='relu', name='hmm_dense1')(hmm_input)
        hmm_bn1 = tf.keras.layers.BatchNormalization(name='hmm_bn1')(hmm_dense1)
        hmm_dropout1 = tf.keras.layers.Dropout(0.3, name='hmm_dropout1')(hmm_bn1)
        hmm_dense2 = tf.keras.layers.Dense(16, activation='relu', name='hmm_dense2')(hmm_dropout1)
        hmm_bn2 = tf.keras.layers.BatchNormalization(name='hmm_bn2')(hmm_dense2)
        hmm_output = tf.keras.layers.Dropout(0.3, name='hmm_dropout2')(hmm_bn2)
    else:
        hmm_output = tf.keras.layers.Lambda(lambda x: x, name='hmm_dummy')(hmm_input)
    
    # Fusion des représentations
    fusion_inputs = []
    if dimensions['tech_dim'] > 0:
        fusion_inputs.append(tech_output)
    if dimensions['llm_dim'] > 0:
        fusion_inputs.append(llm_output)
    if dimensions['mcp_dim'] > 0:
        fusion_inputs.append(mcp_output)
    if dimensions['hmm_dim'] > 0:
        fusion_inputs.append(hmm_output)
    
    # Ajouter un flatten pour gérer le cas où il n'y a qu'une seule entrée
    flattened_inputs = [tf.keras.layers.Flatten(name=f'flatten_{i}')(inp) for i, inp in enumerate(fusion_inputs)]
    
    if len(flattened_inputs) > 1:
        fusion = tf.keras.layers.Concatenate(name='fusion_concat')(flattened_inputs)
    else:
        fusion = flattened_inputs[0]
    
    # Couches partagées
    fusion_dense1 = tf.keras.layers.Dense(256, activation='relu', name='fusion_dense1')(fusion)
    fusion_bn1 = tf.keras.layers.BatchNormalization(name='fusion_bn1')(fusion_dense1)
    fusion_dropout1 = tf.keras.layers.Dropout(0.3, name='fusion_dropout1')(fusion_bn1)
    
    fusion_dense2 = tf.keras.layers.Dense(128, activation='relu', name='fusion_dense2')(fusion_dropout1)
    fusion_bn2 = tf.keras.layers.BatchNormalization(name='fusion_bn2')(fusion_dense2)
    fusion_dropout2 = tf.keras.layers.Dropout(0.3, name='fusion_dropout2')(fusion_bn2)
    
    # Couche partagée finale
    shared_dense1 = tf.keras.layers.Dense(128, activation='relu', name='shared_dense1')(fusion_dropout2)
    shared_bn1 = tf.keras.layers.BatchNormalization(name='shared_bn1')(shared_dense1)
    shared_dropout1 = tf.keras.layers.Dropout(0.3, name='shared_dropout1')(shared_bn1)
    shared_output = tf.keras.layers.Dense(64, activation='relu', name='shared_dense_output')(shared_dropout1)
    
    # Têtes de sortie multi-tâches
    market_regime_output = tf.keras.layers.Dense(4, activation='softmax', name='market_regime')(shared_output)
    signal_output = tf.keras.layers.Dense(5, activation='softmax', name='signal')(shared_output)
    sl_tp_output = tf.keras.layers.Dense(2, activation='linear', name='sl_tp')(shared_output)
    volatility_output = tf.keras.layers.Dense(3, activation='linear', name='volatility_quantiles')(shared_output)
    
    # Créer le modèle
    model = tf.keras.Model(
        inputs=[technical_input, llm_input, mcp_input, hmm_input],
        outputs=[signal_output, market_regime_output, sl_tp_output, volatility_output],
        name='enhanced_hybrid_model_v2'
    )
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy',
            'sl_tp': 'mse',
            'volatility_quantiles': 'mse'
        },
        metrics={
            'signal': ['accuracy'],
            'market_regime': ['accuracy'],
            'sl_tp': ['mae'],
            'volatility_quantiles': ['mae']
        }
    )
    
    logger.info("Modèle flexible construit avec succès")
    return model

def train_model(model, X, y, args):
    """
    Entraîne le modèle sur les données prétraitées.
    """
    logger.info(f"Entraînement du modèle sur {len(X['technical_input'])} échantillons")
    
    # Diviser les données en ensembles d'entraînement et de validation
    X_train = {}
    X_val = {}
    y_train = {}
    y_val = {}
    
    # Créer un index pour la division
    indices = np.arange(len(X['technical_input']))
    train_indices, val_indices = train_test_split(indices, test_size=args.validation_split, random_state=42)
    
    # Diviser les entrées
    for key in X.keys():
        X_train[key] = X[key][train_indices]
        X_val[key] = X[key][val_indices]
    
    # Diviser les sorties
    for key in y.keys():
        y_train[key] = y[key][train_indices]
        y_val[key] = y[key][val_indices]
    
    # Définir les callbacks
    callbacks = []
    
    # Ajouter TensorBoard pour la visualisation
    log_dir = Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard_callback)
    
    # Ajouter ModelCheckpoint pour sauvegarder le meilleur modèle
    checkpoint_path = Path(args.output_dir) / "checkpoints" / "best_model.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Ajouter EarlyStopping si demandé
    if args.early_stopping:
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
    
    # Entraîner le modèle
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder l'historique d'entraînement
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({key: [float(val) for val in values] for key, values in history.history.items()}, f, indent=2)
    
    logger.info(f"Historique d'entraînement sauvegardé dans {history_path}")
    return history

def save_model(model, output_dir, dimensions):
    """
    Sauvegarde le modèle entraîné et ses métadonnées.
    """
    logger.info(f"Sauvegarde du modèle dans {output_dir}")
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle au format SavedModel
    model_path = output_path / "flexible_model"
    model.save(model_path)
    logger.info(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarder également au format H5 pour compatibilité
    h5_path = output_path / "flexible_model.h5"
    model.save(h5_path)
    logger.info(f"Modèle sauvegardé au format H5 dans {h5_path}")
    
    # Sauvegarder les métadonnées du modèle
    metadata = {
        "model_name": "morningstar_flexible_model",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dimensions": {
            "tech_dim": dimensions['tech_dim'],
            "llm_dim": dimensions['llm_dim'],
            "mcp_dim": dimensions['mcp_dim'],
            "hmm_dim": dimensions['hmm_dim']
        },
        "feature_columns": {
            "tech_columns": dimensions['tech_columns'],
            "llm_columns": dimensions['llm_columns'],
            "mcp_columns": dimensions['mcp_columns'],
            "hmm_columns": dimensions['hmm_columns']
        }
    }
    
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Métadonnées du modèle sauvegardées dans {metadata_path}")

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Charger le dataset
    df = load_dataset(args.dataset_path)
    
    # Détecter les dimensions des features
    dimensions = detect_feature_dimensions(df)
    
    # Prétraiter les données
    X, y = preprocess_data(df, dimensions)
    
    # Construire le modèle
    model = build_flexible_model(dimensions)
    model.summary()
    
    # Entraîner le modèle
    history = train_model(model, X, y, args)
    
    # Sauvegarder le modèle
    save_model(model, args.output_dir, dimensions)
    
    logger.info("Entraînement terminé avec succès")

if __name__ == "__main__":
    main()
