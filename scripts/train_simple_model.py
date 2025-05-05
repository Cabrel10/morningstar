#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour entraîner un modèle simplifié sur le dataset normalisé.
Ce script se concentre sur les caractéristiques essentielles et évite les problèmes de conversion de types.
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
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(description="Entraîne un modèle simplifié sur un dataset normalisé.")
    
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
        default=30,
        help="Nombre d'époques d'entraînement"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
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

def prepare_features(df):
    """
    Prépare les caractéristiques pour l'entraînement.
    Sélectionne uniquement les caractéristiques numériques essentielles.
    """
    logger.info("Préparation des caractéristiques pour l'entraînement")
    
    # Sélectionner les caractéristiques numériques essentielles
    essential_features = ['open', 'high', 'low', 'close', 'volume']
    
    # Ajouter les indicateurs techniques s'ils existent
    tech_indicators = ['RSI', 'MACD', 'BBU', 'BBL', 'ATR']
    for indicator in tech_indicators:
        if indicator in df.columns:
            essential_features.append(indicator)
    
    # Ajouter les caractéristiques HMM si elles existent
    hmm_features = ['hmm_regime']
    for feature in hmm_features:
        if feature in df.columns:
            essential_features.append(feature)
    
    # Filtrer pour ne garder que les caractéristiques qui existent dans le DataFrame
    features = [col for col in essential_features if col in df.columns]
    
    if not features:
        logger.error("Aucune caractéristique essentielle trouvée dans le dataset")
        sys.exit(1)
    
    logger.info(f"Caractéristiques sélectionnées: {features}")
    
    # Extraire les caractéristiques
    X = df[features].copy()
    
    # Remplacer les valeurs NaN par 0
    X = X.fillna(0)
    
    # Normaliser les caractéristiques
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Caractéristiques préparées: {X_scaled.shape}")
    return X_scaled, features

def create_targets(df, n_samples):
    """
    Crée des cibles pour l'entraînement.
    Utilise des cibles simulées si elles n'existent pas dans le dataset.
    """
    logger.info("Création des cibles pour l'entraînement")
    
    # Vérifier si des cibles existent dans le dataset
    if 'target_signal' in df.columns:
        logger.info("Utilisation des cibles existantes dans le dataset")
        y_signal = df['target_signal'].values
        y_regime = df['target_regime'].values if 'target_regime' in df.columns else np.random.randint(0, 4, size=n_samples)
    else:
        logger.info("Génération de cibles simulées")
        # Générer des cibles simulées
        y_signal = np.random.randint(0, 3, size=n_samples)  # 0: Vendre, 1: Conserver, 2: Acheter
        y_regime = np.random.randint(0, 4, size=n_samples)  # 4 régimes de marché
    
    # Convertir en one-hot encoding
    y_signal_onehot = tf.keras.utils.to_categorical(y_signal, num_classes=3)
    y_regime_onehot = tf.keras.utils.to_categorical(y_regime, num_classes=4)
    
    logger.info(f"Cibles créées: signal={y_signal_onehot.shape}, regime={y_regime_onehot.shape}")
    return y_signal_onehot, y_regime_onehot

def build_simple_model(input_dim):
    """
    Construit un modèle simplifié pour la prédiction de signaux de trading et de régimes de marché.
    """
    logger.info(f"Construction du modèle simplifié avec dimension d'entrée {input_dim}")
    
    # Définir l'entrée
    inputs = tf.keras.layers.Input(shape=(input_dim,), name='input')
    
    # Couches cachées
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Couche partagée finale
    shared = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Têtes de sortie
    signal_output = tf.keras.layers.Dense(3, activation='softmax', name='signal')(shared)  # 3 classes: Vendre, Conserver, Acheter
    regime_output = tf.keras.layers.Dense(4, activation='softmax', name='regime')(shared)  # 4 régimes de marché
    
    # Créer le modèle
    model = tf.keras.Model(inputs=inputs, outputs=[signal_output, regime_output], name='morningstar_simple_model')
    
    # Compiler le modèle
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'signal': 'categorical_crossentropy',
            'regime': 'categorical_crossentropy'
        },
        metrics={
            'signal': ['accuracy'],
            'regime': ['accuracy']
        }
    )
    
    logger.info("Modèle simplifié construit avec succès")
    return model

def train_model(model, X, y_signal, y_regime, args):
    """
    Entraîne le modèle sur les données préparées.
    """
    logger.info(f"Entraînement du modèle sur {len(X)} échantillons")
    
    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_signal_train, y_signal_val, y_regime_train, y_regime_val = train_test_split(
        X, y_signal, y_regime, test_size=args.validation_split, random_state=42
    )
    
    # Définir les callbacks
    callbacks = []
    
    # Ajouter TensorBoard pour la visualisation
    log_dir = Path(args.output_dir) / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
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
        {'signal': y_signal_train, 'regime': y_regime_train},
        validation_data=(X_val, {'signal': y_signal_val, 'regime': y_regime_val}),
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

def save_model(model, output_dir, features):
    """
    Sauvegarde le modèle entraîné et ses métadonnées.
    """
    logger.info(f"Sauvegarde du modèle dans {output_dir}")
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le modèle au format SavedModel
    model_path = output_path / "simple_model"
    model.save(model_path)
    logger.info(f"Modèle sauvegardé dans {model_path}")
    
    # Sauvegarder également au format H5 pour compatibilité
    h5_path = output_path / "simple_model.h5"
    model.save(h5_path)
    logger.info(f"Modèle sauvegardé au format H5 dans {h5_path}")
    
    # Sauvegarder les métadonnées du modèle
    metadata = {
        "model_name": "morningstar_simple_model",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_features": features,
        "output_classes": {
            "signal": 3,  # Vendre, Conserver, Acheter
            "regime": 4   # 4 régimes de marché
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
    
    # Préparer les caractéristiques
    X, features = prepare_features(df)
    
    # Créer les cibles
    y_signal, y_regime = create_targets(df, len(X))
    
    # Construire le modèle
    model = build_simple_model(X.shape[1])
    model.summary()
    
    # Entraîner le modèle
    history = train_model(model, X, y_signal, y_regime, args)
    
    # Sauvegarder le modèle
    save_model(model, args.output_dir, features)
    
    logger.info("Entraînement terminé avec succès")

if __name__ == "__main__":
    main()
