#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'évaluation amélioré pour les modèles Morningstar.
Compatible avec les fichiers parquet et les modèles simples.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Évalue un modèle Morningstar sur un dataset parquet.")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Chemin vers le modèle entraîné (.h5)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Chemin vers le dataset (format parquet)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Répertoire de sortie pour les résultats d'évaluation"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Effectuer une optimisation après l'évaluation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Taux d'apprentissage pour l'optimisation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d'époques pour l'optimisation"
    )
    
    return parser.parse_args()

def load_data(data_path):
    """
    Charge les données depuis un fichier parquet.
    """
    logger.info(f"Chargement des données depuis {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
    return df

def prepare_features(df):
    """
    Prépare les caractéristiques pour l'évaluation.
    """
    logger.info("Préparation des caractéristiques pour l'évaluation")
    
    # Utiliser exactement les mêmes 6 caractéristiques que celles utilisées lors de l'entraînement
    selected_cols = ['open', 'high', 'low', 'close', 'volume', 'hmm_regime']
    logger.info(f"Caractéristiques sélectionnées: {selected_cols}")
    
    # Vérifier si les colonnes existent dans le DataFrame
    valid_cols = [col for col in selected_cols if col in df.columns]
    
    # Si certaines colonnes sont manquantes, ajouter des colonnes de zéros
    missing_cols = [col for col in selected_cols if col not in valid_cols]
    if missing_cols:
        logger.warning(f"Colonnes manquantes: {missing_cols}. Ajout de colonnes de zéros.")
        for col in missing_cols:
            df[col] = 0.0
        valid_cols = selected_cols
    
    # Extraire les caractéristiques
    X = df[valid_cols].values
    logger.info(f"Caractéristiques préparées: {X.shape}")
    
    return X, valid_cols

def create_targets(df, n_samples):
    """
    Crée des cibles pour l'évaluation.
    """
    logger.info("Création des cibles pour l'évaluation")
    
    # Vérifier si les colonnes de cible existent dans le DataFrame
    has_signal = 'signal' in df.columns
    has_regime = 'market_regime' in df.columns
    
    if has_signal and has_regime:
        logger.info("Utilisation des cibles existantes dans le dataset")
        y_signal = df['signal'].values
        y_regime = df['market_regime'].values
    else:
        logger.info("Génération de cibles synthétiques")
        
        # Signal de trading (3 classes: achat, neutre, vente)
        signal = np.zeros((n_samples, 3))
        for i in range(n_samples):
            if i % 3 == 0:
                signal[i, 0] = 1  # Achat
            elif i % 3 == 1:
                signal[i, 1] = 1  # Neutre
            else:
                signal[i, 2] = 1  # Vente
        
        # Régime de marché (4 classes: baissier, neutre, haussier, volatil)
        regime = np.zeros((n_samples, 4))
        for i in range(n_samples):
            regime[i, i % 4] = 1
        
        y_signal = signal
        y_regime = regime
    
    logger.info(f"Cibles créées: signal={y_signal.shape}, regime={y_regime.shape}")
    return y_signal, y_regime

def evaluate_model(model, X, y_signal, y_regime):
    """
    Évalue le modèle sur les données.
    """
    logger.info("Évaluation du modèle")
    
    # Évaluer le modèle
    results = model.evaluate(X, [y_signal, y_regime], verbose=1)
    
    # Afficher les résultats
    logger.info("Résultats de l'évaluation:")
    logger.info(f"  Loss totale: {results[0]:.4f}")
    logger.info(f"  Loss signal: {results[1]:.4f}")
    logger.info(f"  Loss régime: {results[2]:.4f}")
    logger.info(f"  Précision signal: {results[3]:.4f}")
    logger.info(f"  Précision régime: {results[4]:.4f}")
    
    # Faire des prédictions
    predictions = model.predict(X)
    signal_pred = predictions[0]
    regime_pred = predictions[1]
    
    # Convertir les prédictions en classes
    signal_classes = np.argmax(signal_pred, axis=1)
    regime_classes = np.argmax(regime_pred, axis=1)
    true_signal_classes = np.argmax(y_signal, axis=1)
    true_regime_classes = np.argmax(y_regime, axis=1)
    
    # Rapports de classification
    signal_report = classification_report(
        true_signal_classes, 
        signal_classes,
        target_names=['Achat', 'Neutre', 'Vente'],
        output_dict=True
    )
    
    regime_report = classification_report(
        true_regime_classes, 
        regime_classes,
        target_names=['Baissier', 'Neutre', 'Haussier', 'Volatil'],
        output_dict=True
    )
    
    # Matrices de confusion
    signal_cm = confusion_matrix(true_signal_classes, signal_classes)
    regime_cm = confusion_matrix(true_regime_classes, regime_classes)
    
    # Afficher les rapports
    logger.info("\nRapport de classification pour le signal de trading:")
    logger.info(classification_report(
        true_signal_classes, 
        signal_classes,
        target_names=['Achat', 'Neutre', 'Vente']
    ))
    
    logger.info("\nRapport de classification pour le régime de marché:")
    logger.info(classification_report(
        true_regime_classes, 
        regime_classes,
        target_names=['Baissier', 'Neutre', 'Haussier', 'Volatil']
    ))
    
    return {
        'metrics': {
            'loss': results[0],
            'signal_loss': results[1],
            'regime_loss': results[2],
            'signal_accuracy': results[3],
            'regime_accuracy': results[4]
        },
        'signal_report': signal_report,
        'regime_report': regime_report,
        'signal_cm': signal_cm.tolist(),
        'regime_cm': regime_cm.tolist()
    }

def optimize_model(model, X, y_signal, y_regime, learning_rate=0.0001, epochs=5):
    """
    Optimise le modèle avec un taux d'apprentissage plus bas.
    """
    logger.info(f"Optimisation du modèle avec learning_rate={learning_rate}, epochs={epochs}")
    
    # Recompiler le modèle avec un taux d'apprentissage plus bas
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'signal': 'categorical_crossentropy',
            'regime': 'categorical_crossentropy'
        },
        metrics={
            'signal': ['accuracy'],
            'regime': ['accuracy']
        }
    )
    
    # Diviser les données pour l'entraînement et la validation
    X_train, X_val, y_signal_train, y_signal_val, y_regime_train, y_regime_val = train_test_split(
        X, y_signal, y_regime, test_size=0.2, random_state=42
    )
    
    # Entraînement supplémentaire
    history = model.fit(
        X_train, 
        [y_signal_train, y_regime_train],
        validation_data=(X_val, [y_signal_val, y_regime_val]),
        epochs=epochs,
        batch_size=16,
        verbose=1
    )
    
    # Évaluation finale
    logger.info("Évaluation finale du modèle optimisé")
    results_fine = model.evaluate(X, [y_signal, y_regime], verbose=1)
    
    logger.info("Résultats de l'évaluation finale:")
    logger.info(f"  Loss totale: {results_fine[0]:.4f}")
    logger.info(f"  Loss signal: {results_fine[1]:.4f}")
    logger.info(f"  Loss régime: {results_fine[2]:.4f}")
    logger.info(f"  Précision signal: {results_fine[3]:.4f}")
    logger.info(f"  Précision régime: {results_fine[4]:.4f}")
    
    return model, history.history

def save_results(results, output_dir):
    """
    Sauvegarde les résultats d'évaluation.
    """
    logger.info(f"Sauvegarde des résultats dans {output_dir}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les résultats au format JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Créer des visualisations
    # Matrice de confusion pour le signal
    plt.figure(figsize=(8, 6))
    plt.imshow(results['signal_cm'], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion - Signal de trading')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['Achat', 'Neutre', 'Vente'], rotation=45)
    plt.yticks(tick_marks, ['Achat', 'Neutre', 'Vente'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_confusion_matrix.png'))
    
    # Matrice de confusion pour le régime
    plt.figure(figsize=(8, 6))
    plt.imshow(results['regime_cm'], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de confusion - Régime de marché')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['Baissier', 'Neutre', 'Haussier', 'Volatil'], rotation=45)
    plt.yticks(tick_marks, ['Baissier', 'Neutre', 'Haussier', 'Volatil'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regime_confusion_matrix.png'))
    
    logger.info(f"Résultats sauvegardés dans {output_dir}")

def main():
    """
    Fonction principale.
    """
    args = parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configurer le logging vers un fichier
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Charger le modèle
    logger.info(f"Chargement du modèle depuis {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    model.summary(print_fn=logger.info)
    
    # Charger et préparer les données
    df = load_data(args.data_path)
    X, feature_cols = prepare_features(df)
    y_signal, y_regime = create_targets(df, len(df))
    
    # Évaluer le modèle
    results = evaluate_model(model, X, y_signal, y_regime)
    
    # Optimiser le modèle si demandé
    if args.optimize:
        optimized_model, history = optimize_model(
            model, X, y_signal, y_regime, 
            learning_rate=args.learning_rate,
            epochs=args.epochs
        )
        
        # Sauvegarder le modèle optimisé
        optimized_model_path = os.path.join(args.output_dir, 'optimized_model.h5')
        logger.info(f"Sauvegarde du modèle optimisé dans {optimized_model_path}")
        optimized_model.save(optimized_model_path)
        
        # Ajouter l'historique d'optimisation aux résultats
        results['optimization_history'] = history
    
    # Sauvegarder les résultats
    save_results(results, args.output_dir)
    
    logger.info("Évaluation terminée avec succès")

if __name__ == "__main__":
    main()
