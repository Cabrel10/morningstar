#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement pour le modèle Morningstar avec capacité de raisonnement.
Ce script est compatible avec Google Colab et utilise les techniques les plus récentes
pour l'entraînement de modèles de trading avec capacité d'explication.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.reasoning.reasoning_module import ExplanationDecoder

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
FEATURE_TYPES = {
    'technical': 'Features techniques (RSI, MACD, etc.)',
    'price': 'Données de prix (open, high, low, close, volume)',
    'llm': 'Embeddings de langage (sentiment, contexte)',
    'mcp': 'Features multi-crypto',
    'hmm': 'États cachés du modèle HMM'
}

def load_data(data_path, validation_split=0.1, random_state=42):
    """
    Charge les données normalisées et les divise en ensembles d'entraînement et de validation.
    
    Args:
        data_path: Chemin vers le dataset normalisé
        validation_split: Proportion des données d'entraînement à utiliser pour la validation
        random_state: Graine aléatoire pour la reproductibilité
    
    Returns:
        X_train, X_val, y_train, y_val, feature_names: Dictionnaires de features et labels, et noms des features
    """
    logger.info(f"Chargement des données depuis {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Séparer les ensembles d'entraînement et de test
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    test_df = df[df['split'] == 'test'].drop(columns=['split'])
    logger.info(f"Ensemble d'entraînement: {len(train_df)} lignes, Ensemble de test: {len(test_df)} lignes")
    
    # Diviser l'ensemble d'entraînement en entraînement et validation
    if validation_split > 0:
        train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=random_state)
        logger.info(f"Ensemble d'entraînement final: {len(train_df)} lignes, Ensemble de validation: {len(val_df)} lignes")
    else:
        val_df = test_df.copy()
        logger.info(f"Utilisation de l'ensemble de test comme validation: {len(val_df)} lignes")
    
    # Extraire les features et labels
    # Colonnes techniques
    technical_cols = [col for col in train_df.columns if col not in [
        'timestamp', 'symbol', 'market_regime', 'level_sl', 'level_tp',
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
    ] and not col.startswith('llm_') and not col.startswith('mcp_')]
    
    # Colonnes LLM (si présentes)
    llm_cols = [col for col in train_df.columns if col.startswith('llm_')]
    if not llm_cols:  # Si pas de colonnes LLM, créer un vecteur vide
        train_df['llm_dummy'] = 0.0
        val_df['llm_dummy'] = 0.0
        test_df['llm_dummy'] = 0.0
        llm_cols = ['llm_dummy']
    
    # Colonnes MCP (si présentes)
    mcp_cols = [col for col in train_df.columns if col.startswith('mcp_')]
    if not mcp_cols:  # Si pas de colonnes MCP, créer un vecteur vide
        train_df['mcp_dummy'] = 0.0
        val_df['mcp_dummy'] = 0.0
        test_df['mcp_dummy'] = 0.0
        mcp_cols = ['mcp_dummy']
    
    # Colonnes HMM
    hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Convertir les symboles en entiers pour l'embedding
    symbol_mapping = {symbol: i for i, symbol in enumerate(train_df['symbol'].unique())}
    train_df['symbol_id'] = train_df['symbol'].map(symbol_mapping)
    val_df['symbol_id'] = val_df['symbol'].map(symbol_mapping)
    test_df['symbol_id'] = test_df['symbol'].map(symbol_mapping)
    
    # Créer un dictionnaire de noms de features avec leurs types
    feature_names = []
    for col in technical_cols:
        feature_names.append(col)
    
    for col in llm_cols:
        feature_names.append(col)
    
    for col in mcp_cols:
        feature_names.append(col)
    
    for col in hmm_cols:
        feature_names.append(col)
    
    # Préparer les dictionnaires de features
    X_train = {
        'technical_input': train_df[technical_cols].values,
        'llm_input': train_df[llm_cols].values,
        'mcp_input': train_df[mcp_cols].values,
        'hmm_input': train_df[hmm_cols].values,
        'instrument_input': train_df[['symbol_id']].values
    }
    
    X_val = {
        'technical_input': val_df[technical_cols].values,
        'llm_input': val_df[llm_cols].values,
        'mcp_input': val_df[mcp_cols].values,
        'hmm_input': val_df[hmm_cols].values,
        'instrument_input': val_df[['symbol_id']].values
    }
    
    # Préparer les dictionnaires de labels
    y_train = {
        'market_regime': train_df['market_regime'].values,
        'sl_tp': train_df[['level_sl', 'level_tp']].values
    }
    
    y_val = {
        'market_regime': val_df['market_regime'].values,
        'sl_tp': val_df[['level_sl', 'level_tp']].values
    }
    
    # Préparer les données de test
    X_test = {
        'technical_input': test_df[technical_cols].values,
        'llm_input': test_df[llm_cols].values,
        'mcp_input': test_df[mcp_cols].values,
        'hmm_input': test_df[hmm_cols].values,
        'instrument_input': test_df[['symbol_id']].values
    }
    
    y_test = {
        'market_regime': test_df['market_regime'].values,
        'sl_tp': test_df[['level_sl', 'level_tp']].values
    }
    
    return X_train, X_val, y_train, y_val, X_test, y_test, feature_names

def train_model(X_train, y_train, X_val, y_val, feature_names, hyperparams, output_dir):
    """
    Entraîne le modèle avec capacité de raisonnement.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        feature_names: Noms des features pour le décodage des explications
        hyperparams: Dictionnaire d'hyperparamètres
        output_dir: Répertoire de sortie pour les résultats
    
    Returns:
        Modèle entraîné et historique d'entraînement
    """
    # Créer le modèle
    logger.info("Création du modèle avec capacité de raisonnement")
    model = build_reasoning_model(
        tech_input_shape=(X_train['technical_input'].shape[1],),
        llm_embedding_dim=X_train['llm_input'].shape[1],
        mcp_input_dim=X_train['mcp_input'].shape[1],
        hmm_input_dim=X_train['hmm_input'].shape[1],
        instrument_vocab_size=10,
        instrument_embedding_dim=8,
        num_market_regime_classes=4,
        num_sl_tp_outputs=2,
        l2_reg=hyperparams.get('l2_reg', 0.001),
        dropout_rate=hyperparams.get('dropout_rate', 0.3),
        use_batch_norm=hyperparams.get('use_batch_norm', True),
        num_reasoning_steps=hyperparams.get('num_reasoning_steps', 3),
        reasoning_units=hyperparams.get('reasoning_units', 128),
        num_attention_heads=hyperparams.get('num_attention_heads', 4),
        attention_key_dim=hyperparams.get('attention_key_dim', 64),
        feature_names=feature_names
    )
    
    # Compiler le modèle
    model = compile_reasoning_model(model, learning_rate=hyperparams.get('learning_rate', 0.001))
    
    # Résumé du modèle
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Entraîner le modèle
    logger.info("Début de l'entraînement du modèle")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparams.get('epochs', 50),
        batch_size=hyperparams.get('batch_size', 32),
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modèle final
    model.save(os.path.join(output_dir, 'final_model.h5'))
    logger.info(f"Modèle final sauvegardé dans {os.path.join(output_dir, 'final_model.h5')}")
    
    # Sauvegarder l'historique d'entraînement
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump({
            key: [float(val) for val in values] for key, values in history.history.items()
        }, f, indent=4)
    
    # Sauvegarder les noms des features
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=4)
    
    return model, history

def evaluate_model(model, X_test, y_test, feature_names, output_dir):
    """
    Évalue le modèle sur l'ensemble de test et génère des explications.
    
    Args:
        model: Modèle entraîné
        X_test, y_test: Données de test
        feature_names: Noms des features pour le décodage des explications
        output_dir: Répertoire de sortie pour les résultats
    
    Returns:
        Dictionnaire de métriques d'évaluation
    """
    # Évaluer le modèle
    logger.info("Évaluation du modèle sur l'ensemble de test")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Sauvegarder les résultats de l'évaluation
    test_metrics = {}
    for i, metric_name in enumerate(model.metrics_names):
        test_metrics[metric_name] = float(test_results[i])
        logger.info(f"{metric_name}: {test_results[i]}")
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Faire des prédictions avec explications
    logger.info("Génération des prédictions et explications")
    predictions = model.predict(X_test)
    
    # Créer un décodeur d'explications
    explanation_decoder = ExplanationDecoder(
        feature_names=feature_names,
        market_regime_names=['sideways', 'bullish', 'bearish', 'volatile']
    )
    
    # Générer des explications pour quelques exemples
    num_examples = min(5, len(X_test['technical_input']))
    examples = []
    
    for i in range(num_examples):
        # Prédictions
        market_regime_pred = np.argmax(predictions['market_regime'][i])
        sl_pred = predictions['sl_tp'][i][0]
        tp_pred = predictions['sl_tp'][i][1]
        
        # Vraies valeurs
        market_regime_true = y_test['market_regime'][i]
        sl_true = y_test['sl_tp'][i][0]
        tp_true = y_test['sl_tp'][i][1]
        
        # Explications
        if 'market_regime_explanation' in predictions:
            market_regime_explanation = explanation_decoder.decode_market_regime_explanation(
                market_regime_pred,
                predictions['market_regime_explanation'][i],
                predictions['attention_scores'][i],
                top_k=3
            )
        else:
            market_regime_explanation = "Explication non disponible"
        
        if 'sl_explanation' in predictions and 'tp_explanation' in predictions:
            sl_explanation, tp_explanation = explanation_decoder.decode_sl_tp_explanation(
                sl_pred, tp_pred,
                predictions['sl_explanation'][i],
                predictions['tp_explanation'][i],
                predictions['attention_scores'][i],
                top_k=3
            )
        else:
            sl_explanation = "Explication non disponible"
            tp_explanation = "Explication non disponible"
        
        # Étapes de raisonnement
        reasoning_steps = []
        for j in range(3):  # Supposons 3 étapes de raisonnement
            if f'reasoning_step_{j}' in predictions:
                reasoning_steps.append(predictions[f'reasoning_step_{j}'][i])
        
        if reasoning_steps:
            reasoning_explanations = explanation_decoder.decode_reasoning_steps(reasoning_steps)
        else:
            reasoning_explanations = ["Étapes de raisonnement non disponibles"]
        
        # Ajouter l'exemple
        examples.append({
            'id': i,
            'predictions': {
                'market_regime': int(market_regime_pred),
                'market_regime_name': explanation_decoder.market_regime_names[market_regime_pred],
                'sl': float(sl_pred),
                'tp': float(tp_pred)
            },
            'true_values': {
                'market_regime': int(market_regime_true),
                'market_regime_name': explanation_decoder.market_regime_names[market_regime_true] if market_regime_true < len(explanation_decoder.market_regime_names) else 'unknown',
                'sl': float(sl_true),
                'tp': float(tp_true)
            },
            'explanations': {
                'market_regime': market_regime_explanation,
                'sl': sl_explanation,
                'tp': tp_explanation,
                'reasoning_steps': reasoning_explanations
            }
        })
    
    # Sauvegarder les exemples avec explications
    with open(os.path.join(output_dir, 'explanation_examples.json'), 'w') as f:
        json.dump(examples, f, indent=4)
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser(description='Entraîne le modèle Morningstar avec capacité de raisonnement.')
    parser.add_argument('--data-path', type=str, required=True, help='Chemin vers le dataset normalisé')
    parser.add_argument('--output-dir', type=str, default='model/reasoning_model', help='Répertoire de sortie pour les résultats')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Proportion des données d\'entraînement à utiliser pour la validation')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--batch-size', type=int, default=32, help='Taille du batch')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Taux d\'apprentissage')
    parser.add_argument('--l2-reg', type=float, default=0.001, help='Coefficient de régularisation L2')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Taux de dropout')
    parser.add_argument('--use-batch-norm', action='store_true', help='Utiliser la normalisation par batch')
    parser.add_argument('--num-reasoning-steps', type=int, default=3, help='Nombre d\'étapes de raisonnement')
    parser.add_argument('--reasoning-units', type=int, default=128, help='Nombre d\'unités dans les couches de raisonnement')
    parser.add_argument('--num-attention-heads', type=int, default=4, help='Nombre de têtes d\'attention')
    parser.add_argument('--attention-key-dim', type=int, default=64, help='Dimension des clefs d\'attention')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Charger les données
    X_train, X_val, y_train, y_val, X_test, y_test, feature_names = load_data(
        args.data_path, args.validation_split
    )
    
    # Hyperparamètres
    hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'l2_reg': args.l2_reg,
        'dropout_rate': args.dropout_rate,
        'use_batch_norm': args.use_batch_norm,
        'num_reasoning_steps': args.num_reasoning_steps,
        'reasoning_units': args.reasoning_units,
        'num_attention_heads': args.num_attention_heads,
        'attention_key_dim': args.attention_key_dim
    }
    
    # Entraîner le modèle
    model, history = train_model(
        X_train, y_train, X_val, y_val, feature_names, hyperparams, args.output_dir
    )
    
    # Évaluer le modèle
    evaluate_model(model, X_test, y_test, feature_names, args.output_dir)

if __name__ == "__main__":
    main()
