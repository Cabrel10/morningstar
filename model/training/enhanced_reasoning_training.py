#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entrau00eenement amu00e9lioru00e9 pour le modu00e8le avec capacitu00e9 de raisonnement.
Ce script utilise les donnu00e9es enrichies pour entrau00eener un modu00e8le plus pru00e9cis.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.reasoning.reasoning_module import ReasoningModule, ExplanationDecoder

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
    parser = argparse.ArgumentParser(description="Entrau00eenement du modu00e8le avec capacitu00e9 de raisonnement")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/enriched/enriched_dataset.parquet",
        help="Chemin vers le dataset enrichi"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model/enhanced_reasoning_model",
        help="Ru00e9pertoire de sortie pour le modu00e8le"
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
        help="Taille du batch"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Taux d'apprentissage"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.01,
        help="Coefficient de ru00e9gularisation L2"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Taux de dropout"
    )
    parser.add_argument(
        "--use-batch-norm",
        action="store_true",
        help="Utiliser la normalisation par batch"
    )
    parser.add_argument(
        "--num-reasoning-steps",
        type=int,
        default=3,
        help="Nombre d'u00e9tapes de raisonnement"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Proportion des donnu00e9es d'entrau00eenement u00e0 utiliser pour la validation"
    )
    
    return parser.parse_args()

def load_data(data_path):
    """
    Charge les donnu00e9es enrichies et les divise en ensembles d'entrau00eenement et de test.
    
    Args:
        data_path: Chemin vers le dataset enrichi
    
    Returns:
        X_train, X_val, y_train, y_val, X_test, y_test, feature_names: Dictionnaires de features et labels, et noms des features
    """
    logger.info(f"Chargement des donnu00e9es depuis {data_path}")
    
    # Charger les donnu00e9es
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Su00e9parer les ensembles d'entrau00eenement, de validation et de test
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    val_df = df[df['split'] == 'val'].drop(columns=['split'])
    test_df = df[df['split'] == 'test'].drop(columns=['split'])
    
    logger.info(f"Ensemble d'entrau00eenement: {len(train_df)} lignes, Ensemble de validation: {len(val_df)} lignes, Ensemble de test: {len(test_df)} lignes")
    
    # Identifier les diffu00e9rents types de colonnes
    exclude_cols = ['timestamp', 'symbol', 'market_regime', 'level_sl', 'level_tp',
                   'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2',
                   'future_return_1d', 'future_return_3d', 'future_return_7d']
    
    # Colonnes techniques (toutes les colonnes qui ne sont pas exclues et qui ne commencent pas par certains pru00e9fixes)
    technical_cols = [col for col in train_df.columns if col not in exclude_cols and 
                     not col.startswith('cryptobert_embedding_') and 
                     not col.startswith('sentiment_') and 
                     not col.startswith('global_') and 
                     not col.startswith('BTC_') and 
                     not col.startswith('ETH_')]
    
    # Colonnes de sentiment
    sentiment_cols = [col for col in train_df.columns if col.startswith('sentiment_')]
    
    # Colonnes d'embeddings CryptoBERT
    cryptobert_cols = [col for col in train_df.columns if col.startswith('cryptobert_embedding_')]
    
    # Colonnes de marchu00e9 global
    market_cols = [col for col in train_df.columns if col.startswith('global_') or 
                  col.startswith('BTC_') or col.startswith('ETH_')]
    
    # Colonnes HMM
    hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Convertir les symboles en entiers pour l'embedding
    symbol_mapping = {symbol: i for i, symbol in enumerate(train_df['symbol'].unique())}
    train_df['symbol_id'] = train_df['symbol'].map(symbol_mapping)
    val_df['symbol_id'] = val_df['symbol'].map(symbol_mapping)
    test_df['symbol_id'] = test_df['symbol'].map(symbol_mapping)
    
    # Cru00e9er un dictionnaire de noms de features
    feature_names = technical_cols + sentiment_cols + cryptobert_cols + market_cols + hmm_cols
    
    # Pru00e9parer les dictionnaires de features
    X_train = {
        'technical_input': train_df[technical_cols].values,
        'sentiment_input': train_df[sentiment_cols].values if sentiment_cols else np.zeros((len(train_df), 1)),
        'cryptobert_input': train_df[cryptobert_cols].values if cryptobert_cols else np.zeros((len(train_df), 1)),
        'market_input': train_df[market_cols].values if market_cols else np.zeros((len(train_df), 1)),
        'hmm_input': train_df[hmm_cols].values if all(col in train_df.columns for col in hmm_cols) else np.zeros((len(train_df), 4)),
        'instrument_input': train_df[['symbol_id']].values
    }
    
    X_val = {
        'technical_input': val_df[technical_cols].values,
        'sentiment_input': val_df[sentiment_cols].values if sentiment_cols else np.zeros((len(val_df), 1)),
        'cryptobert_input': val_df[cryptobert_cols].values if cryptobert_cols else np.zeros((len(val_df), 1)),
        'market_input': val_df[market_cols].values if market_cols else np.zeros((len(val_df), 1)),
        'hmm_input': val_df[hmm_cols].values if all(col in val_df.columns for col in hmm_cols) else np.zeros((len(val_df), 4)),
        'instrument_input': val_df[['symbol_id']].values
    }
    
    X_test = {
        'technical_input': test_df[technical_cols].values,
        'sentiment_input': test_df[sentiment_cols].values if sentiment_cols else np.zeros((len(test_df), 1)),
        'cryptobert_input': test_df[cryptobert_cols].values if cryptobert_cols else np.zeros((len(test_df), 1)),
        'market_input': test_df[market_cols].values if market_cols else np.zeros((len(test_df), 1)),
        'hmm_input': test_df[hmm_cols].values if all(col in test_df.columns for col in hmm_cols) else np.zeros((len(test_df), 4)),
        'instrument_input': test_df[['symbol_id']].values
    }
    
    # Pru00e9parer les dictionnaires de labels
    y_train = {
        'market_regime': train_df['market_regime'].values,
        'sl_tp': train_df[['level_sl', 'level_tp']].values
    }
    
    y_val = {
        'market_regime': val_df['market_regime'].values,
        'sl_tp': val_df[['level_sl', 'level_tp']].values
    }
    
    y_test = {
        'market_regime': test_df['market_regime'].values,
        'sl_tp': test_df[['level_sl', 'level_tp']].values
    }
    
    return X_train, X_val, y_train, y_val, X_test, y_test, feature_names

def build_enhanced_reasoning_model(X_train, feature_names, hyperparams):
    """
    Construit un modu00e8le de raisonnement amu00e9lioru00e9 avec des entru00e9es pour les donnu00e9es enrichies.
    
    Args:
        X_train: Donnu00e9es d'entrau00eenement pour du00e9terminer les dimensions d'entru00e9e
        feature_names: Noms des features
        hyperparams: Hyperparamu00e8tres du modu00e8le
    
    Returns:
        Modu00e8le de raisonnement amu00e9lioru00e9
    """
    logger.info("Construction du modu00e8le de raisonnement amu00e9lioru00e9")
    
    # Du00e9terminer les dimensions d'entru00e9e
    tech_input_shape = X_train['technical_input'].shape[1:]
    sentiment_input_shape = X_train['sentiment_input'].shape[1:]
    cryptobert_input_shape = X_train['cryptobert_input'].shape[1:]
    market_input_shape = X_train['market_input'].shape[1:]
    hmm_input_shape = X_train['hmm_input'].shape[1:]
    instrument_vocab_size = np.max(X_train['instrument_input']) + 1
    
    # Construire le modu00e8le
    model = build_reasoning_model(
        tech_input_shape=tech_input_shape,
        sentiment_input_dim=sentiment_input_shape[0] if len(sentiment_input_shape) > 0 else 1,
        cryptobert_input_dim=cryptobert_input_shape[0] if len(cryptobert_input_shape) > 0 else 1,
        market_input_dim=market_input_shape[0] if len(market_input_shape) > 0 else 1,
        hmm_input_dim=hmm_input_shape[0] if len(hmm_input_shape) > 0 else 4,
        instrument_vocab_size=int(instrument_vocab_size),
        instrument_embedding_dim=8,
        num_market_regime_classes=2,  # 0: stable/baissier, 1: haussier
        num_sl_tp_outputs=2,
        l2_reg=hyperparams['l2_reg'],
        dropout_rate=hyperparams['dropout_rate'],
        use_batch_norm=hyperparams['use_batch_norm'],
        num_reasoning_steps=hyperparams['num_reasoning_steps'],
        reasoning_units=128,
        num_attention_heads=4,
        attention_key_dim=64,
        feature_names=feature_names
    )
    
    # Compiler le modu00e8le
    model = compile_reasoning_model(model, learning_rate=hyperparams['learning_rate'])
    
    # Afficher un ru00e9sumu00e9 du modu00e8le
    model.summary()
    
    return model

def train_model(X_train, y_train, X_val, y_val, feature_names, hyperparams, output_dir):
    """
    Entrau00eene le modu00e8le de raisonnement amu00e9lioru00e9.
    
    Args:
        X_train: Donnu00e9es d'entrau00eenement
        y_train: Labels d'entrau00eenement
        X_val: Donnu00e9es de validation
        y_val: Labels de validation
        feature_names: Noms des features
        hyperparams: Hyperparamu00e8tres du modu00e8le
        output_dir: Ru00e9pertoire de sortie pour le modu00e8le
    
    Returns:
        Modu00e8le entrau00eenu00e9 et historique d'entrau00eenement
    """
    logger.info("Entrau00eenement du modu00e8le de raisonnement amu00e9lioru00e9")
    
    # Construire le modu00e8le
    model = build_enhanced_reasoning_model(X_train, feature_names, hyperparams)
    
    # Cru00e9er le ru00e9pertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Cru00e9er les callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Entrau00eener le modu00e8le
    logger.info("Du00e9but de l'entrau00eenement du modu00e8le")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modu00e8le final
    model.save(os.path.join(output_dir, 'final_model.h5'))
    logger.info(f"Modu00e8le final sauvegardu00e9 dans {os.path.join(output_dir, 'final_model.h5')}")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    u00c9value le modu00e8le sur l'ensemble de test.
    
    Args:
        model: Modu00e8le entrau00eenu00e9
        X_test: Donnu00e9es de test
        y_test: Labels de test
    
    Returns:
        Mu00e9triques d'u00e9valuation
    """
    logger.info("u00c9valuation du modu00e8le sur l'ensemble de test")
    
    # u00c9valuer le modu00e8le
    metrics = model.evaluate(X_test, y_test, verbose=1)
    
    # Afficher les mu00e9triques
    for i, metric_name in enumerate(model.metrics_names):
        logger.info(f"{metric_name}: {metrics[i]}")
    
    return metrics

def generate_explanations(model, X_test, feature_names, output_dir, num_examples=5):
    """
    Gu00e9nu00e8re des explications pour les pru00e9dictions du modu00e8le.
    
    Args:
        model: Modu00e8le entrau00eenu00e9
        X_test: Donnu00e9es de test
        feature_names: Noms des features
        output_dir: Ru00e9pertoire de sortie pour les explications
        num_examples: Nombre d'exemples u00e0 expliquer
    
    Returns:
        DataFrame avec les explications
    """
    logger.info("Gu00e9nu00e9ration des pru00e9dictions et explications")
    
    # Limiter le nombre d'exemples
    num_examples = min(num_examples, len(X_test['technical_input']))
    
    # Cru00e9er un sous-ensemble pour les exemples
    X_examples = {}
    for key, value in X_test.items():
        X_examples[key] = value[:num_examples]
    
    # Faire des pru00e9dictions
    predictions = model.predict(X_examples)
    
    # Extraire les pru00e9dictions et les sorties de raisonnement
    if isinstance(predictions, dict):
        market_regime_pred = predictions['market_regime']
        sl_tp_pred = predictions['sl_tp']
        reasoning_output = predictions.get('reasoning_output', None)
        attention_scores = predictions.get('attention_scores', None)
    else:
        market_regime_pred = predictions[0]
        sl_tp_pred = predictions[1]
        reasoning_output = predictions[2] if len(predictions) > 2 else None
        attention_scores = predictions[3] if len(predictions) > 3 else None
    
    # Convertir les pru00e9dictions de ru00e9gime de marchu00e9 en classes
    market_regime_classes = np.argmax(market_regime_pred, axis=1)
    
    # Cru00e9er un du00e9codeur d'explications
    market_regime_names = ['stable/baissier', 'haussier']
    decoder = ExplanationDecoder(feature_names=feature_names, market_regime_names=market_regime_names)
    
    # Gu00e9nu00e9rer des explications pour chaque exemple
    explanations = []
    for i in range(num_examples):
        example_dict = {
            'example_id': i,
            'market_regime_pred': market_regime_classes[i],
            'market_regime_confidence': np.max(market_regime_pred[i]),
            'sl_pred': sl_tp_pred[i][0],
            'tp_pred': sl_tp_pred[i][1],
        }
        
        # Ajouter des explications si disponibles
        if reasoning_output is not None and attention_scores is not None:
            try:
                # Extraire les scores d'attention pour cet exemple
                example_attention = attention_scores[i] if len(attention_scores.shape) > 2 else attention_scores
                
                # Gu00e9nu00e9rer des explications
                market_regime_explanation = decoder.decode_market_regime_explanation(
                    market_regime_classes[i],
                    reasoning_output[i],
                    example_attention,
                    top_k=5
                )
                
                sl_tp_explanation = decoder.decode_sl_tp_explanation(
                    sl_tp_pred[i][0],
                    sl_tp_pred[i][1],
                    reasoning_output[i],
                    reasoning_output[i],
                    example_attention,
                    top_k=5
                )
                
                example_dict['market_regime_explanation'] = market_regime_explanation
                example_dict['sl_explanation'] = sl_tp_explanation[0]
                example_dict['tp_explanation'] = sl_tp_explanation[1]
            except Exception as e:
                logger.warning(f"Erreur lors de la gu00e9nu00e9ration des explications pour l'exemple {i}: {str(e)}")
                example_dict['market_regime_explanation'] = f"Le modu00e8le pru00e9dit un marchu00e9 {market_regime_names[market_regime_classes[i]]} avec une confiance de {np.max(market_regime_pred[i]):.2f}."
                example_dict['sl_explanation'] = f"Stop loss placu00e9 u00e0 {sl_tp_pred[i][0]:.4f}."
                example_dict['tp_explanation'] = f"Take profit placu00e9 u00e0 {sl_tp_pred[i][1]:.4f}."
        
        explanations.append(example_dict)
    
    # Cru00e9er un DataFrame avec les explications
    explanations_df = pd.DataFrame(explanations)
    
    # Sauvegarder les explications
    os.makedirs(output_dir, exist_ok=True)
    explanations_path = os.path.join(output_dir, 'explanations.csv')
    explanations_df.to_csv(explanations_path, index=False)
    logger.info(f"Explications sauvegardu00e9es dans {explanations_path}")
    
    return explanations_df

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Charger les donnu00e9es
    X_train, X_val, y_train, y_val, X_test, y_test, feature_names = load_data(args.data_path)
    
    # Pru00e9parer les hyperparamu00e8tres
    hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'l2_reg': args.l2_reg,
        'dropout_rate': args.dropout_rate,
        'use_batch_norm': args.use_batch_norm,
        'num_reasoning_steps': args.num_reasoning_steps
    }
    
    # Entrau00eener le modu00e8le
    model, history = train_model(
        X_train, y_train, X_val, y_val, feature_names, hyperparams, args.output_dir
    )
    
    # u00c9valuer le modu00e8le
    evaluate_model(model, X_test, y_test)
    
    # Gu00e9nu00e9rer des explications
    generate_explanations(model, X_test, feature_names, args.output_dir)

if __name__ == "__main__":
    main()
