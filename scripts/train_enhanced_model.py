#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour entrau00eener le modu00e8le hybride amu00e9lioru00e9 avec les donnu00e9es enrichies.
Ce script charge les donnu00e9es enrichies, pru00e9pare les entru00e9es pour le modu00e8le et l'entrau00eene.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(description="Entrau00eenement du modu00e8le hybride amu00e9lioru00e9")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/enriched/enriched_dataset.parquet",
        help="Chemin vers le dataset enrichi au format parquet"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="models/enhanced",
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
        help="Taille du batch pour l'entrau00eenement"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Utiliser les embeddings LLM si disponibles"
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        help="Utiliser les features HMM"
    )
    return parser.parse_args()

def prepare_data(df, use_llm=False, use_hmm=True):
    """
    Pru00e9pare les donnu00e9es pour l'entrau00eenement du modu00e8le.
    
    Args:
        df: DataFrame avec les donnu00e9es enrichies
        use_llm: Utiliser les embeddings LLM si disponibles
        use_hmm: Utiliser les features HMM
    
    Returns:
        Dictionnaire avec les entru00e9es et sorties pour le modu00e8le
    """
    logger.info("Pru00e9paration des donnu00e9es pour l'entrau00eenement")
    
    # Su00e9parer les ensembles d'entrau00eenement, de validation et de test
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    logger.info(f"Tailles des ensembles - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Pru00e9parer les features techniques disponibles dans le dataset
    tech_features = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bollinger_middle', 'bollinger_std', 'bollinger_upper', 'bollinger_lower', 'bollinger_width',
        'stoch_k', 'stoch_d', 'atr_14',
        'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b', 'ichimoku_chikou'
    ]
    
    # Filtrer pour ne garder que les colonnes qui existent dans le DataFrame
    tech_features = [col for col in tech_features if col in df.columns]
    logger.info(f"Utilisation de {len(tech_features)} features techniques: {', '.join(tech_features[:5])}...")
    
    # Normaliser les features techniques
    scaler = StandardScaler()
    train_tech = scaler.fit_transform(train_df[tech_features].fillna(0))
    val_tech = scaler.transform(val_df[tech_features].fillna(0))
    test_tech = scaler.transform(test_df[tech_features].fillna(0))
    
    # Pru00e9parer les features HMM si demandu00e9
    hmm_features = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    if use_hmm and all(feature in df.columns for feature in hmm_features):
        logger.info("Utilisation des features HMM")
        train_hmm = train_df[hmm_features].fillna(0).values
        val_hmm = val_df[hmm_features].fillna(0).values
        test_hmm = test_df[hmm_features].fillna(0).values
    else:
        logger.warning("Features HMM non disponibles ou non utilisu00e9es, utilisation de vecteurs de zu00e9ros")
        train_hmm = np.zeros((len(train_df), 4))
        val_hmm = np.zeros((len(val_df), 4))
        test_hmm = np.zeros((len(test_df), 4))
    
    # Pru00e9parer les features LLM (CryptoBERT) si demandu00e9
    cryptobert_features = [col for col in df.columns if col.startswith('cryptobert_dim_')]
    
    if use_llm and cryptobert_features:
        logger.info("Utilisation des embeddings CryptoBERT")
        # Cru00e9er un vecteur LLM de dimension 768 en remplissant avec des zu00e9ros
        train_llm = np.zeros((len(train_df), 768))
        val_llm = np.zeros((len(val_df), 768))
        test_llm = np.zeros((len(test_df), 768))
        
        # Remplir les premiu00e8res dimensions avec les valeurs CryptoBERT disponibles
        for i, feature in enumerate(cryptobert_features):
            if i < 768:  # Limiter u00e0 la dimension de l'embedding LLM
                train_llm[:, i] = train_df[feature].fillna(0).values
                val_llm[:, i] = val_df[feature].fillna(0).values
                test_llm[:, i] = test_df[feature].fillna(0).values
    else:
        logger.warning("Embeddings CryptoBERT non disponibles ou non utilisu00e9s, utilisation de vecteurs de zu00e9ros")
        train_llm = np.zeros((len(train_df), 768))
        val_llm = np.zeros((len(val_df), 768))
        test_llm = np.zeros((len(test_df), 768))
    
    # Pru00e9parer les features MCP (Market Context Processor)
    # Utiliser les features de sentiment si disponibles
    sentiment_features = ['sentiment_score', 'sentiment_magnitude', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
    market_info_features = [col for col in df.columns if col.startswith('market_info_')]
    
    mcp_features = sentiment_features + market_info_features
    
    if any(feature in df.columns for feature in mcp_features):
        logger.info("Utilisation des features MCP (sentiment et/ou market info)")
        # Cru00e9er un vecteur MCP de dimension 128 en remplissant avec des zu00e9ros
        train_mcp = np.zeros((len(train_df), 128))
        val_mcp = np.zeros((len(val_df), 128))
        test_mcp = np.zeros((len(test_df), 128))
        
        # Remplir les premiu00e8res dimensions avec les valeurs disponibles
        available_features = [f for f in mcp_features if f in df.columns]
        for i, feature in enumerate(available_features):
            if i < 128:  # Limiter u00e0 la dimension MCP
                train_mcp[:, i] = train_df[feature].fillna(0).values
                val_mcp[:, i] = val_df[feature].fillna(0).values
                test_mcp[:, i] = test_df[feature].fillna(0).values
    else:
        logger.warning("Features MCP non disponibles, utilisation de vecteurs de zu00e9ros")
        train_mcp = np.zeros((len(train_df), 128))
        val_mcp = np.zeros((len(val_df), 128))
        test_mcp = np.zeros((len(test_df), 128))
    
    # Pru00e9parer l'entru00e9e instrument (utiliser 0 pour spot par du00e9faut)
    train_instrument = np.zeros((len(train_df), 1), dtype=np.int64)
    val_instrument = np.zeros((len(val_df), 1), dtype=np.int64)
    test_instrument = np.zeros((len(test_df), 1), dtype=np.int64)
    
    # Pru00e9parer les sorties
    # 1. Signal de trading (market_regime comme proxy)
    train_signal = tf.keras.utils.to_categorical(train_df['market_regime'].fillna(0).astype(int).values, num_classes=5)
    val_signal = tf.keras.utils.to_categorical(val_df['market_regime'].fillna(0).astype(int).values, num_classes=5)
    test_signal = tf.keras.utils.to_categorical(test_df['market_regime'].fillna(0).astype(int).values, num_classes=5)
    
    # 2. Market regime
    train_regime = tf.keras.utils.to_categorical(train_df['market_regime'].fillna(0).astype(int).values, num_classes=4)
    val_regime = tf.keras.utils.to_categorical(val_df['market_regime'].fillna(0).astype(int).values, num_classes=4)
    test_regime = tf.keras.utils.to_categorical(test_df['market_regime'].fillna(0).astype(int).values, num_classes=4)
    
    # 3. SL/TP
    train_sl_tp = np.column_stack((train_df['level_sl'].fillna(-0.02).values, train_df['level_tp'].fillna(0.03).values))
    val_sl_tp = np.column_stack((val_df['level_sl'].fillna(-0.02).values, val_df['level_tp'].fillna(0.03).values))
    test_sl_tp = np.column_stack((test_df['level_sl'].fillna(-0.02).values, test_df['level_tp'].fillna(0.03).values))
    
    # Cru00e9er les dictionnaires d'entru00e9es et sorties
    train_inputs = {
        'technical_input': train_tech,
        'llm_input': train_llm,
        'mcp_input': train_mcp,
        'hmm_input': train_hmm,
        'instrument_input': train_instrument
    }
    
    val_inputs = {
        'technical_input': val_tech,
        'llm_input': val_llm,
        'mcp_input': val_mcp,
        'hmm_input': val_hmm,
        'instrument_input': val_instrument
    }
    
    test_inputs = {
        'technical_input': test_tech,
        'llm_input': test_llm,
        'mcp_input': test_mcp,
        'hmm_input': test_hmm,
        'instrument_input': test_instrument
    }
    
    train_outputs = {
        'signal': train_signal,
        'market_regime': train_regime,
        'sl_tp': train_sl_tp
    }
    
    val_outputs = {
        'signal': val_signal,
        'market_regime': val_regime,
        'sl_tp': val_sl_tp
    }
    
    test_outputs = {
        'signal': test_signal,
        'market_regime': test_regime,
        'sl_tp': test_sl_tp
    }
    
    return {
        'train_inputs': train_inputs,
        'val_inputs': val_inputs,
        'test_inputs': test_inputs,
        'train_outputs': train_outputs,
        'val_outputs': val_outputs,
        'test_outputs': test_outputs,
        'tech_input_shape': (train_tech.shape[1],)
    }

def build_model(tech_input_shape, use_llm=False):
    """
    Construit le modu00e8le hybride amu00e9lioru00e9.
    
    Args:
        tech_input_shape: Shape des features techniques
        use_llm: Utiliser les embeddings LLM
    
    Returns:
        Modu00e8le Keras compilu00e9
    """
    logger.info("Construction du modu00e8le hybride amu00e9lioru00e9")
    
    # Du00e9finir les sorties actives
    active_outputs = ['signal', 'market_regime', 'sl_tp']
    
    # Construire le modu00e8le
    model = build_enhanced_hybrid_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=768,
        mcp_input_dim=128,
        hmm_input_dim=4,
        instrument_vocab_size=10,
        instrument_embedding_dim=8,
        num_trading_classes=5,
        num_market_regime_classes=4,
        num_volatility_quantiles=3,
        num_sl_tp_outputs=2,
        active_outputs=active_outputs,
        use_llm=use_llm,
        llm_fallback_strategy='technical_projection'
    )
    
    # Compiler le modu00e8le avec des pertes et mu00e9triques appropriu00e9es
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'signal': 'categorical_crossentropy',
            'market_regime': 'categorical_crossentropy',
            'sl_tp': 'mse'
        },
        metrics={
            'signal': ['accuracy'],
            'market_regime': ['accuracy'],
            'sl_tp': ['mae']
        },
        loss_weights={
            'signal': 1.0,
            'market_regime': 1.0,
            'sl_tp': 0.5
        }
    )
    
    return model

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Cru00e9er le ru00e9pertoire de sortie s'il n'existe pas
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # Charger les donnu00e9es enrichies
    logger.info(f"Chargement des donnu00e9es depuis {args.data_path}")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Donnu00e9es chargu00e9es: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Pru00e9parer les donnu00e9es
    data = prepare_data(df, use_llm=args.use_llm, use_hmm=args.use_hmm)
    
    # Construire le modu00e8le
    model = build_model(data['tech_input_shape'], use_llm=args.use_llm)
    
    # Ru00e9sumu00e9 du modu00e8le
    model.summary()
    
    # Du00e9finir les callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_output_dir, 'best_model.h5'),
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
            log_dir=os.path.join(args.model_output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S')),
            histogram_freq=1
        )
    ]
    
    # Entrau00eener le modu00e8le
    logger.info(f"Du00e9but de l'entrau00eenement pour {args.epochs} u00e9poques")
    history = model.fit(
        data['train_inputs'],
        data['train_outputs'],
        validation_data=(data['val_inputs'], data['val_outputs']),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # u00c9valuer le modu00e8le sur l'ensemble de test
    logger.info("\u00c9valuation du modu00e8le sur l'ensemble de test")
    test_results = model.evaluate(
        data['test_inputs'],
        data['test_outputs'],
        verbose=1
    )
    
    # Sauvegarder le modu00e8le final
    model_path = os.path.join(args.model_output_dir, 'final_model.h5')
    model.save(model_path)
    logger.info(f"Modu00e8le final sauvegardu00e9 dans {model_path}")
    
    # Sauvegarder les ru00e9sultats de l'u00e9valuation
    results_path = os.path.join(args.model_output_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        for i, metric_name in enumerate(model.metrics_names):
            f.write(f"{metric_name}: {test_results[i]}\n")
    logger.info(f"Ru00e9sultats de l'u00e9valuation sauvegardu00e9s dans {results_path}")

if __name__ == "__main__":
    main()
