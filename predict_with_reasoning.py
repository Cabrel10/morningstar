#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour faire des pru00e9dictions avec le modu00e8le de raisonnement.
Ce script utilise le modu00e8le entrau00eenu00e9 pour faire des pru00e9dictions sur de nouvelles donnu00e9es.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, StandardScaler

from model.architecture.reasoning_model import build_reasoning_model, compile_reasoning_model
from model.reasoning.reasoning_module import ReasoningModule

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
    parser = argparse.ArgumentParser(description="Pru00e9dictions avec le modu00e8le de raisonnement")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/real/final_dataset.parquet",
        help="Chemin vers le dataset de donnu00e9es ru00e9elles"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/predictions",
        help="Ru00e9pertoire de sortie pour les pru00e9dictions"
    )
    
    return parser.parse_args()

def preprocess_data(data_path):
    """
    Pru00e9traite les donnu00e9es pour les pru00e9dictions.
    
    Args:
        data_path: Chemin vers le dataset
    
    Returns:
        Donnu00e9es pru00e9traitu00e9es et noms des features
    """
    # Charger les donnu00e9es
    logger.info(f"Chargement des donnu00e9es depuis {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Identifier les diffu00e9rents types de colonnes
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    technical_cols = [col for col in df.columns if col not in [
        'timestamp', 'symbol', 'market_regime', 'level_sl', 'level_tp',
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'split'
    ] and not col.startswith('llm_') and not col.startswith('mcp_') 
      and not col.startswith('sentiment_') and not col.startswith('cryptobert_')
      and not col.startswith('market_info_')]
    
    # Exclure les colonnes non numu00e9riques
    exclude_cols = ['timestamp', 'symbol', 'split']
    technical_cols = [col for col in technical_cols if col not in exclude_cols and 
                     df[col].dtype != 'object' and 
                     not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Normaliser les donnu00e9es manuellement
    logger.info("Normalisation des donnu00e9es")
    df_norm = df.copy()
    
    # Normalisation simple des donnu00e9es numu00e9riques
    for col in price_cols + technical_cols:
        if col in df.columns and df[col].dtype != 'object' and not pd.api.types.is_datetime64_any_dtype(df[col]):
            # Normalisation robuste (similaire u00e0 RobustScaler)
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df_norm[col] = (df[col] - q1) / iqr
            else:
                # Fallback u00e0 la normalisation standard
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                else:
                    df_norm[col] = 0
    
    # Colonnes LLM (si pru00e9sentes)
    llm_cols = [col for col in df.columns if col.startswith('llm_')]
    if not llm_cols:  # Si pas de colonnes LLM, cru00e9er un vecteur vide
        df_norm['llm_dummy'] = 0.0
        llm_cols = ['llm_dummy']
    
    # Colonnes MCP (si pru00e9sentes)
    mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
    if not mcp_cols:  # Si pas de colonnes MCP, cru00e9er un vecteur vide
        df_norm['mcp_dummy'] = 0.0
        mcp_cols = ['mcp_dummy']
    
    # Colonnes HMM
    hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Nouvelles colonnes de sentiment (si pru00e9sentes)
    sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
    if not sentiment_cols:  # Si pas de colonnes de sentiment, cru00e9er un vecteur vide
        df_norm['sentiment_dummy'] = 0.0
        sentiment_cols = ['sentiment_dummy']
    
    # Nouvelles colonnes CryptoBERT (si pru00e9sentes)
    cryptobert_cols = [col for col in df.columns if col.startswith('cryptobert_')]
    if not cryptobert_cols:  # Si pas de colonnes CryptoBERT, cru00e9er un vecteur vide
        df_norm['cryptobert_dummy'] = 0.0
        cryptobert_cols = ['cryptobert_dummy']
    
    # Nouvelles colonnes d'informations de marchu00e9 (si pru00e9sentes)
    market_info_cols = [col for col in df.columns if col.startswith('market_info_')]
    if not market_info_cols:  # Si pas de colonnes d'informations de marchu00e9, cru00e9er un vecteur vide
        df_norm['market_info_dummy'] = 0.0
        market_info_cols = ['market_info_dummy']
    
    # Convertir les symboles en entiers pour l'embedding
    symbol_mapping = {symbol: i for i, symbol in enumerate(df_norm['symbol'].unique())}
    df_norm['symbol_id'] = df_norm['symbol'].map(symbol_mapping)
    
    # Cru00e9er un dictionnaire de noms de features
    feature_names = technical_cols + llm_cols + mcp_cols + hmm_cols + sentiment_cols + cryptobert_cols + market_info_cols
    
    # Pru00e9parer les donnu00e9es d'entru00e9e
    X = {
        'technical_input': df_norm[technical_cols].values,
        'llm_input': df_norm[llm_cols].values,
        'mcp_input': df_norm[mcp_cols].values,
        'hmm_input': df_norm[hmm_cols].values,
        'instrument_input': df_norm[['symbol_id']].values,
        'sentiment_input': df_norm[sentiment_cols].values,
        'cryptobert_input': df_norm[cryptobert_cols].values,
        'market_input': df_norm[market_info_cols].values
    }
    
    return X, df_norm, feature_names

def create_model(X, feature_names):
    """
    Cru00e9e un modu00e8le de raisonnement pour les pru00e9dictions.
    
    Args:
        X: Donnu00e9es d'entru00e9e
        feature_names: Noms des features
    
    Returns:
        Modu00e8le de raisonnement
    """
    # Du00e9terminer les dimensions d'entru00e9e
    tech_input_shape = X['technical_input'].shape[1:]
    llm_input_shape = X['llm_input'].shape[1:]
    mcp_input_shape = X['mcp_input'].shape[1:]
    hmm_input_shape = X['hmm_input'].shape[1:]
    sentiment_input_shape = X['sentiment_input'].shape[1:]
    cryptobert_input_shape = X['cryptobert_input'].shape[1:]
    market_input_shape = X['market_input'].shape[1:]
    
    # Charger le modu00e8le entrau00eenu00e9 ou le reconstruire
    model_path = os.path.join('model', 'enhanced_reasoning_model', 'model.keras')
    weights_path = os.path.join('model', 'enhanced_reasoning_model', 'weights.h5')
    
    if os.path.exists(model_path):
        logger.info(f"Chargement du modu00e8le depuis {model_path}")
        try:
            # Essayer de charger le modu00e8le complet
            custom_objects = {'ReasoningModule': ReasoningModule}
            model = load_model(model_path, custom_objects=custom_objects)
            logger.info("Modu00e8le chargu00e9 avec succu00e8s")
            return model
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du modu00e8le: {e}")
            logger.info("Reconstruction du modu00e8le et chargement des poids")
    
    # Si le modu00e8le n'existe pas ou n'a pas pu u00eatre chargu00e9, le reconstruire
    logger.info("Construction du modu00e8le de raisonnement")
    model = build_reasoning_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=llm_input_shape[0],
        mcp_input_dim=mcp_input_shape[0],
        hmm_input_dim=hmm_input_shape[0],
        sentiment_input_dim=sentiment_input_shape[0],
        cryptobert_input_dim=cryptobert_input_shape[0],
        market_input_dim=market_input_shape[0],
        instrument_vocab_size=100,  # Valeur arbitraire, sera ajustu00e9e par les poids
        instrument_embedding_dim=8,
        num_market_regime_classes=2,
        num_sl_tp_outputs=2,
        l2_reg=0.001,
        dropout_rate=0.3,
        use_batch_norm=True,
        num_reasoning_steps=3,
        reasoning_units=128,
        feature_names=feature_names
    )
    
    # Compiler le modu00e8le
    compile_reasoning_model(model)
    
    # Charger les poids si disponibles
    if os.path.exists(weights_path):
        logger.info(f"Chargement des poids depuis {weights_path}")
        model.load_weights(weights_path)
    
    return model

def generate_predictions(model, X, df):
    """
    Gu00e9nu00e8re des pru00e9dictions avec le modu00e8le.
    
    Args:
        model: Modu00e8le entrau00eenu00e9
        X: Donnu00e9es d'entru00e9e
        df: DataFrame original
    
    Returns:
        DataFrame avec les pru00e9dictions
    """
    # Faire des pru00e9dictions
    logger.info("Gu00e9nu00e9ration des pru00e9dictions")
    predictions = model.predict(X)
    
    # Extraire les pru00e9dictions
    if isinstance(predictions, dict):
        market_regime_pred = predictions['market_regime']
        sl_tp_pred = predictions['sl_tp']
    else:
        market_regime_pred = predictions[0]
        sl_tp_pred = predictions[1]
    
    # Convertir les pru00e9dictions de ru00e9gime de marchu00e9 en classes
    market_regime_classes = np.argmax(market_regime_pred, axis=1)
    market_regime_confidence = np.max(market_regime_pred, axis=1)
    
    # Cru00e9er un DataFrame avec les pru00e9dictions
    results = []
    for i in range(len(df)):
        result = {
            'timestamp': df['timestamp'].iloc[i],
            'symbol': df['symbol'].iloc[i],
            'close': df['close'].iloc[i],
            'market_regime_pred': market_regime_classes[i],
            'market_regime_confidence': market_regime_confidence[i],
            'sl_pred': sl_tp_pred[i][0],
            'tp_pred': sl_tp_pred[i][1],
            'risk_reward_ratio': abs(sl_tp_pred[i][1] / sl_tp_pred[i][0]) if sl_tp_pred[i][0] != 0 else 0
        }
        results.append(result)
    
    return pd.DataFrame(results)

def generate_trading_insights(predictions_df):
    """
    Gu00e9nu00e8re des insights de trading u00e0 partir des pru00e9dictions.
    
    Args:
        predictions_df: DataFrame avec les pru00e9dictions
    
    Returns:
        DataFrame avec les insights de trading
    """
    logger.info("Gu00e9nu00e9ration des insights de trading")
    
    # Ajouter des insights de trading
    predictions_df['trading_decision'] = predictions_df.apply(
        lambda row: 'BUY' if row['market_regime_pred'] == 1 and row['risk_reward_ratio'] >= 2.0 else
                   'SELL' if row['market_regime_pred'] == 0 and row['risk_reward_ratio'] >= 2.0 else
                   'HOLD',
        axis=1
    )
    
    predictions_df['trading_confidence'] = predictions_df.apply(
        lambda row: row['market_regime_confidence'] * min(row['risk_reward_ratio'] / 3.0, 1.0),
        axis=1
    )
    
    predictions_df['reasoning'] = predictions_df.apply(
        lambda row: f"Marchu00e9 {'haussier' if row['market_regime_pred'] == 1 else 'stable/baissier'} " +
                   f"(confiance: {row['market_regime_confidence']:.2f}). " +
                   f"Ratio risque/ru00e9compense: {row['risk_reward_ratio']:.2f}. " +
                   f"Stop loss u00e0 {row['sl_pred']:.4f}, Take profit u00e0 {row['tp_pred']:.4f}.",
        axis=1
    )
    
    return predictions_df

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Cru00e9er le ru00e9pertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pru00e9traiter les donnu00e9es
    X, df_norm, feature_names = preprocess_data(args.data_path)
    
    # Cru00e9er le modu00e8le
    model = create_model(X, feature_names)
    
    # Gu00e9nu00e9rer des pru00e9dictions
    predictions_df = generate_predictions(model, X, df_norm)
    
    # Gu00e9nu00e9rer des insights de trading
    insights_df = generate_trading_insights(predictions_df)
    
    # Sauvegarder les pru00e9dictions
    predictions_path = os.path.join(args.output_dir, 'trading_predictions.csv')
    insights_df.to_csv(predictions_path, index=False)
    logger.info(f"Pru00e9dictions sauvegardu00e9es dans {predictions_path}")
    
    # Afficher quelques exemples de pru00e9dictions
    logger.info("\nExemples de pru00e9dictions:")
    for i, (_, row) in enumerate(insights_df.head(5).iterrows()):
        logger.info(f"\nExemple {i+1}:")
        logger.info(f"Timestamp: {row['timestamp']}")
        logger.info(f"Symbole: {row['symbol']}")
        logger.info(f"Prix de clu00f4ture: {row['close']}")
        logger.info(f"Du00e9cision de trading: {row['trading_decision']} (confiance: {row['trading_confidence']:.2f})")
        logger.info(f"Raisonnement: {row['reasoning']}")

if __name__ == "__main__":
    main()
