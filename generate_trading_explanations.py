#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour gu00e9nu00e9rer des explications de trading u00e0 partir du modu00e8le entrau00eenu00e9.
Ce script utilise le modu00e8le de raisonnement pour gu00e9nu00e9rer des explications pour les du00e9cisions de trading.
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
    parser = argparse.ArgumentParser(description="Gu00e9nu00e9ration d'explications de trading")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/real/final_dataset.parquet",
        help="Chemin vers le dataset de donnu00e9es ru00e9elles"
    )
    parser.add_argument(
        "--scalers-path",
        type=str,
        default="data/processed/normalized/scalers/scalers.npz",
        help="Chemin vers les scalers enregistru00e9s"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/explanations",
        help="Ru00e9pertoire de sortie pour les explications"
    )
    
    return parser.parse_args()

def preprocess_data(data_path, scalers_path):
    """
    Pru00e9traite les donnu00e9es pour les pru00e9dictions.
    
    Args:
        data_path: Chemin vers le dataset
        scalers_path: Chemin vers les scalers enregistru00e9s
    
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
    
    # Charger les scalers
    logger.info(f"Chargement des scalers depuis {scalers_path}")
    scalers = np.load(scalers_path, allow_pickle=True)
    price_scaler = scalers['price_scaler'].item()
    tech_scaler = scalers['tech_scaler'].item()
    mcp_scaler = scalers.get('mcp_scaler', None)
    sl_tp_scaler = scalers['sl_tp_scaler'].item()
    
    # Identifier les diffu00e9rents types de colonnes
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    technical_cols = [col for col in df.columns if col not in [
        'timestamp', 'symbol', 'market_regime', 'level_sl', 'level_tp',
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2', 'split'
    ] and not col.startswith('llm_') and not col.startswith('mcp_')]
    
    # Exclure les colonnes non numu00e9riques
    exclude_cols = ['timestamp', 'symbol', 'split']
    technical_cols = [col for col in technical_cols if col not in exclude_cols and 
                     df[col].dtype != 'object' and 
                     not pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Normaliser les donnu00e9es manuellement plutu00f4t qu'avec les scalers
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
    
    # Convertir les symboles en entiers pour l'embedding
    symbol_mapping = {symbol: i for i, symbol in enumerate(df_norm['symbol'].unique())}
    df_norm['symbol_id'] = df_norm['symbol'].map(symbol_mapping)
    
    # Cru00e9er un dictionnaire de noms de features
    feature_names = technical_cols + llm_cols + mcp_cols + hmm_cols
    
    # Pru00e9parer les donnu00e9es d'entru00e9e
    X = {
        'technical_input': df_norm[technical_cols].values,
        'llm_input': df_norm[llm_cols].values,
        'mcp_input': df_norm[mcp_cols].values,
        'hmm_input': df_norm[hmm_cols].values,
        'instrument_input': df_norm[['symbol_id']].values
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
    instrument_vocab_size = np.max(X['instrument_input']) + 1
    
    # Construire le modu00e8le
    logger.info("Construction du modu00e8le de raisonnement")
    model = build_reasoning_model(
        tech_input_shape=tech_input_shape,
        llm_embedding_dim=llm_input_shape[0] if len(llm_input_shape) > 0 else 1,
        mcp_input_dim=mcp_input_shape[0] if len(mcp_input_shape) > 0 else 1,
        hmm_input_dim=hmm_input_shape[0] if len(hmm_input_shape) > 0 else 4,
        instrument_vocab_size=int(instrument_vocab_size),
        instrument_embedding_dim=8,
        num_market_regime_classes=2,  # Adaptu00e9 u00e0 notre cas
        num_sl_tp_outputs=2,
        l2_reg=0.01,
        dropout_rate=0.3,
        use_batch_norm=True,
        num_reasoning_steps=2,
        reasoning_units=128,
        num_attention_heads=4,
        attention_key_dim=64,
        feature_names=feature_names
    )
    
    # Compiler le modu00e8le
    model = compile_reasoning_model(model, learning_rate=0.001)
    
    return model

def generate_explanations(model, X, df, feature_names):
    """
    Gu00e9nu00e8re des explications pour les pru00e9dictions du modu00e8le.
    
    Args:
        model: Modu00e8le entrau00eenu00e9
        X: Donnu00e9es d'entru00e9e
        df: DataFrame original
        feature_names: Noms des features
    
    Returns:
        DataFrame avec les pru00e9dictions et les explications
    """
    # Faire des pru00e9dictions
    logger.info("Gu00e9nu00e9ration des pru00e9dictions")
    predictions = model.predict(X)
    
    # Extraire les pru00e9dictions
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
    market_regime_names = ['sideways', 'bullish']
    decoder = ExplanationDecoder(feature_names=feature_names, market_regime_names=market_regime_names)
    
    # Gu00e9nu00e9rer des explications pour chaque exemple
    explanations = []
    for i in range(len(df)):
        example_dict = {
            'timestamp': df['timestamp'].iloc[i],
            'symbol': df['symbol'].iloc[i],
            'close': df['close'].iloc[i],
            'market_regime_pred': market_regime_classes[i],
            'market_regime_confidence': np.max(market_regime_pred[i]),
            'sl_pred': sl_tp_pred[i][0],
            'tp_pred': sl_tp_pred[i][1],
        }
        
        # Ajouter des explications si disponibles
        if reasoning_output is not None and attention_scores is not None:
            # Extraire les scores d'attention pour cet exemple
            example_attention = attention_scores[i] if len(attention_scores.shape) > 2 else attention_scores
            
            # Gu00e9nu00e9rer des explications
            try:
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
                logger.warning(f"Erreur lors de la gu00e9nu00e9ration des explications pour l'exemple {i}: {e}")
                example_dict['market_regime_explanation'] = f"Le modu00e8le pru00e9dit un marchu00e9 {market_regime_names[market_regime_classes[i]]}"
                example_dict['sl_explanation'] = f"Stop loss placu00e9 u00e0 {sl_tp_pred[i][0]:.4f}"
                example_dict['tp_explanation'] = f"Take profit placu00e9 u00e0 {sl_tp_pred[i][1]:.4f}"
        
        explanations.append(example_dict)
    
    return pd.DataFrame(explanations)

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Cru00e9er le ru00e9pertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pru00e9traiter les donnu00e9es
    X, df_norm, feature_names = preprocess_data(args.data_path, args.scalers_path)
    
    # Cru00e9er le modu00e8le
    model = create_model(X, feature_names)
    
    # Gu00e9nu00e9rer des explications
    logger.info("Gu00e9nu00e9ration des explications")
    explanations_df = generate_explanations(model, X, df_norm, feature_names)
    
    # Sauvegarder les explications
    explanations_path = os.path.join(args.output_dir, 'trading_explanations.csv')
    explanations_df.to_csv(explanations_path, index=False)
    logger.info(f"Explications sauvegardu00e9es dans {explanations_path}")
    
    # Afficher quelques exemples d'explications
    logger.info("\nExemples d'explications:")
    for i, (_, row) in enumerate(explanations_df.head(3).iterrows()):
        logger.info(f"\nExemple {i+1}:")
        logger.info(f"Timestamp: {row['timestamp']}")
        logger.info(f"Symbole: {row['symbol']}")
        logger.info(f"Prix de clu00f4ture: {row['close']}")
        logger.info(f"Pru00e9diction de ru00e9gime de marchu00e9: {row['market_regime_pred']} (confiance: {row['market_regime_confidence']:.2f})")
        logger.info(f"Pru00e9diction de SL/TP: SL={row['sl_pred']:.4f}, TP={row['tp_pred']:.4f}")
        logger.info(f"Explication du ru00e9gime de marchu00e9: {row['market_regime_explanation']}")
        logger.info(f"Explication du SL: {row['sl_explanation']}")
        logger.info(f"Explication du TP: {row['tp_explanation']}")

if __name__ == "__main__":
    main()
