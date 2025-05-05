#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour générer un dataset amélioré pour le modèle Morningstar avec capacité de raisonnement.
Ce script ajoute des features supplémentaires et enrichit le dataset existant.
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from datetime import datetime, timedelta
import ccxt

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_additional_data(start_date, end_date, symbols=['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT']):
    """
    Récupère des données supplémentaires depuis l'API de l'exchange.
    
    Args:
        start_date: Date de début
        end_date: Date de fin
        symbols: Liste des paires de trading
        
    Returns:
        DataFrame contenant les données supplémentaires
    """
    logger.info(f"Récupération des données supplémentaires pour {len(symbols)} paires de trading")
    
    # Utiliser l'API mock si en mode test
    if os.environ.get('USE_MOCK_EXCHANGE', 'false').lower() == 'true':
        logger.info("Utilisation de l'exchange mock")
        return generate_mock_data(start_date, end_date, symbols)
    
    # Initialiser l'exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    all_data = []
    
    for symbol in symbols:
        logger.info(f"Récupération des données pour {symbol}")
        
        # Convertir les dates en timestamps
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Récupérer les données OHLCV
        ohlcv = []
        current = since
        
        while current < until:
            try:
                batch = exchange.fetch_ohlcv(symbol, '1d', current, 1000)
                if len(batch) == 0:
                    break
                
                ohlcv.extend(batch)
                current = batch[-1][0] + 1
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
                break
        
        # Convertir en DataFrame
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Données récupérées: {len(combined_df)} lignes")
        return combined_df
    else:
        logger.warning("Aucune donnée récupérée")
        return pd.DataFrame()

def generate_mock_data(start_date, end_date, symbols):
    """
    Génère des données mock pour les tests.
    """
    logger.info("Génération de données mock")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    
    all_data = []
    
    for symbol in symbols:
        # Générer une série temporelle
        timestamps = [start + timedelta(days=i) for i in range(days)]
        
        # Générer des prix aléatoires avec une tendance
        base_price = 100 if 'BTC' in symbol else 10 if 'ETH' in symbol else 1
        trend = np.random.choice([-1, 1]) * 0.01  # Tendance à la hausse ou à la baisse
        
        prices = []
        current_price = base_price
        
        for _ in range(days):
            current_price *= (1 + trend + np.random.normal(0, 0.02))
            prices.append(current_price)
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': [abs(np.random.normal(1000, 500)) * base_price for _ in range(days)],
            'symbol': symbol
        })
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Données mock générées: {len(combined_df)} lignes")
    return combined_df

def add_sentiment_features(df, num_sentiment_features=10):
    """
    Ajoute des features de sentiment simulées.
    """
    logger.info("Ajout de features de sentiment")
    
    # Simuler des embeddings LLM de sentiment
    for i in range(num_sentiment_features):
        df[f'llm_sentiment_{i}'] = np.random.normal(0, 1, size=len(df))
    
    return df

def add_market_context_features(df):
    """
    Ajoute des features de contexte de marché.
    """
    logger.info("Ajout de features de contexte de marché")
    
    # Ajouter des features de volatilité
    df['volatility_1d'] = df.groupby('symbol')['close'].pct_change().abs()
    df['volatility_5d'] = df.groupby('symbol')['close'].pct_change(5).abs()
    
    # Ajouter des features de tendance
    df['trend_1d'] = df.groupby('symbol')['close'].pct_change()
    df['trend_5d'] = df.groupby('symbol')['close'].pct_change(5)
    df['trend_20d'] = df.groupby('symbol')['close'].pct_change(20)
    
    # Ajouter des features de volume
    df['volume_change_1d'] = df.groupby('symbol')['volume'].pct_change()
    df['volume_change_5d'] = df.groupby('symbol')['volume'].pct_change(5)
    
    # Ajouter des features de range
    df['daily_range'] = (df['high'] - df['low']) / df['low']
    df['weekly_range'] = df.groupby('symbol')['daily_range'].rolling(5).mean().reset_index(level=0, drop=True)
    
    return df

def add_technical_indicators(df):
    """
    Ajoute des indicateurs techniques supplémentaires.
    """
    logger.info("Ajout d'indicateurs techniques supplémentaires")
    
    # RSI
    delta = df.groupby('symbol')['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby('symbol').rolling(14).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby('symbol').rolling(14).mean().reset_index(level=0, drop=True)
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['sma_20'] = df.groupby('symbol')['close'].rolling(20).mean().reset_index(level=0, drop=True)
    df['std_20'] = df.groupby('symbol')['close'].rolling(20).std().reset_index(level=0, drop=True)
    df['bollinger_upper'] = df['sma_20'] + 2 * df['std_20']
    df['bollinger_lower'] = df['sma_20'] - 2 * df['std_20']
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_20']
    
    # MACD
    df['ema_12'] = df.groupby('symbol')['close'].ewm(span=12).mean().reset_index(level=0, drop=True)
    df['ema_26'] = df.groupby('symbol')['close'].ewm(span=26).mean().reset_index(level=0, drop=True)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df.groupby('symbol')['macd'].ewm(span=9).mean().reset_index(level=0, drop=True)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.groupby(df['symbol']).rolling(14).mean().reset_index(level=0, drop=True)
    
    return df

def add_cross_asset_features(df):
    """
    Ajoute des features de corrélation entre actifs.
    """
    logger.info("Ajout de features de corrélation entre actifs")
    
    # Pivoter pour avoir une colonne par symbole
    pivot_close = df.pivot(index='timestamp', columns='symbol', values='close')
    pivot_close.columns = [f"{col.replace('/', '_')}_close" for col in pivot_close.columns]
    
    # Calculer les rendements
    returns = pivot_close.pct_change()
    
    # Calculer les corrélations glissantes
    for col1 in returns.columns:
        for col2 in returns.columns:
            if col1 != col2:
                corr_col = f"corr_{col1}_{col2}"
                rolling_corr = returns[col1].rolling(30).corr(returns[col2])
                pivot_close[corr_col] = rolling_corr
    
    # Fusionner avec le DataFrame original
    df_with_timestamp = df.copy()
    df_with_timestamp = df_with_timestamp.merge(pivot_close, left_on='timestamp', right_index=True, how='left')
    
    return df_with_timestamp

def generate_enhanced_dataset(input_path, output_path, add_sentiment=True, add_context=True, add_technical=True, add_cross_asset=True):
    """
    Génère un dataset amélioré à partir du dataset existant.
    
    Args:
        input_path: Chemin vers le dataset existant
        output_path: Chemin de sortie pour le dataset amélioré
        add_sentiment: Ajouter des features de sentiment
        add_context: Ajouter des features de contexte de marché
        add_technical: Ajouter des indicateurs techniques supplémentaires
        add_cross_asset: Ajouter des features de corrélation entre actifs
    """
    logger.info(f"Génération d'un dataset amélioré à partir de {input_path}")
    
    # Charger le dataset existant
    df = pd.read_parquet(input_path)
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Extraire les dates min et max
    if 'timestamp' in df.columns:
        min_date = df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d')
    else:
        min_date = '2020-01-01'
        max_date = '2023-12-31'
    
    # Récupérer des données supplémentaires
    additional_data = fetch_additional_data(min_date, max_date)
    
    # Fusionner avec le dataset existant si des données ont été récupérées
    if not additional_data.empty and 'timestamp' in df.columns and 'symbol' in df.columns:
        df = df.merge(additional_data, on=['timestamp', 'symbol'], how='left', suffixes=('', '_new'))
        
        # Utiliser les nouvelles données si disponibles, sinon conserver les anciennes
        for col in additional_data.columns:
            if col not in ['timestamp', 'symbol'] and f"{col}_new" in df.columns:
                df[col] = df[f"{col}_new"].fillna(df[col])
                df.drop(f"{col}_new", axis=1, inplace=True)
    
    # Ajouter des features supplémentaires
    if add_sentiment:
        df = add_sentiment_features(df)
    
    if add_context:
        df = add_market_context_features(df)
    
    if add_technical:
        df = add_technical_indicators(df)
    
    if add_cross_asset:
        df = add_cross_asset_features(df)
    
    # Supprimer les lignes avec des valeurs manquantes
    na_count_before = df.isna().sum().sum()
    df.dropna(inplace=True)
    na_count_after = df.isna().sum().sum()
    logger.info(f"Valeurs manquantes supprimées: {na_count_before - na_count_after}")
    
    # Sauvegarder le dataset amélioré
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset amélioré sauvegardé dans {output_path} avec {len(df)} lignes et {len(df.columns)} colonnes")

def main():
    parser = argparse.ArgumentParser(description='Génère un dataset amélioré pour le modèle Morningstar.')
    parser.add_argument('--input', type=str, required=True, help='Chemin vers le dataset existant')
    parser.add_argument('--output', type=str, required=True, help='Chemin de sortie pour le dataset amélioré')
    parser.add_argument('--no-sentiment', action='store_true', help='Ne pas ajouter de features de sentiment')
    parser.add_argument('--no-context', action='store_true', help='Ne pas ajouter de features de contexte de marché')
    parser.add_argument('--no-technical', action='store_true', help='Ne pas ajouter d\'indicateurs techniques supplémentaires')
    parser.add_argument('--no-cross-asset', action='store_true', help='Ne pas ajouter de features de corrélation entre actifs')
    
    args = parser.parse_args()
    
    generate_enhanced_dataset(
        args.input,
        args.output,
        add_sentiment=not args.no_sentiment,
        add_context=not args.no_context,
        add_technical=not args.no_technical,
        add_cross_asset=not args.no_cross_asset
    )

if __name__ == "__main__":
    main()
