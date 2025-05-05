#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script optimisé pour collecter des données enrichies avec des informations de marché externes.
Utilise le multithreading et la mise en cache pour accélérer le processus de collecte.
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Importer nos modules personnalisés
from data_collectors.market_data_collector import MarketDataCollector
from data_collectors.news_collector import CryptoNewsCollector
from data_collectors.sentiment_analyzer import GeminiSentimentAnalyzer
from data_collectors.market_info_collector import initialize_market_info_collector
from data_processors.cryptobert_processor import CryptoBERTProcessor
from data_processors.hmm_regime_detector import HMMRegimeDetector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer un répertoire de cache si nécessaire
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    """
    Parse les arguments de la ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description="Collecte optimisée de données enrichies")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,ADA/USDT,AVAX/USDT,DOT/USDT,MATIC/USDT,LINK/USDT,DOGE/USDT,UNI/USDT,ATOM/USDT,LTC/USDT,BCH/USDT",
        help="Liste des symboles de crypto-monnaies séparés par des virgules"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        help="Timeframe pour les données OHLCV (1m, 5m, 15m, 1h, 4h, 1d)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Date de début au format YYYY-MM-DD"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-01-31",
        help="Date de fin au format YYYY-MM-DD"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/enriched",
        help="Répertoire de sortie pour les données enrichies"
    )
    parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="Inclure l'analyse de sentiment"
    )
    parser.add_argument(
        "--use-news",
        action="store_true",
        help="Inclure les actualités crypto"
    )
    parser.add_argument(
        "--use-market-info",
        action="store_true",
        help="Inclure les informations de marché de CoinMarketCap"
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        help="Inclure la détection de régime HMM"
    )
    parser.add_argument(
        "--use-cryptobert",
        action="store_true",
        help="Inclure les embeddings CryptoBERT"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Nombre maximum de workers pour le multithreading"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Utiliser le cache pour accélérer la collecte"
    )
    return parser.parse_args()

# Décorateur pour la mise en cache des résultats
def cached_result(cache_file):
    def decorator(func):
        def wrapper(*args, **kwargs):
            use_cache = kwargs.get('use_cache', True)
            cache_path = CACHE_DIR / cache_file
            
            # Si le cache existe et que l'utilisation du cache est activée, charger depuis le cache
            if use_cache and cache_path.exists():
                try:
                    logger.info(f"Chargement des données depuis le cache: {cache_path}")
                    if cache_file.endswith('.parquet'):
                        return pd.read_parquet(cache_path)
                    elif cache_file.endswith('.json'):
                        with open(cache_path, 'r') as f:
                            return json.load(f)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du cache {cache_path}: {e}")
            
            # Exécuter la fonction
            result = func(*args, **kwargs)
            
            # Sauvegarder le résultat dans le cache
            if use_cache and result is not None:
                try:
                    logger.info(f"Sauvegarde des données dans le cache: {cache_path}")
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if cache_file.endswith('.parquet'):
                        if isinstance(result, pd.DataFrame):
                            result.to_parquet(cache_path, index=False)
                    elif cache_file.endswith('.json'):
                        with open(cache_path, 'w') as f:
                            json.dump(result, f)
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde du cache {cache_path}: {e}")
            
            return result
        return wrapper
    return decorator

@cached_result('market_data.parquet')
def collect_market_data(symbols, timeframe, start_date, end_date, use_cache=True, max_workers=4):
    """
    Collecte les données de marché de base en parallèle.
    
    Args:
        symbols: Liste des symboles
        timeframe: Timeframe pour les données OHLCV
        start_date: Date de début
        end_date: Date de fin
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        DataFrame avec les données de marché
    """
    # Convertir la liste de symboles en liste si elle est fournie comme une chaîne
    if isinstance(symbols, str):
        symbols_list = symbols.split(',')
    else:
        symbols_list = symbols
    
    logger.info(f"Collecte des données de marché pour {len(symbols_list)} symboles")
    
    # Initialiser le collecteur de données de marché
    market_collector = MarketDataCollector()
    
    # Fonction pour collecter les données pour un symbole
    def collect_for_symbol(symbol):
        logger.info(f"Collecte des données pour {symbol}")
        try:
            # Utiliser la méthode fetch_ohlcv disponible
            symbol_data = market_collector.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=start_date,
                until=end_date
            )
            
            if symbol_data is not None and len(symbol_data) > 0:
                symbol_data['symbol'] = symbol
                return symbol_data
            else:
                logger.warning(f"Aucune donnée collectée pour {symbol}")
                return None
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des données pour {symbol}: {e}")
            return None
    
    # Collecter les données en parallèle
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(collect_for_symbol, symbol): symbol for symbol in symbols_list}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol_data = future.result()
                if symbol_data is not None:
                    all_data.append(symbol_data)
            except Exception as e:
                logger.error(f"Exception lors de la collecte pour {symbol}: {e}")
    
    # Concaténer tous les DataFrames
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Données collectées: {len(df)} lignes pour {len(all_data)} symboles")
        
        # Ajouter les indicateurs techniques
        df = market_collector.add_technical_indicators(df)
        
        logger.info(f"Données de marché collectées avec succès: {len(df)} lignes")
        return df
    else:
        logger.error("Aucune donnée collectée")
        return pd.DataFrame()

@cached_result('news_data.json')
def collect_news_data(symbols, start_date, end_date, use_cache=True, max_workers=4):
    """
    Collecte les actualités crypto en parallèle.
    
    Args:
        symbols: Liste des symboles
        start_date: Date de début
        end_date: Date de fin
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        Dictionnaire avec les actualités par symbole et par date
    """
    # Convertir la liste de symboles en liste si elle est fournie comme une chaîne
    if isinstance(symbols, str):
        symbols_list = symbols.split(',')
    else:
        symbols_list = symbols
    
    logger.info(f"Collecte des actualités pour {len(symbols_list)} symboles")
    
    # Initialiser le collecteur d'actualités
    news_collector = CryptoNewsCollector()
    
    # Fonction pour collecter les actualités pour un symbole
    def collect_news_for_symbol(symbol):
        logger.info(f"Collecte des actualités pour {symbol}")
        try:
            # Extraire le symbole de base (sans /USDT)
            base_symbol = symbol.split('/')[0]
            
            # Convertir les dates en objets datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Collecter les actualités pour chaque jour dans la plage de dates
            all_news = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime('%Y-%m-%d')
                try:
                    # Utiliser la méthode fetch_crypto_news avec les bons paramètres
                    news = news_collector.fetch_crypto_news(date_str, base_symbol)
                    for item in news:
                        if 'date' not in item:
                            item['date'] = date_str
                    all_news.extend(news)
                except Exception as e:
                    logger.warning(f"Erreur lors de la collecte des actualités pour {base_symbol} le {date_str}: {e}")
                
                # Passer au jour suivant
                current_dt += timedelta(days=1)
            
            if all_news:
                return {base_symbol: all_news}
            else:
                logger.warning(f"Aucune actualité collectée pour {base_symbol}")
                return {base_symbol: []}
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des actualités pour {symbol}: {e}")
            return {symbol.split('/')[0]: []}
    
    # Collecter les actualités en parallèle
    news_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(collect_news_for_symbol, symbol): symbol for symbol in symbols_list}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                news_data.update(result)
            except Exception as e:
                logger.error(f"Exception lors de la collecte des actualités pour {symbol}: {e}")
    
    logger.info(f"Actualités collectées pour {len(news_data)} symboles")
    return news_data

@cached_result('sentiment_data.json')
def analyze_sentiment(news_data, use_cache=True, max_workers=4):
    """
    Analyse le sentiment des actualités en parallèle.
    
    Args:
        news_data: Dictionnaire avec les actualités par symbole
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        Dictionnaire avec les sentiments par symbole et par date
    """
    logger.info("Analyse du sentiment des actualités")
    
    # Initialiser l'analyseur de sentiment
    sentiment_analyzer = GeminiSentimentAnalyzer()
    
    # Fonction pour analyser le sentiment pour un symbole
    def analyze_for_symbol(symbol_news_tuple):
        symbol, news_list = symbol_news_tuple
        logger.info(f"Analyse du sentiment pour {symbol} ({len(news_list)} articles)")
        try:
            if not news_list:
                return symbol, {}
            
            # Analyser le sentiment
            sentiment_results = sentiment_analyzer.analyze_sentiment(news_list, symbol)
            
            # Organiser les résultats par date
            sentiment_by_date = {}
            for article, sentiment in zip(news_list, sentiment_results):
                date = article.get('date')
                if date:
                    # Convertir la date en chaîne si nécessaire
                    if isinstance(date, datetime):
                        date = date.strftime('%Y-%m-%d')
                    
                    if date not in sentiment_by_date:
                        sentiment_by_date[date] = []
                    
                    sentiment_by_date[date].append(sentiment)
            
            # Calculer la moyenne du sentiment pour chaque date
            for date in sentiment_by_date:
                sentiments = sentiment_by_date[date]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_by_date[date] = avg_sentiment
            
            return symbol, sentiment_by_date
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du sentiment pour {symbol}: {e}")
            return symbol, {}
    
    # Analyser le sentiment en parallèle
    sentiment_data = {}
    symbol_news_items = [(symbol, news) for symbol, news in news_data.items()]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(analyze_for_symbol, item): item[0] for item in symbol_news_items}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result_symbol, result_sentiment = future.result()
                sentiment_data[result_symbol] = result_sentiment
            except Exception as e:
                logger.error(f"Exception lors de l'analyse du sentiment pour {symbol}: {e}")
    
    logger.info(f"Sentiment analysé pour {len(sentiment_data)} symboles")
    return sentiment_data

@cached_result('cryptobert_data.json')
def generate_cryptobert_embeddings(news_data, use_cache=True, max_workers=4):
    """
    Génère des embeddings CryptoBERT pour les actualités en parallèle.
    
    Args:
        news_data: Dictionnaire avec les actualités par symbole
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        Dictionnaire avec les embeddings par symbole et par date
    """
    logger.info("Génération des embeddings CryptoBERT")
    
    # Initialiser le processeur CryptoBERT
    cryptobert_processor = CryptoBERTProcessor()
    
    # Fonction pour générer des embeddings pour un symbole
    def generate_for_symbol(symbol_news_tuple):
        symbol, news_list = symbol_news_tuple
        logger.info(f"Génération des embeddings pour {symbol} ({len(news_list)} articles)")
        try:
            if not news_list:
                return symbol, {}
            
            # Extraire les contenus des articles
            texts = [article.get('content', '') for article in news_list]
            
            # Générer les embeddings
            embeddings = cryptobert_processor.generate_embeddings(texts)
            
            # Organiser les résultats par date
            embeddings_by_date = {}
            for article, embedding in zip(news_list, embeddings):
                date = article.get('date')
                if date and embedding is not None:
                    # Convertir la date en chaîne si nécessaire
                    if isinstance(date, datetime):
                        date = date.strftime('%Y-%m-%d')
                    
                    if date not in embeddings_by_date:
                        embeddings_by_date[date] = []
                    
                    embeddings_by_date[date].append(embedding)
            
            # Calculer la moyenne des embeddings pour chaque date
            for date in embeddings_by_date:
                date_embeddings = embeddings_by_date[date]
                if date_embeddings:
                    # Convertir en array numpy pour le calcul de la moyenne
                    date_embeddings_array = np.array(date_embeddings)
                    avg_embedding = date_embeddings_array.mean(axis=0).tolist()
                    embeddings_by_date[date] = avg_embedding
            
            return symbol, embeddings_by_date
        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings pour {symbol}: {e}")
            return symbol, {}
    
    # Générer les embeddings en parallèle
    embeddings_data = {}
    symbol_news_items = [(symbol, news) for symbol, news in news_data.items()]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(generate_for_symbol, item): item[0] for item in symbol_news_items}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result_symbol, result_embeddings = future.result()
                embeddings_data[result_symbol] = result_embeddings
            except Exception as e:
                logger.error(f"Exception lors de la génération des embeddings pour {symbol}: {e}")
    
    logger.info(f"Embeddings générés pour {len(embeddings_data)} symboles")
    return embeddings_data

@cached_result('hmm_data.json')
def detect_hmm_regimes(market_data, use_cache=True, max_workers=4):
    """
    Détecte les régimes de marché avec HMM en parallèle.
    
    Args:
        market_data: DataFrame avec les données de marché
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        Dictionnaire avec les régimes HMM par symbole et par date
    """
    logger.info("Détection des régimes de marché avec HMM")
    
    # Initialiser le détecteur de régime HMM
    hmm_detector = HMMRegimeDetector()
    
    # Fonction pour détecter les régimes pour un symbole
    def detect_for_symbol(symbol):
        logger.info(f"Détection des régimes HMM pour {symbol}")
        try:
            # Filtrer les données pour ce symbole
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 30:  # Vérifier qu'il y a suffisamment de données
                logger.warning(f"Pas assez de données pour {symbol} pour la détection HMM")
                return symbol, {}
            
            # Extraire les rendements
            symbol_data['returns'] = symbol_data['close'].pct_change()
            symbol_data = symbol_data.dropna(subset=['returns'])
            returns = symbol_data['returns'].values
            
            # Détecter les régimes
            hmm_results = hmm_detector.detect_regimes(returns)
            
            # Organiser les résultats par date
            regimes_by_date = {}
            for date, regime in zip(symbol_data['date'], hmm_results):
                # Convertir la date en chaîne si nécessaire
                if isinstance(date, pd.Timestamp) or isinstance(date, datetime):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                
                regimes_by_date[date_str] = int(regime)
            
            return symbol, regimes_by_date
        except Exception as e:
            logger.error(f"Erreur lors de la détection des régimes HMM pour {symbol}: {e}")
            return symbol, {}
    
    # Détecter les régimes en parallèle
    hmm_data = {}
    symbols = market_data['symbol'].unique()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(detect_for_symbol, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result_symbol, result_regimes = future.result()
                hmm_data[result_symbol] = result_regimes
            except Exception as e:
                logger.error(f"Exception lors de la détection des régimes HMM pour {symbol}: {e}")
    
    logger.info(f"Régimes HMM détectés pour {len(hmm_data)} symboles")
    return hmm_data

@cached_result('market_info_data.json')
def collect_market_info(symbols, use_cache=True, max_workers=4):
    """
    Collecte les informations de marché en parallèle.
    
    Args:
        symbols: Liste des symboles
        use_cache: Utiliser le cache
        max_workers: Nombre maximum de workers
    
    Returns:
        Dictionnaire avec les informations de marché par symbole
    """
    logger.info("Collecte des informations de marché")
    
    # Initialiser le collecteur d'informations de marché
    market_info_collector = initialize_market_info_collector()
    
    # Fonction pour collecter les informations pour un symbole
    def collect_info_for_symbol(symbol):
        logger.info(f"Collecte des informations de marché pour {symbol}")
        try:
            # Extraire le symbole de base (sans /USDT)
            base_symbol = symbol.split('/')[0]
            
            # Collecter les métadonnées
            metadata = market_info_collector.get_crypto_metadata(base_symbol)
            
            # Collecter les métriques de marché
            metrics = market_info_collector.get_market_metrics(base_symbol)
            
            return base_symbol, {
                'metadata': metadata,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des informations de marché pour {symbol}: {e}")
            return symbol.split('/')[0], {}
    
    # Convertir la liste de symboles en liste si elle est fournie comme une chaîne
    if isinstance(symbols, str):
        symbols_list = symbols.split(',')
    else:
        symbols_list = symbols
    
    # Collecter les informations en parallèle
    market_info_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(collect_info_for_symbol, symbol): symbol for symbol in symbols_list}
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result_symbol, result_info = future.result()
                market_info_data[result_symbol] = result_info
            except Exception as e:
                logger.error(f"Exception lors de la collecte des informations de marché pour {symbol}: {e}")
    
    # Collecter les données globales du marché
    try:
        global_data = market_info_collector.get_global_market_data()
        if global_data:
            market_info_data['global'] = global_data
    except Exception as e:
        logger.error(f"Erreur lors de la collecte des données globales du marché: {e}")
    
    logger.info(f"Informations de marché collectées pour {len(market_info_data)} symboles")
    return market_info_data

def enrich_dataframe(market_data, news_data, sentiment_data, cryptobert_data, hmm_data, market_info_data):
    """
    Enrichit le DataFrame de marché avec toutes les données collectées.
    
    Args:
        market_data: DataFrame avec les données de marché
        news_data: Dictionnaire avec les actualités
        sentiment_data: Dictionnaire avec les sentiments
        cryptobert_data: Dictionnaire avec les embeddings CryptoBERT
        hmm_data: Dictionnaire avec les régimes HMM
        market_info_data: Dictionnaire avec les informations de marché
    
    Returns:
        DataFrame enrichi
    """
    logger.info("Enrichissement du DataFrame avec toutes les données collectées")
    
    # Créer une copie du DataFrame
    df = market_data.copy()
    
    # Vérifier que la colonne 'date' existe, sinon la créer
    if 'date' not in df.columns:
        # Essayer de trouver une colonne qui pourrait contenir des dates
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            # Utiliser la première colonne de date trouvée
            df['date'] = df[date_columns[0]]
            logger.info(f"Colonne 'date' créée à partir de la colonne {date_columns[0]}")
        else:
            # Si aucune colonne de date n'est trouvée, utiliser l'index comme date
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
            logger.warning("Aucune colonne de date trouvée, utilisation d'une séquence de dates arbitraire")
    
    # Convertir la colonne 'date' en format string YYYY-MM-DD si ce n'est pas déjà le cas
    df['date_str'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) else str(x))
    
    # Ajouter les sentiments
    logger.info("Ajout des sentiments")
    df['sentiment_score'] = np.nan
    for idx, row in df.iterrows():
        symbol = row['symbol'].split('/')[0] if 'symbol' in df.columns else None
        date_str = row['date_str']
        
        if symbol and symbol in sentiment_data and date_str in sentiment_data[symbol]:
            df.at[idx, 'sentiment_score'] = sentiment_data[symbol][date_str]
    
    # Ajouter les embeddings CryptoBERT
    logger.info("Ajout des embeddings CryptoBERT")
    embedding_dim = 10  # Dimension des embeddings CryptoBERT (réduite pour éviter de trop augmenter la taille du dataset)
    for i in range(embedding_dim):
        df[f'cryptobert_dim_{i}'] = np.nan
    
    for idx, row in df.iterrows():
        symbol = row['symbol'].split('/')[0] if 'symbol' in df.columns else None
        date_str = row['date_str']
        
        if symbol and symbol in cryptobert_data and date_str in cryptobert_data[symbol]:
            embedding = cryptobert_data[symbol][date_str]
            for i, value in enumerate(embedding):
                if i < embedding_dim:  # Éviter les erreurs d'index
                    df.at[idx, f'cryptobert_dim_{i}'] = value
    
    # Ajouter les régimes HMM
    logger.info("Ajout des régimes HMM")
    df['hmm_regime'] = np.nan
    for idx, row in df.iterrows():
        symbol = row['symbol'] if 'symbol' in df.columns else None
        date_str = row['date_str']
        
        if symbol and symbol in hmm_data and date_str in hmm_data[symbol]:
            df.at[idx, 'hmm_regime'] = hmm_data[symbol][date_str]
    
    # Ajouter les métriques de marché
    logger.info("Ajout des métriques de marché")
    market_metrics = [
        'market_cap', 'volume_24h', 'percent_change_1h', 'percent_change_24h',
        'percent_change_7d', 'circulating_supply', 'total_supply', 'max_supply',
        'num_market_pairs', 'cmc_rank'
    ]
    
    for metric in market_metrics:
        df[f'market_{metric}'] = np.nan
    
    for idx, row in df.iterrows():
        symbol = row['symbol'].split('/')[0] if 'symbol' in df.columns else None
        
        if symbol and symbol in market_info_data and 'metrics' in market_info_data[symbol]:
            metrics = market_info_data[symbol]['metrics']
            for metric in market_metrics:
                if metric in metrics:
                    df.at[idx, f'market_{metric}'] = metrics[metric]
    
    # Ajouter les données globales du marché
    logger.info("Ajout des données globales du marché")
    global_metrics = [
        'total_market_cap', 'total_volume_24h', 'btc_dominance',
        'eth_dominance', 'active_cryptocurrencies', 'total_cryptocurrencies'
    ]
    
    for metric in global_metrics:
        df[f'global_{metric}'] = np.nan
    
    if 'global' in market_info_data:
        global_data = market_info_data['global']
        for idx, row in df.iterrows():
            for metric in global_metrics:
                if metric in global_data:
                    df.at[idx, f'global_{metric}'] = global_data[metric]
    
    # Remplacer les valeurs manquantes par des zéros ou des moyennes selon le cas
    logger.info("Remplacement des valeurs manquantes")
    
    # Pour les colonnes numériques, remplacer par la moyenne
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['open', 'high', 'low', 'close', 'volume']:  # Garder les valeurs OHLCV intactes
            df[col] = df[col].fillna(df[col].mean() if not df[col].isna().all() else 0)
    
    # Supprimer la colonne date_str temporaire
    df.drop('date_str', axis=1, inplace=True)
    
    logger.info(f"DataFrame enrichi avec succès: {len(df)} lignes, {len(df.columns)} colonnes")
    return df

def main():
    """
    Fonction principale qui orchestre la collecte et l'enrichissement des données.
    """
    # Parser les arguments
    args = parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collecter les données de marché de base
    market_data = collect_market_data(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        use_cache=args.use_cache,
        max_workers=args.max_workers
    )
    
    if market_data.empty:
        logger.error("Aucune donnée de marché collectée. Arrêt du script.")
        sys.exit(1)
    
    # Initialiser les dictionnaires pour les données enrichies
    news_data = {}
    sentiment_data = {}
    cryptobert_data = {}
    hmm_data = {}
    market_info_data = {}
    
    # Collecter les actualités si demandé
    if args.use_news or args.use_sentiment or args.use_cryptobert:
        news_data = collect_news_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=args.use_cache,
            max_workers=args.max_workers
        )
    
    # Analyser le sentiment si demandé
    if args.use_sentiment and news_data:
        sentiment_data = analyze_sentiment(
            news_data=news_data,
            use_cache=args.use_cache,
            max_workers=args.max_workers
        )
    
    # Générer les embeddings CryptoBERT si demandé
    if args.use_cryptobert and news_data:
        cryptobert_data = generate_cryptobert_embeddings(
            news_data=news_data,
            use_cache=args.use_cache,
            max_workers=args.max_workers
        )
    
    # Détecter les régimes HMM si demandé
    if args.use_hmm and not market_data.empty:
        hmm_data = detect_hmm_regimes(
            market_data=market_data,
            use_cache=args.use_cache,
            max_workers=args.max_workers
        )
    
    # Collecter les informations de marché si demandé
    if args.use_market_info:
        market_info_data = collect_market_info(
            symbols=args.symbols,
            use_cache=args.use_cache,
            max_workers=args.max_workers
        )
    
    # Enrichir le DataFrame avec toutes les données collectées
    enriched_df = enrich_dataframe(
        market_data=market_data,
        news_data=news_data,
        sentiment_data=sentiment_data,
        cryptobert_data=cryptobert_data,
        hmm_data=hmm_data,
        market_info_data=market_info_data
    )
    
    # Sauvegarder le DataFrame enrichi
    output_file = output_dir / "enriched_dataset.parquet"
    enriched_df.to_parquet(output_file, index=False)
    logger.info(f"Dataset enrichi sauvegardé dans {output_file}")
    
    # Afficher les statistiques du dataset
    logger.info(f"Statistiques du dataset enrichi:")
    logger.info(f"  - Nombre de lignes: {len(enriched_df)}")
    logger.info(f"  - Nombre de colonnes: {len(enriched_df.columns)}")
    logger.info(f"  - Symboles: {enriched_df['symbol'].nunique()}")
    logger.info(f"  - Période: {enriched_df['date'].min()} à {enriched_df['date'].max()}")
    
    # Sauvegarder les métadonnées du dataset
    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": args.symbols,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": len(enriched_df),
        "columns": len(enriched_df.columns),
        "features": list(enriched_df.columns),
        "use_sentiment": args.use_sentiment,
        "use_news": args.use_news,
        "use_market_info": args.use_market_info,
        "use_hmm": args.use_hmm,
        "use_cryptobert": args.use_cryptobert
    }
    
    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Métadonnées du dataset sauvegardées dans {metadata_file}")
    logger.info("Collecte et enrichissement des données terminés avec succès!")

if __name__ == "__main__":
    main()
