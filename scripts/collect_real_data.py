#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour collecter des donnu00e9es ru00e9elles pour le modu00e8le Morningstar.
Ce script utilise l'API CCXT pour les donnu00e9es de marchu00e9, l'API Gemini pour l'analyse de sentiment,
et CryptoBERT pour l'interpru00e9tation des donnu00e9es textuelles.
"""

import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import ccxt
import time

# Import des modules personnalisu00e9s
from data_collectors.market_data_collector import MarketDataCollector
from data_collectors.sentiment_analyzer import GeminiSentimentAnalyzer
from data_collectors.news_collector import CryptoNewsCollector
from data_processors.hmm_regime_detector import HMMRegimeDetector
from data_processors.cryptobert_processor import CryptoBERTProcessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataCollector:
    """
    Classe principale pour collecter des donnu00e9es ru00e9elles pour le modu00e8le Morningstar.
    """
    def __init__(self, 
                 output_dir, 
                 gemini_api_keys=None, 
                 start_date=None, 
                 end_date=None,
                 symbols=None,
                 timeframe='1d',
                 use_cryptobert=True,
                 use_hmm=True,
                 use_sentiment=True):
        """
        Initialise le collecteur de donnu00e9es ru00e9elles.
        
        Args:
            output_dir: Ru00e9pertoire de sortie pour les donnu00e9es collectu00e9es
            gemini_api_keys: Liste des clu00e9s API Gemini
            start_date: Date de du00e9but (format: 'YYYY-MM-DD')
            end_date: Date de fin (format: 'YYYY-MM-DD')
            symbols: Liste des paires de trading
            timeframe: Intervalle de temps ('1d', '1h', etc.)
            use_cryptobert: Utiliser CryptoBERT pour l'analyse de texte
            use_hmm: Utiliser HMM pour la du00e9tection de ru00e9gime
            use_sentiment: Utiliser l'analyse de sentiment Gemini
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paramu00e8tres par du00e9faut
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT']
        self.timeframe = timeframe
        
        # Fonctionnalitu00e9s activu00e9es
        self.use_cryptobert = use_cryptobert
        self.use_hmm = use_hmm
        self.use_sentiment = use_sentiment
        
        # Initialiser les collecteurs de donnu00e9es
        self.market_collector = MarketDataCollector()
        
        # Initialiser l'analyseur de sentiment si activu00e9
        if self.use_sentiment:
            if not gemini_api_keys:
                logger.warning("Aucune clu00e9 API Gemini fournie. L'analyse de sentiment sera limitu00e9e.")
                gemini_api_keys = [os.environ.get('GEMINI_API_KEY', '')]
            
            self.sentiment_analyzer = GeminiSentimentAnalyzer(api_keys=gemini_api_keys)
            self.news_collector = CryptoNewsCollector()
        
        # Initialiser les processeurs de donnu00e9es
        if self.use_hmm:
            self.hmm_detector = HMMRegimeDetector()
        
        if self.use_cryptobert:
            self.cryptobert = CryptoBERTProcessor()
    
    def collect_market_data(self):
        """
        Collecte les donnu00e9es de marchu00e9 pour toutes les paires de trading.
        
        Returns:
            DataFrame contenant les donnu00e9es de marchu00e9
        """
        logger.info(f"Collecte des donnu00e9es de marchu00e9 pour {len(self.symbols)} paires de trading")
        
        all_market_data = []
        
        for symbol in tqdm(self.symbols, desc="Collecte des donnu00e9es de marchu00e9"):
            try:
                # Collecter les donnu00e9es OHLCV
                market_data = self.market_collector.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    since=self.start_date,
                    until=self.end_date
                )
                
                # Ajouter les donnu00e9es de volume et de liquiditu00e9
                market_data = self.market_collector.add_volume_metrics(market_data, symbol)
                
                # Ajouter les indicateurs techniques
                market_data = self.market_collector.add_technical_indicators(market_data)
                
                all_market_data.append(market_data)
                
                # Sauvegarder les donnu00e9es intermu00e9diaires
                market_data_path = self.output_dir / f"market_data_{symbol.replace('/', '_')}.csv"
                market_data.to_csv(market_data_path, index=False)
                logger.info(f"Donnu00e9es de marchu00e9 pour {symbol} sauvegardu00e9es dans {market_data_path}")
                
                # Respecter les limites de taux de l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des donnu00e9es pour {symbol}: {e}")
        
        if all_market_data:
            combined_market_data = pd.concat(all_market_data, ignore_index=True)
            combined_market_data_path = self.output_dir / "combined_market_data.csv"
            combined_market_data.to_csv(combined_market_data_path, index=False)
            logger.info(f"Donnu00e9es de marchu00e9 combinu00e9es sauvegardu00e9es dans {combined_market_data_path}")
            return combined_market_data
        else:
            logger.warning("Aucune donnu00e9e de marchu00e9 collectu00e9e")
            return pd.DataFrame()
    
    def collect_news_and_sentiment(self, market_data):
        """
        Collecte les actualitu00e9s et analyse le sentiment pour chaque date et symbole.
        
        Args:
            market_data: DataFrame contenant les donnu00e9es de marchu00e9
            
        Returns:
            DataFrame avec les donnu00e9es de marchu00e9 enrichies de sentiment
        """
        if not self.use_sentiment:
            logger.info("Analyse de sentiment du00e9sactivu00e9e, ignoru00e9e")
            return market_data
        
        logger.info("Collecte des actualitu00e9s et analyse du sentiment")
        
        # Cru00e9er un DataFrame unique avec toutes les dates et symboles
        unique_dates = pd.to_datetime(market_data['timestamp'].unique())
        unique_symbols = market_data['symbol'].unique()
        
        news_data = []
        
        for date in tqdm(unique_dates, desc="Collecte des actualitu00e9s"):
            date_str = date.strftime('%Y-%m-%d')
            
            # Collecter les actualitu00e9s gu00e9nu00e9rales sur les crypto-monnaies pour cette date
            general_news = self.news_collector.fetch_crypto_news(date_str)
            
            for symbol in unique_symbols:
                # Extraire le nom de la crypto-monnaie du symbole (ex: BTC/USDT -> BTC)
                crypto_name = symbol.split('/')[0]
                
                # Collecter les actualitu00e9s spu00e9cifiques u00e0 cette crypto-monnaie
                crypto_news = self.news_collector.fetch_crypto_news(date_str, crypto_name)
                
                # Combiner les actualitu00e9s gu00e9nu00e9rales et spu00e9cifiques
                all_news = general_news + crypto_news
                
                if all_news:
                    # Analyser le sentiment avec Gemini
                    sentiment_results = self.sentiment_analyzer.analyze_sentiment(all_news, crypto_name)
                    
                    # Ajouter les ru00e9sultats au DataFrame
                    news_data.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'news_count': len(all_news),
                        'sentiment_score': sentiment_results.get('sentiment_score', 0),
                        'sentiment_magnitude': sentiment_results.get('sentiment_magnitude', 0),
                        'bullish_probability': sentiment_results.get('bullish_probability', 0.5),
                        'bearish_probability': sentiment_results.get('bearish_probability', 0.5),
                        'news_summary': sentiment_results.get('summary', ''),
                        'key_events': json.dumps(sentiment_results.get('key_events', []))
                    })
                
                # Respecter les limites de taux de l'API
                time.sleep(2)
        
        if news_data:
            news_df = pd.DataFrame(news_data)
            news_df_path = self.output_dir / "news_sentiment_data.csv"
            news_df.to_csv(news_df_path, index=False)
            logger.info(f"Donnu00e9es d'actualitu00e9s et de sentiment sauvegardu00e9es dans {news_df_path}")
            
            # Fusionner avec les donnu00e9es de marchu00e9
            market_data = market_data.merge(news_df, on=['timestamp', 'symbol'], how='left')
            
            # Remplir les valeurs manquantes
            for col in ['sentiment_score', 'sentiment_magnitude', 'bullish_probability', 'bearish_probability']:
                if col in market_data.columns:
                    market_data[col] = market_data[col].fillna(0)
            
            market_data['news_count'] = market_data['news_count'].fillna(0)
            market_data['news_summary'] = market_data['news_summary'].fillna('')
            market_data['key_events'] = market_data['key_events'].fillna('[]')
            
            return market_data
        else:
            logger.warning("Aucune donnu00e9e d'actualitu00e9s collectu00e9e")
            return market_data
    
    def process_with_cryptobert(self, market_data):
        """
        Traite les donnu00e9es textuelles avec CryptoBERT pour extraire des embeddings.
        
        Args:
            market_data: DataFrame contenant les donnu00e9es de marchu00e9 et de sentiment
            
        Returns:
            DataFrame enrichi avec les embeddings CryptoBERT
        """
        if not self.use_cryptobert:
            logger.info("Traitement CryptoBERT du00e9sactivu00e9, ignoru00e9")
            return market_data
        
        logger.info("Traitement des donnu00e9es textuelles avec CryptoBERT")
        
        # Vu00e9rifier si les colonnes de texte existent
        if 'news_summary' not in market_data.columns:
            logger.warning("Aucune donnu00e9e textuelle u00e0 traiter avec CryptoBERT")
            return market_data
        
        # Traiter les donnu00e9es textuelles par lots
        batch_size = 32
        num_batches = (len(market_data) + batch_size - 1) // batch_size
        
        all_embeddings = []
        
        for i in tqdm(range(num_batches), desc="Traitement CryptoBERT"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(market_data))
            batch = market_data.iloc[start_idx:end_idx]
            
            # Extraire les textes
            texts = batch['news_summary'].tolist()
            
            # Gu00e9nu00e9rer les embeddings
            embeddings = self.cryptobert.generate_embeddings(texts)
            all_embeddings.extend(embeddings)
        
        # Ajouter les embeddings au DataFrame
        embedding_cols = [f'cryptobert_embedding_{i}' for i in range(len(all_embeddings[0]) if all_embeddings else 0)]
        
        if all_embeddings and embedding_cols:
            embeddings_df = pd.DataFrame(all_embeddings, columns=embedding_cols)
            
            # Concatu00e9ner avec les donnu00e9es de marchu00e9
            market_data = pd.concat([market_data.reset_index(drop=True), embeddings_df], axis=1)
            
            # Sauvegarder les donnu00e9es intermu00e9diaires
            embeddings_path = self.output_dir / "cryptobert_embeddings.csv"
            embeddings_df.to_csv(embeddings_path, index=False)
            logger.info(f"Embeddings CryptoBERT sauvegardu00e9s dans {embeddings_path}")
        
        return market_data
    
    def detect_market_regimes(self, market_data):
        """
        Du00e9tecte les ru00e9gimes de marchu00e9 en utilisant HMM.
        
        Args:
            market_data: DataFrame contenant les donnu00e9es de marchu00e9
            
        Returns:
            DataFrame enrichi avec les ru00e9gimes de marchu00e9
        """
        if not self.use_hmm:
            logger.info("Du00e9tection de ru00e9gime HMM du00e9sactivu00e9e, ignoru00e9e")
            return market_data
        
        logger.info("Du00e9tection des ru00e9gimes de marchu00e9 avec HMM")
        
        # Traiter chaque symbole su00e9paru00e9ment
        unique_symbols = market_data['symbol'].unique()
        
        all_regime_data = []
        
        for symbol in tqdm(unique_symbols, desc="Du00e9tection des ru00e9gimes"):
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            # Trier par timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Du00e9tecter les ru00e9gimes
            returns = symbol_data['close'].pct_change().fillna(0).values.reshape(-1, 1)
            regimes, probs = self.hmm_detector.detect_regimes(returns)
            
            # Ajouter les ru00e9gimes et probabilitu00e9s au DataFrame
            symbol_data['hmm_regime'] = regimes
            
            for i in range(probs.shape[1]):
                symbol_data[f'hmm_prob_{i}'] = probs[:, i]
            
            all_regime_data.append(symbol_data)
        
        if all_regime_data:
            regime_data = pd.concat(all_regime_data, ignore_index=True)
            
            # Sauvegarder les donnu00e9es intermu00e9diaires
            regime_path = self.output_dir / "hmm_regime_data.csv"
            regime_data.to_csv(regime_path, index=False)
            logger.info(f"Donnu00e9es de ru00e9gime HMM sauvegardu00e9es dans {regime_path}")
            
            return regime_data
        else:
            logger.warning("Aucune donnu00e9e de ru00e9gime gu00e9nu00e9ru00e9e")
            return market_data
    
    def generate_labels(self, market_data):
        """
        Gu00e9nu00e8re les u00e9tiquettes pour l'entrau00eenement supervisu00e9.
        
        Args:
            market_data: DataFrame contenant les donnu00e9es de marchu00e9
            
        Returns:
            DataFrame avec les u00e9tiquettes ajoutu00e9es
        """
        logger.info("Gu00e9nu00e9ration des u00e9tiquettes pour l'entrau00eenement supervisu00e9")
        
        # Traiter chaque symbole su00e9paru00e9ment
        unique_symbols = market_data['symbol'].unique()
        
        all_labeled_data = []
        
        for symbol in tqdm(unique_symbols, desc="Gu00e9nu00e9ration des u00e9tiquettes"):
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            # Trier par timestamp
            symbol_data = symbol_data.sort_values('timestamp')
            
            # Gu00e9nu00e9rer les u00e9tiquettes de ru00e9gime de marchu00e9
            # 0: sideways, 1: bullish, 2: bearish, 3: volatile
            returns = symbol_data['close'].pct_change().fillna(0)
            volatility = returns.rolling(window=5).std().fillna(0)
            
            # Du00e9finir les seuils
            return_threshold = 0.02
            volatility_threshold = 0.03
            
            # Classifier les ru00e9gimes
            conditions = [
                (volatility <= volatility_threshold) & (abs(returns) <= return_threshold),  # sideways
                (returns > return_threshold),  # bullish
                (returns < -return_threshold),  # bearish
                (volatility > volatility_threshold)  # volatile
            ]
            choices = [0, 1, 2, 3]
            symbol_data['market_regime'] = np.select(conditions, choices, default=0)
            
            # Gu00e9nu00e9rer les niveaux de stop loss et take profit optimaux
            # Simplification: SL u00e0 -2% et TP u00e0 +3% du prix actuel
            symbol_data['level_sl'] = symbol_data['close'] * 0.98
            symbol_data['level_tp'] = symbol_data['close'] * 1.03
            
            all_labeled_data.append(symbol_data)
        
        if all_labeled_data:
            labeled_data = pd.concat(all_labeled_data, ignore_index=True)
            
            # Sauvegarder les donnu00e9es u00e9tiquetu00e9es
            labeled_path = self.output_dir / "labeled_data.csv"
            labeled_data.to_csv(labeled_path, index=False)
            logger.info(f"Donnu00e9es u00e9tiquetu00e9es sauvegardu00e9es dans {labeled_path}")
            
            return labeled_data
        else:
            logger.warning("Aucune donnu00e9e u00e9tiquetu00e9e gu00e9nu00e9ru00e9e")
            return market_data
    
    def split_dataset(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Divise le dataset en ensembles d'entrau00eenement, de validation et de test.
        
        Args:
            data: DataFrame contenant toutes les donnu00e9es
            train_ratio: Proportion des donnu00e9es d'entrau00eenement
            val_ratio: Proportion des donnu00e9es de validation
            test_ratio: Proportion des donnu00e9es de test
            
        Returns:
            DataFrame avec une colonne 'split' indiquant l'ensemble
        """
        logger.info("Division du dataset en ensembles d'entrau00eenement, de validation et de test")
        
        # Trier par timestamp
        data = data.sort_values('timestamp')
        
        # Ajouter une colonne pour l'ensemble
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        data['split'] = 'train'
        data.iloc[train_end:val_end, data.columns.get_loc('split')] = 'val'
        data.iloc[val_end:, data.columns.get_loc('split')] = 'test'
        
        # Sauvegarder le dataset final
        final_path = self.output_dir / "final_dataset.parquet"
        data.to_parquet(final_path, index=False)
        logger.info(f"Dataset final sauvegardu00e9 dans {final_path}")
        
        # Afficher les statistiques
        split_counts = data['split'].value_counts()
        logger.info(f"Statistiques de division: {split_counts.to_dict()}")
        
        return data
    
    def collect_and_process_data(self):
        """
        Exu00e9cute le processus complet de collecte et de traitement des donnu00e9es.
        
        Returns:
            DataFrame contenant le dataset final
        """
        logger.info("Du00e9but du processus de collecte et de traitement des donnu00e9es")
        
        # 1. Collecter les donnu00e9es de marchu00e9
        market_data = self.collect_market_data()
        if market_data.empty:
            logger.error("Aucune donnu00e9e de marchu00e9 collectu00e9e, arru00eat du processus")
            return pd.DataFrame()
        
        # 2. Collecter les actualitu00e9s et analyser le sentiment
        market_data = self.collect_news_and_sentiment(market_data)
        
        # 3. Traiter les donnu00e9es textuelles avec CryptoBERT
        market_data = self.process_with_cryptobert(market_data)
        
        # 4. Du00e9tecter les ru00e9gimes de marchu00e9
        market_data = self.detect_market_regimes(market_data)
        
        # 5. Gu00e9nu00e9rer les u00e9tiquettes
        labeled_data = self.generate_labels(market_data)
        
        # 6. Diviser le dataset
        final_dataset = self.split_dataset(labeled_data)
        
        logger.info("Processus de collecte et de traitement des donnu00e9es terminu00e9")
        return final_dataset

def main():
    parser = argparse.ArgumentParser(description='Collecte des donnu00e9es ru00e9elles pour le modu00e8le Morningstar.')
    parser.add_argument('--output-dir', type=str, default='data/real', help='Ru00e9pertoire de sortie pour les donnu00e9es collectu00e9es')
    parser.add_argument('--start-date', type=str, help='Date de du00e9but (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='Date de fin (format: YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Liste des paires de trading')
    parser.add_argument('--timeframe', type=str, default='1d', help='Intervalle de temps (1d, 1h, etc.)')
    parser.add_argument('--gemini-api-keys', type=str, nargs='+', help='Liste des clu00e9s API Gemini')
    parser.add_argument('--no-cryptobert', action='store_true', help='Du00e9sactiver CryptoBERT')
    parser.add_argument('--no-hmm', action='store_true', help='Du00e9sactiver HMM')
    parser.add_argument('--no-sentiment', action='store_true', help='Du00e9sactiver l\'analyse de sentiment')
    
    args = parser.parse_args()
    
    # Cru00e9er le collecteur de donnu00e9es
    collector = RealDataCollector(
        output_dir=args.output_dir,
        gemini_api_keys=args.gemini_api_keys,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=args.symbols,
        timeframe=args.timeframe,
        use_cryptobert=not args.no_cryptobert,
        use_hmm=not args.no_hmm,
        use_sentiment=not args.no_sentiment
    )
    
    # Collecter et traiter les donnu00e9es
    final_dataset = collector.collect_and_process_data()
    
    if not final_dataset.empty:
        logger.info(f"Dataset final gu00e9nu00e9ru00e9 avec succu00e8s: {len(final_dataset)} lignes, {len(final_dataset.columns)} colonnes")
    else:
        logger.error("u00c9chec de la gu00e9nu00e9ration du dataset final")

if __name__ == "__main__":
    main()
