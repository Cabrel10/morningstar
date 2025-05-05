#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour collecter des informations de marché à partir de sources gratuites.
Utilise CoinMarketCap API et Google APIs pour enrichir les données.
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

class MarketInfoCollector:
    """
    Collecteur d'informations de marché à partir de sources gratuites.
    """
    
    def __init__(self, coinmarketcap_api_keys=None, google_api_keys=None):
        """
        Initialise le collecteur d'informations de marché.
        
        Args:
            coinmarketcap_api_keys: Liste de clés API CoinMarketCap (optionnel)
            google_api_keys: Liste de clés API Google (optionnel)
        """
        # Si aucune clé n'est fournie, essayer de récupérer les clés depuis les variables d'environnement
        if not coinmarketcap_api_keys:
            coinmarketcap_api_keys = []
            # Chercher toutes les clés COINMARKETCAP_API_KEY_* dans l'environnement
            for key, value in os.environ.items():
                if key.startswith('COINMARKETCAP_API_KEY'):
                    if value and value not in coinmarketcap_api_keys:
                        coinmarketcap_api_keys.append(value)
        
        if not google_api_keys:
            google_api_keys = []
            # Chercher toutes les clés GOOGLE_API_KEY_* dans l'environnement
            for key, value in os.environ.items():
                if key.startswith('GOOGLE_API_KEY_'):
                    if value and value not in google_api_keys:
                        google_api_keys.append(value)
        
        self.coinmarketcap_api_keys = coinmarketcap_api_keys or []
        self.google_api_keys = google_api_keys or []
        
        # Index pour la rotation des clés
        self.cmc_key_index = 0
        self.google_key_index = 0
        
        # Compteurs d'utilisation et d'erreurs pour chaque clé
        self.cmc_key_usage = {key: 0 for key in self.coinmarketcap_api_keys}
        self.cmc_key_errors = {key: 0 for key in self.coinmarketcap_api_keys}
        self.google_key_usage = {key: 0 for key in self.google_api_keys}
        self.google_key_errors = {key: 0 for key in self.google_api_keys}
        
        if not self.coinmarketcap_api_keys:
            logger.warning("Aucune clé API CoinMarketCap fournie. Certaines fonctionnalités seront limitées.")
        else:
            logger.info(f"Initialisation avec {len(self.coinmarketcap_api_keys)} clés API CoinMarketCap")
        
        if not self.google_api_keys:
            logger.warning("Aucune clé API Google fournie. Certaines fonctionnalités seront limitées.")
        else:
            logger.info(f"Initialisation avec {len(self.google_api_keys)} clés API Google")
        
        # Limites d'appels API pour éviter de dépasser les quotas
        self.cmc_last_call_time = 0
        self.cmc_call_limit = 30  # 30 appels par minute pour le plan gratuit
        
        # Cache pour éviter des appels API répétés
        self.crypto_metadata_cache = {}
        self.market_metrics_cache = {}
        self.news_cache = {}
    
    def get_coinmarketcap_key(self):
        """
        Récupère une clé API CoinMarketCap en utilisant une stratégie de rotation.
        Privilégie les clés les moins utilisées et évite celles qui ont généré des erreurs.
        
        Returns:
            Clé API CoinMarketCap ou None si aucune clé n'est disponible
        """
        if not self.coinmarketcap_api_keys:
            return None
        
        # Trier les clés par nombre d'erreurs puis par nombre d'utilisations
        sorted_keys = sorted(self.coinmarketcap_api_keys, 
                             key=lambda k: (self.cmc_key_errors.get(k, 0), self.cmc_key_usage.get(k, 0)))
        
        # Sélectionner la clé avec le moins d'erreurs et d'utilisations
        key = sorted_keys[0]
        self.cmc_key_usage[key] = self.cmc_key_usage.get(key, 0) + 1
        
        return key
    
    def get_google_key(self):
        """
        Récupère une clé API Google en utilisant une stratégie de rotation.
        Privilégie les clés les moins utilisées et évite celles qui ont généré des erreurs.
        
        Returns:
            Clé API Google ou None si aucune clé n'est disponible
        """
        if not self.google_api_keys:
            return None
        
        # Trier les clés par nombre d'erreurs puis par nombre d'utilisations
        sorted_keys = sorted(self.google_api_keys, 
                             key=lambda k: (self.google_key_errors.get(k, 0), self.google_key_usage.get(k, 0)))
        
        # Sélectionner la clé avec le moins d'erreurs et d'utilisations
        key = sorted_keys[0]
        self.google_key_usage[key] = self.google_key_usage.get(key, 0) + 1
        
        return key
    
    def _respect_rate_limit(self, last_call_time, call_limit):
        """
        Respecte les limites de taux d'appels API.
        
        Args:
            last_call_time: Horodatage du dernier appel
            call_limit: Limite d'appels par minute
        
        Returns:
            Horodatage mis à jour
        """
        current_time = time.time()
        elapsed = current_time - last_call_time
        min_interval = 60.0 / call_limit  # Intervalle minimum entre les appels
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug(f"Attente de {sleep_time:.2f} secondes pour respecter la limite d'appels API")
            time.sleep(sleep_time)
        
        return time.time()
    
    def get_crypto_metadata(self, symbol):
        """
        Récupère les métadonnées d'une crypto-monnaie.
        
        Args:
            symbol: Symbole de la crypto-monnaie (ex: BTC)
        
        Returns:
            Dictionnaire de métadonnées
        """
        # Vérifier si les données sont en cache
        if symbol in self.crypto_metadata_cache:
            return self.crypto_metadata_cache[symbol]
        
        if not self.coinmarketcap_api_keys:
            logger.warning("Impossible de récupérer les métadonnées: aucune clé API CoinMarketCap")
            return {}
        
        # Récupérer une clé API CoinMarketCap
        api_key = self.get_coinmarketcap_key()
        if not api_key:
            logger.error("Impossible de récupérer une clé API CoinMarketCap")
            return {}
        
        # Respecter la limite de taux d'appels
        self.cmc_last_call_time = self._respect_rate_limit(self.cmc_last_call_time, self.cmc_call_limit)
        
        # Construire l'URL de l'API
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/info'
        parameters = {
            'symbol': symbol
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            data = response.json()
            
            if response.status_code == 200 and 'data' in data:
                metadata = data['data'].get(symbol, {})
                self.crypto_metadata_cache[symbol] = metadata
                return metadata
            else:
                logger.error(f"Erreur lors de la récupération des métadonnées pour {symbol}: {data.get('status', {}).get('error_message', 'Erreur inconnue')}")
                self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
                return {}
        except Exception as e:
            logger.error(f"Exception lors de la récupération des métadonnées pour {symbol}: {str(e)}")
            self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
            return {}
    
    def get_market_metrics(self, symbol):
        """
        Récupère les métriques de marché d'une crypto-monnaie.
        
        Args:
            symbol: Symbole de la crypto-monnaie (ex: BTC)
        
        Returns:
            Dictionnaire de métriques de marché
        """
        # Vérifier si les données sont en cache et si elles sont récentes (moins de 1 heure)
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.market_metrics_cache:
            return self.market_metrics_cache[cache_key]
        
        if not self.coinmarketcap_api_keys:
            logger.warning("Impossible de récupérer les métriques de marché: aucune clé API CoinMarketCap")
            return {}
        
        # Récupérer une clé API CoinMarketCap
        api_key = self.get_coinmarketcap_key()
        if not api_key:
            logger.error("Impossible de récupérer une clé API CoinMarketCap")
            return {}
        
        # Respecter la limite de taux d'appels
        self.cmc_last_call_time = self._respect_rate_limit(self.cmc_last_call_time, self.cmc_call_limit)
        
        # Construire l'URL de l'API
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        parameters = {
            'symbol': symbol,
            'convert': 'USD'
        }
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            data = response.json()
            
            if response.status_code == 200 and 'data' in data:
                metrics = data['data'].get(symbol, {})
                quote = metrics.get('quote', {}).get('USD', {})
                
                # Extraire les métriques pertinentes
                result = {
                    'market_cap': quote.get('market_cap', 0),
                    'volume_24h': quote.get('volume_24h', 0),
                    'percent_change_1h': quote.get('percent_change_1h', 0),
                    'percent_change_24h': quote.get('percent_change_24h', 0),
                    'percent_change_7d': quote.get('percent_change_7d', 0),
                    'circulating_supply': metrics.get('circulating_supply', 0),
                    'total_supply': metrics.get('total_supply', 0),
                    'max_supply': metrics.get('max_supply', 0),
                    'cmc_rank': metrics.get('cmc_rank', 0),
                    'num_market_pairs': metrics.get('num_market_pairs', 0),
                    'last_updated': quote.get('last_updated', '')
                }
                
                self.market_metrics_cache[cache_key] = result
                return result
            else:
                logger.error(f"Erreur lors de la récupération des métriques pour {symbol}: {data.get('status', {}).get('error_message', 'Erreur inconnue')}")
                self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
                return {}
        except Exception as e:
            logger.error(f"Exception lors de la récupération des métriques pour {symbol}: {str(e)}")
            self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
            return {}
    
    def get_crypto_news(self, symbol, limit=10):
        """
        Récupère les actualités récentes d'une crypto-monnaie en utilisant l'API Google Custom Search.
        
        Args:
            symbol: Symbole de la crypto-monnaie (ex: BTC)
            limit: Nombre maximum d'articles à récupérer
        
        Returns:
            Liste d'articles d'actualité
        """
        # Vérifier si les données sont en cache et si elles sont récentes (moins de 6 heures)
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in self.news_cache:
            return self.news_cache[cache_key]
        
        if not self.google_api_keys:
            logger.warning("Impossible de récupérer les actualités: aucune clé API Google")
            return []
        
        # Récupérer une clé API Google
        api_key = self.get_google_key()
        if not api_key:
            logger.error("Impossible de récupérer une clé API Google")
            return []
        
        # Obtenir le nom complet de la crypto-monnaie pour améliorer la recherche
        crypto_name = symbol
        metadata = self.get_crypto_metadata(symbol)
        if metadata and 'name' in metadata:
            crypto_name = metadata['name']
        
        # Construire l'URL de l'API
        url = 'https://www.googleapis.com/customsearch/v1'
        parameters = {
            'key': api_key,
            'cx': '017576662512468239146:omuauf_lfve',  # ID du moteur de recherche personnalisé
            'q': f"{crypto_name} cryptocurrency news",
            'num': limit,
            'sort': 'date'
        }
        
        try:
            response = requests.get(url, params=parameters)
            data = response.json()
            
            if response.status_code == 200 and 'items' in data:
                news_items = []
                for item in data['items']:
                    news_items.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': item.get('displayLink', ''),
                        'published_date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')
                    })
                
                self.news_cache[cache_key] = news_items
                return news_items
            else:
                logger.error(f"Erreur lors de la récupération des actualités pour {symbol}: {data.get('error', {}).get('message', 'Erreur inconnue')}")
                self.google_key_errors[api_key] = self.google_key_errors.get(api_key, 0) + 1
                return []
        except Exception as e:
            logger.error(f"Exception lors de la récupération des actualités pour {symbol}: {str(e)}")
            self.google_key_errors[api_key] = self.google_key_errors.get(api_key, 0) + 1
            return []
    
    def get_global_market_data(self):
        """
        Récupère les données globales du marché des crypto-monnaies.
        
        Returns:
            Dictionnaire de données globales du marché
        """
        if not self.coinmarketcap_api_keys:
            logger.warning("Impossible de récupérer les données globales du marché: aucune clé API CoinMarketCap")
            return {}
        
        # Récupérer une clé API CoinMarketCap
        api_key = self.get_coinmarketcap_key()
        if not api_key:
            logger.error("Impossible de récupérer une clé API CoinMarketCap")
            return {}
        
        # Respecter la limite de taux d'appels
        self.cmc_last_call_time = self._respect_rate_limit(self.cmc_last_call_time, self.cmc_call_limit)
        
        # Construire l'URL de l'API
        url = 'https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest'
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,
        }
        
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if response.status_code == 200 and 'data' in data:
                metrics = data['data']
                quote = metrics.get('quote', {}).get('USD', {})
                
                # Extraire les métriques pertinentes
                result = {
                    'total_market_cap': quote.get('total_market_cap', 0),
                    'total_volume_24h': quote.get('total_volume_24h', 0),
                    'btc_dominance': metrics.get('btc_dominance', 0),
                    'eth_dominance': metrics.get('eth_dominance', 0),
                    'active_cryptocurrencies': metrics.get('active_cryptocurrencies', 0),
                    'total_cryptocurrencies': metrics.get('total_cryptocurrencies', 0),
                    'active_exchanges': metrics.get('active_exchanges', 0),
                    'total_exchanges': metrics.get('total_exchanges', 0),
                    'last_updated': metrics.get('last_updated', '')
                }
                
                return result
            else:
                logger.error(f"Erreur lors de la récupération des données globales du marché: {data.get('status', {}).get('error_message', 'Erreur inconnue')}")
                self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
                return {}
        except Exception as e:
            logger.error(f"Exception lors de la récupération des données globales du marché: {str(e)}")
            self.cmc_key_errors[api_key] = self.cmc_key_errors.get(api_key, 0) + 1
            return {}
    
    def enrich_market_data(self, df, symbols):
        """
        Enrichit un DataFrame avec des données de marché supplémentaires.
        
        Args:
            df: DataFrame à enrichir
            symbols: Liste des symboles de crypto-monnaies
        
        Returns:
            DataFrame enrichi
        """
        logger.info(f"Enrichissement des données de marché pour {len(symbols)} symboles")
        
        # Créer une copie du DataFrame pour éviter de modifier l'original
        enriched_df = df.copy()
        
        # Récupérer les données globales du marché
        global_data = self.get_global_market_data()
        
        # Ajouter les colonnes pour les données globales du marché
        if global_data:
            for key, value in global_data.items():
                if key != 'last_updated':
                    enriched_df[f'global_{key}'] = value
        
        # Pour chaque symbole, récupérer les métriques de marché
        for symbol in symbols:
            # Extraire le symbole de base (sans la partie quote)
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            
            # Récupérer les métriques de marché
            metrics = self.get_market_metrics(base_symbol)
            
            if metrics:
                # Ajouter les métriques au DataFrame pour les lignes correspondant à ce symbole
                mask = enriched_df['symbol'] == symbol
                for key, value in metrics.items():
                    if key != 'last_updated':
                        enriched_df.loc[mask, f'{base_symbol}_{key}'] = value
        
        logger.info(f"Données de marché enrichies avec succès")
        return enriched_df

# Fonction pour initialiser le collecteur avec les clés API
def initialize_market_info_collector():
    """
    Initialise le collecteur d'informations de marché avec les clés API.
    
    Returns:
        Instance de MarketInfoCollector
    """
    # Définir les clés API
    coinmarketcap_api_keys = [
        os.getenv('COINMARKETCAP_API_KEY_1', '0853fffa-27e7-45c7-b549-5e857416275f'),
        os.getenv('COINMARKETCAP_API_KEY_2', 'b54bcf4d-1bca-4e8e-9a24-22ff2c3d462c'),
        os.getenv('COINMARKETCAP_API_KEY_3', '9e10c322-94c4-4965-b44e-3c3b3cd3d713')
    ]
    
    google_api_keys = [
        os.getenv('GOOGLE_API_KEY_1', 'AIzaSyD6NcyTvM73dLJupxS5NFGvv5AtaWhbifU'),
        os.getenv('GOOGLE_API_KEY_2', 'AIzaSyCH9ocKRxb_4l7AcdKbzPJAAV1xZ1s-tMQ'),
        os.getenv('GOOGLE_API_KEY_3', 'AIzaSyAW7_-kTU1EH8dMXv0oBeDt49mH6lmkZhg')
    ]
    
    # Initialiser le collecteur
    collector = MarketInfoCollector(
        coinmarketcap_api_keys=coinmarketcap_api_keys,
        google_api_keys=google_api_keys
    )
    
    return collector

# Test du module si exécuté directement
if __name__ == "__main__":
    # Initialiser le collecteur
    collector = initialize_market_info_collector()
    
    # Tester la récupération des métadonnées
    btc_metadata = collector.get_crypto_metadata('BTC')
    print("\nBTC Metadata:")
    print(json.dumps(btc_metadata, indent=2))
    
    # Tester la récupération des métriques de marché
    btc_metrics = collector.get_market_metrics('BTC')
    print("\nBTC Market Metrics:")
    print(json.dumps(btc_metrics, indent=2))
    
    # Tester la récupération des actualités
    btc_news = collector.get_crypto_news('BTC', limit=3)
    print("\nBTC News:")
    for news in btc_news:
        print(f"- {news['title']} ({news['source']})")
    
    # Tester la récupération des données globales du marché
    global_data = collector.get_global_market_data()
    print("\nGlobal Market Data:")
    print(json.dumps(global_data, indent=2))
