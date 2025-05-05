#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour l'intégration des API Binance et Bitget.
Ce script permet de tester la connexion aux API et de vérifier les fonctionnalités
de base comme la récupération des soldes, des prix et l'exécution d'ordres de test.
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(str(Path(__file__).resolve().parent))

from live.exchange_manager import ExchangeManager
from config.api_keys import EXCHANGE_API_KEYS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_exchange_connection(exchange_id, use_testnet=True):
    """
    Teste la connexion à un échange.
    
    Args:
        exchange_id (str): Identifiant de l'échange ('binance', 'bitget')
        use_testnet (bool): Utiliser le testnet au lieu du mainnet
    
    Returns:
        bool: True si la connexion est réussie, False sinon
    """
    try:
        logger.info(f"Test de connexion à l'échange {exchange_id} (testnet: {use_testnet})")
        
        # Créer le gestionnaire d'échange
        exchange_manager = ExchangeManager(exchange_id=exchange_id, use_testnet=use_testnet)
        
        # Vérifier que l'échange est initialisé
        if exchange_manager.exchange is None:
            logger.error(f"Échec de l'initialisation de l'échange {exchange_id}")
            return False
        
        # Récupérer les paramètres spécifiques à l'échange
        params = exchange_manager.get_exchange_parameters()
        logger.info(f"Paramètres de l'échange {exchange_id}: {params}")
        
        # Test réussi
        logger.info(f"Connexion à l'échange {exchange_id} réussie")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du test de connexion à {exchange_id}: {e}")
        return False

def test_market_data(exchange_id, symbol="BTC/USDT", use_testnet=True):
    """
    Teste la récupération des données de marché.
    
    Args:
        exchange_id (str): Identifiant de l'échange
        symbol (str): Symbole à tester
        use_testnet (bool): Utiliser le testnet
    
    Returns:
        bool: True si le test est réussi, False sinon
    """
    try:
        logger.info(f"Test de récupération des données de marché pour {symbol} sur {exchange_id}")
        
        # Créer le gestionnaire d'échange
        exchange_manager = ExchangeManager(exchange_id=exchange_id, use_testnet=use_testnet)
        
        # Récupérer les données OHLCV
        ohlcv_data = exchange_manager.get_market_data(symbol=symbol, timeframe="1h", limit=10)
        
        if ohlcv_data.empty:
            logger.error(f"Aucune donnée OHLCV récupérée pour {symbol} sur {exchange_id}")
            return False
        
        logger.info(f"Données OHLCV pour {symbol} sur {exchange_id}:")
        logger.info(f"\n{ohlcv_data.head()}")
        
        # Récupérer les caractéristiques spécifiques à l'échange
        features = exchange_manager.get_exchange_specific_features(symbol)
        logger.info(f"Caractéristiques spécifiques pour {symbol} sur {exchange_id}: {features}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du test des données de marché sur {exchange_id}: {e}")
        return False

def test_account_balance(exchange_id, use_testnet=True):
    """
    Teste la récupération des soldes du compte.
    
    Args:
        exchange_id (str): Identifiant de l'échange
        use_testnet (bool): Utiliser le testnet
    
    Returns:
        bool: True si le test est réussi, False sinon
    """
    try:
        logger.info(f"Test de récupération des soldes sur {exchange_id}")
        
        # Créer le gestionnaire d'échange
        exchange_manager = ExchangeManager(exchange_id=exchange_id, use_testnet=use_testnet)
        
        # Récupérer les soldes
        balances = exchange_manager.get_balance(force_update=True)
        
        if not balances:
            logger.warning(f"Aucun solde récupéré pour {exchange_id}. Vérifiez les clés API.")
            return False
        
        logger.info(f"Soldes sur {exchange_id}:")
        for currency, amount in balances.items():
            if amount > 0:
                logger.info(f"  {currency}: {amount}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du test des soldes sur {exchange_id}: {e}")
        return False

def test_order_management(exchange_id, symbol="BTC/USDT", use_testnet=True):
    """
    Teste la gestion des ordres (récupération des ordres ouverts).
    
    Args:
        exchange_id (str): Identifiant de l'échange
        symbol (str): Symbole à tester
        use_testnet (bool): Utiliser le testnet
    
    Returns:
        bool: True si le test est réussi, False sinon
    """
    try:
        logger.info(f"Test de gestion des ordres pour {symbol} sur {exchange_id}")
        
        # Créer le gestionnaire d'échange
        exchange_manager = ExchangeManager(exchange_id=exchange_id, use_testnet=use_testnet)
        
        # Récupérer les ordres ouverts
        open_orders = exchange_manager.get_open_orders(symbol)
        
        logger.info(f"Ordres ouverts pour {symbol} sur {exchange_id}: {len(open_orders)}")
        for order in open_orders:
            logger.info(f"  Ordre {order.get('id')}: {order.get('side')} {order.get('amount')} @ {order.get('price')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du test de gestion des ordres sur {exchange_id}: {e}")
        return False

def test_all_exchanges(use_testnet=True):
    """
    Teste tous les échanges configurés.
    
    Args:
        use_testnet (bool): Utiliser le testnet
    """
    results = {}
    
    # Tester chaque échange configuré
    for exchange_id in EXCHANGE_API_KEYS.keys():
        logger.info(f"=== Test de l'échange {exchange_id} ===")
        
        # Vérifier si les clés API sont configurées
        api_config = EXCHANGE_API_KEYS.get(exchange_id, {})
        if not api_config.get('api_key') or not api_config.get('api_secret'):
            logger.warning(f"Clés API non configurées pour {exchange_id}. Test ignoré.")
            results[exchange_id] = "Non configuré"
            continue
        
        # Tester la connexion
        connection_result = test_exchange_connection(exchange_id, use_testnet)
        
        if not connection_result:
            logger.error(f"Échec de connexion à {exchange_id}. Tests suivants ignorés.")
            results[exchange_id] = "Échec de connexion"
            continue
        
        # Tester les données de marché
        market_data_result = test_market_data(exchange_id, use_testnet=use_testnet)
        
        # Tester les soldes
        balance_result = test_account_balance(exchange_id, use_testnet=use_testnet)
        
        # Tester la gestion des ordres
        order_result = test_order_management(exchange_id, use_testnet=use_testnet)
        
        # Résumé des résultats
        results[exchange_id] = {
            "connexion": "OK" if connection_result else "Échec",
            "données_marché": "OK" if market_data_result else "Échec",
            "soldes": "OK" if balance_result else "Échec",
            "ordres": "OK" if order_result else "Échec"
        }
    
    # Afficher le résumé des résultats
    logger.info("\n=== Résumé des tests ===")
    for exchange_id, result in results.items():
        if isinstance(result, str):
            logger.info(f"{exchange_id}: {result}")
        else:
            logger.info(f"{exchange_id}:")
            for test_name, test_result in result.items():
                logger.info(f"  {test_name}: {test_result}")

def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description="Test d'intégration des API d'échange")
    parser.add_argument('--exchange', '-e', type=str, choices=['binance', 'bitget', 'all'], default='all',
                        help="Échange à tester (binance, bitget, all)")
    parser.add_argument('--mainnet', action='store_true', help="Utiliser le mainnet au lieu du testnet")
    parser.add_argument('--symbol', '-s', type=str, default="BTC/USDT", help="Symbole à tester")
    
    args = parser.parse_args()
    use_testnet = not args.mainnet
    
    logger.info(f"Début des tests d'intégration (testnet: {use_testnet})")
    
    if args.exchange == 'all':
        test_all_exchanges(use_testnet=use_testnet)
    else:
        # Vérifier si les clés API sont configurées
        api_config = EXCHANGE_API_KEYS.get(args.exchange, {})
        if not api_config.get('api_key') or not api_config.get('api_secret'):
            logger.warning(f"Clés API non configurées pour {args.exchange}.")
            return
        
        # Tester l'échange spécifié
        connection_result = test_exchange_connection(args.exchange, use_testnet)
        
        if connection_result:
            test_market_data(args.exchange, args.symbol, use_testnet)
            test_account_balance(args.exchange, use_testnet)
            test_order_management(args.exchange, args.symbol, use_testnet)
    
    logger.info("Tests terminés")

if __name__ == "__main__":
    main()
