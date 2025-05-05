#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour vu00e9rifier la connexion aux exchanges et les fonctionnalitu00e9s de base.

Ce script permet de tester rapidement si la configuration de l'API est correcte
et si les fonctionnalitu00e9s essentielles (ru00e9cupu00e9ration de donnu00e9es, soldes, etc.) fonctionnent.

Utilisation:
    python test_exchange_connection.py --exchange bitget --testnet
"""

import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any

from live.exchange_integration import ExchangeFactory

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def load_api_keys(exchange_id: str) -> Dict[str, str]:
    """
    Charge les clu00e9s API depuis le fichier de configuration.
    
    Args:
        exchange_id: Identifiant de l'exchange
        
    Returns:
        Dictionnaire contenant les clu00e9s API
    """
    config_path = Path("config/trading_live.yaml")
    
    if not config_path.exists():
        logger.warning(f"Fichier de configuration {config_path} non trouvu00e9. Utilisation de clu00e9s vides.")
        return {"api_key": "", "api_secret": ""}
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        exchange_config = config.get("exchange", {})
        
        if exchange_config.get("exchange_id", "").lower() == exchange_id.lower():
            return {
                "api_key": exchange_config.get("api_key", ""),
                "api_secret": exchange_config.get("api_secret", ""),
                "password": exchange_config.get("password", "")  # Pour KuCoin
            }
        else:
            logger.warning(f"Configuration pour {exchange_id} non trouvu00e9e dans {config_path}")
            return {"api_key": "", "api_secret": ""}
            
    except Exception as e:
        logger.error(f"Erreur lors du chargement des clu00e9s API: {e}")
        return {"api_key": "", "api_secret": ""}

def test_exchange_connection(exchange_id: str, testnet: bool = False, symbol: str = "BTC/USDT"):
    """
    Teste la connexion u00e0 un exchange et ses fonctionnalitu00e9s de base.
    
    Args:
        exchange_id: Identifiant de l'exchange
        testnet: Utiliser le testnet
        symbol: Symbole de la paire u00e0 tester
    """
    logger.info(f"Test de connexion u00e0 l'exchange {exchange_id} (testnet: {testnet})")
    
    # Charger les clu00e9s API
    api_keys = load_api_keys(exchange_id)
    
    # Configuration de l'exchange
    config = {
        "exchange_id": exchange_id,
        "testnet": testnet,
        "api_key": api_keys.get("api_key", ""),
        "api_secret": api_keys.get("api_secret", ""),
        "password": api_keys.get("password", ""),  # Pour KuCoin
        "max_retries": 3,
        "retry_delay": 2.0
    }
    
    try:
        # Cru00e9er l'instance de l'exchange
        exchange = ExchangeFactory.create_exchange(exchange_id, config)
        logger.info(f"Connexion u00e0 {exchange_id} u00e9tablie avec succu00e8s")
        
        # Test 1: Ru00e9cupu00e9ration des marchu00e9s
        logger.info("Test 1: Ru00e9cupu00e9ration des marchu00e9s")
        markets = exchange.markets
        logger.info(f"Nombre de marchu00e9s disponibles: {len(markets)}")
        
        # Test 2: Ru00e9cupu00e9ration du ticker
        logger.info(f"Test 2: Ru00e9cupu00e9ration du ticker pour {symbol}")
        ticker = exchange.fetch_ticker(symbol)
        logger.info(f"Ticker pour {symbol}: Prix = {ticker.get('last', 'N/A')}")
        
        # Test 3: Ru00e9cupu00e9ration des donnu00e9es OHLCV
        logger.info(f"Test 3: Ru00e9cupu00e9ration des donnu00e9es OHLCV pour {symbol}")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=10)
        logger.info(f"OHLCV pour {symbol}: {len(ohlcv)} bougies ru00e9cupu00e9ru00e9es")
        
        # Test 4: Ru00e9cupu00e9ration du solde (si clu00e9s API fournies)
        if api_keys.get("api_key") and api_keys.get("api_secret"):
            logger.info("Test 4: Ru00e9cupu00e9ration du solde")
            try:
                balance = exchange.fetch_balance()
                # Afficher uniquement les soldes non nuls
                non_zero_balances = {currency: data for currency, data in balance.items()
                                   if isinstance(data, dict) and data.get('total', 0) > 0}
                logger.info(f"Soldes non nuls: {json.dumps(non_zero_balances, indent=2)}")
            except Exception as e:
                logger.error(f"Erreur lors de la ru00e9cupu00e9ration du solde: {e}")
        else:
            logger.warning("Test 4: Ru00e9cupu00e9ration du solde ignoru00e9e (clu00e9s API non fournies)")
        
        # Test 5: Test spu00e9cifique u00e0 l'exchange
        logger.info(f"Test 5: Test spu00e9cifique u00e0 {exchange_id}")
        if exchange_id.lower() == "bitget":
            # Test de la paire de test #BTC pour Bitget
            if hasattr(exchange, "get_btc_test_pair"):
                test_pair = exchange.get_btc_test_pair()
                logger.info(f"Paire de test Bitget: {test_pair}")
                test_ticker = exchange.fetch_ticker(test_pair)
                logger.info(f"Ticker pour {test_pair}: Prix = {test_ticker.get('last', 'N/A')}")
        elif exchange_id.lower() == "binance":
            # Test de ru00e9cupu00e9ration du taux de financement pour Binance Futures
            if hasattr(exchange, "fetch_funding_rate") and testnet:
                try:
                    funding_rate = exchange.fetch_funding_rate(symbol)
                    logger.info(f"Taux de financement pour {symbol}: {funding_rate}")
                except Exception as e:
                    logger.error(f"Erreur lors de la ru00e9cupu00e9ration du taux de financement: {e}")
        elif exchange_id.lower() == "kucoin":
            # Test de ru00e9cupu00e9ration des devises pour KuCoin
            if hasattr(exchange, "fetch_currencies"):
                try:
                    currencies = exchange.fetch_currencies()
                    logger.info(f"Nombre de devises disponibles sur KuCoin: {len(currencies)}")
                except Exception as e:
                    logger.error(f"Erreur lors de la ru00e9cupu00e9ration des devises: {e}")
        
        logger.info("Tous les tests ont u00e9tu00e9 exu00e9cutu00e9s avec succu00e8s")
        
    except Exception as e:
        logger.error(f"Erreur lors du test de connexion u00e0 {exchange_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test de connexion aux exchanges crypto")
    parser.add_argument("--exchange", type=str, default="bitget", 
                        choices=["bitget", "binance", "kucoin"],
                        help="Exchange u00e0 tester")
    parser.add_argument("--testnet", action="store_true",
                        help="Utiliser le testnet")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Symbole de la paire u00e0 tester")
    
    args = parser.parse_args()
    
    test_exchange_connection(args.exchange, args.testnet, args.symbol)

if __name__ == "__main__":
    main()
