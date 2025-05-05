#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour configurer les clu00e9s API et les variables d'environnement.
Ce script cru00e9e un fichier .env avec les clu00e9s API nu00e9cessaires.
"""

import os
import argparse
import logging
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Configuration des clu00e9s API")
    parser.add_argument(
        "--coinmarketcap-api-key",
        type=str,
        default="0853fffa-27e7-45c7-b549-5e857416275f",
        help="Clu00e9 API CoinMarketCap"
    )
    parser.add_argument(
        "--google-api-key-1",
        type=str,
        default="AIzaSyD6NcyTvM73dLJupxS5NFGvv5AtaWhbifU",
        help="Clu00e9 API Google 1"
    )
    parser.add_argument(
        "--google-api-key-2",
        type=str,
        default="AIzaSyCH9ocKRxb_4l7AcdKbzPJAAV1xZ1s-tMQ",
        help="Clu00e9 API Google 2"
    )
    parser.add_argument(
        "--google-api-key-3",
        type=str,
        default="AIzaSyAW7_-kTU1EH8dMXv0oBeDt49mH6lmkZhg",
        help="Clu00e9 API Google 3"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default="",
        help="Clu00e9 API Gemini (optionnelle)"
    )
    parser.add_argument(
        "--exchange-api-key",
        type=str,
        default="",
        help="Clu00e9 API de l'u00e9change (optionnelle)"
    )
    parser.add_argument(
        "--exchange-api-secret",
        type=str,
        default="",
        help="Secret API de l'u00e9change (optionnelle)"
    )
    parser.add_argument(
        "--use-mock-exchange",
        action="store_true",
        help="Utiliser un u00e9change simulu00e9 pour les tests"
    )
    
    return parser.parse_args()

def setup_env_file(args):
    """
    Configure le fichier .env avec les clu00e9s API.
    
    Args:
        args: Arguments parsu00e9s
    """
    # Cru00e9er le contenu du fichier .env
    env_content = f"""# Fichier de configuration des clu00e9s API et variables d'environnement
# Cru00e9u00e9 automatiquement par setup_api_keys.py

# CoinMarketCap API
COINMARKETCAP_API_KEY={args.coinmarketcap_api_key}

# Google API
GOOGLE_API_KEY_1={args.google_api_key_1}
GOOGLE_API_KEY_2={args.google_api_key_2}
GOOGLE_API_KEY_3={args.google_api_key_3}

# Gemini API
GEMINI_API_KEY={args.gemini_api_key}

# Exchange API
EXCHANGE_API_KEY={args.exchange_api_key}
EXCHANGE_API_SECRET={args.exchange_api_secret}
USE_MOCK_EXCHANGE={'true' if args.use_mock_exchange else 'false'}
"""
    
    # u00c9crire le fichier .env
    env_path = Path(".env")
    env_path.write_text(env_content)
    
    logger.info(f"Fichier .env cru00e9u00e9 avec succu00e8s dans {env_path.absolute()}")
    logger.info("Variables d'environnement configurées:")
    logger.info(f"- COINMARKETCAP_API_KEY: {'*' * 8 + args.coinmarketcap_api_key[-4:] if args.coinmarketcap_api_key else 'Non configurée'}")
    logger.info(f"- GOOGLE_API_KEY_1: {'*' * 8 + args.google_api_key_1[-4:] if args.google_api_key_1 else 'Non configurée'}")
    logger.info(f"- GOOGLE_API_KEY_2: {'*' * 8 + args.google_api_key_2[-4:] if args.google_api_key_2 else 'Non configurée'}")
    logger.info(f"- GOOGLE_API_KEY_3: {'*' * 8 + args.google_api_key_3[-4:] if args.google_api_key_3 else 'Non configurée'}")
    logger.info(f"- GEMINI_API_KEY: {'Configurée' if args.gemini_api_key else 'Non configurée'}")
    logger.info(f"- EXCHANGE_API_KEY: {'Configurée' if args.exchange_api_key else 'Non configurée'}")
    logger.info(f"- EXCHANGE_API_SECRET: {'Configurée' if args.exchange_api_secret else 'Non configurée'}")
    logger.info(f"- USE_MOCK_EXCHANGE: {'true' if args.use_mock_exchange else 'false'}")

def main():
    """
    Fonction principale.
    """
    # Parser les arguments
    args = parse_args()
    
    # Configurer le fichier .env
    setup_env_file(args)
    
    logger.info("Configuration des clu00e9s API terminée")
    logger.info("Vous pouvez maintenant exu00e9cuter les scripts de collecte de donnu00e9es et d'entrau00eenement du modu00e8le")

if __name__ == "__main__":
    main()
