#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter le trading live avec le modèle Morningstar.

Ce script charge la configuration, initialise le moteur de trading et lance
la boucle de trading en temps réel sur les exchanges configurés.

Utilisation:
    python run_trading_live.py --config config/trading_live.yaml
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from live.trading_engine import TradingEngine

# Configuration du logging
log_dir = Path("logs/trading")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "trading_live.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
        
    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas
        yaml.YAMLError: Si le fichier YAML est mal formé
    """
    logger.info(f"Chargement de la configuration depuis {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info("Configuration chargée avec succès")
        return config
    except FileNotFoundError:
        logger.error(f"Fichier de configuration non trouvé: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Erreur dans le fichier YAML: {e}")
        raise

def create_default_config(config_path: str) -> None:
    """
    Crée un fichier de configuration par défaut si aucun n'existe.
    
    Args:
        config_path: Chemin où créer le fichier de configuration
    """
    logger.info(f"Création d'un fichier de configuration par défaut: {config_path}")
    
    default_config = {
        "exchange": {
            "exchange_id": "bitget",
            "testnet": True,
            "api_key": "",  # À remplir
            "api_secret": "",  # À remplir
            "max_retries": 3,
            "retry_delay": 2.0
        },
        "model_path": "model/saved_model/morningstar_final.h5",
        "hmm_model_path": "model/saved_model/market_regime_detector.joblib",
        "trading_params": {
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "live_trading": False,  # Désactivé par défaut pour la sécurité
            "interval_seconds": 3600  # 1 heure
        },
        "risk_params": {
            "risk_per_trade": 0.02,  # 2% du capital par trade
            "max_position_size": 0.1,  # 10% du capital max
            "min_confidence": 0.7,  # Confiance minimale pour exécuter un signal
            "use_sl_tp": True  # Utiliser les stop-loss et take-profit
        },
        "log_dir": "logs/trading"
    }
    
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Écrire la configuration par défaut
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration par défaut créée: {config_path}")
    logger.info("IMPORTANT: Veuillez éditer ce fichier pour configurer vos clés API et autres paramètres avant de lancer le trading live.")

def main():
    """
    Fonction principale qui initialise et exécute le trading live.
    """
    parser = argparse.ArgumentParser(description="Trading live avec le modèle Morningstar")
    parser.add_argument("--config", type=str, default="config/trading_live.yaml",
                        help="Chemin vers le fichier de configuration")
    parser.add_argument("--create-config", action="store_true",
                        help="Crée un fichier de configuration par défaut")
    parser.add_argument("--interval", type=int, default=None,
                        help="Intervalle entre les itérations en secondes (remplace la valeur de la config)")
    parser.add_argument("--test-run", action="store_true",
                        help="Exécute une seule itération pour tester la configuration")
    
    args = parser.parse_args()
    
    # Créer une configuration par défaut si demandé
    if args.create_config:
        create_default_config(args.config)
        logger.info("Configuration par défaut créée. Veuillez l'éditer avant de lancer le trading live.")
        return
    
    try:
        # Charger la configuration
        config = load_config(args.config)
        
        # Remplacer l'intervalle si spécifié en ligne de commande
        if args.interval is not None:
            config["trading_params"]["interval_seconds"] = args.interval
            logger.info(f"Intervalle remplacé par {args.interval} secondes")
        
        # Initialiser le moteur de trading
        engine = TradingEngine(config)
        
        # Exécuter le trading
        if args.test_run:
            logger.info("Exécution d'une itération de test...")
            results = engine.run_single_iteration()
            logger.info(f"Résultats du test: {results}")
        else:
            # Récupérer l'intervalle depuis la configuration
            interval = config["trading_params"].get("interval_seconds", 3600)  # 1 heure par défaut
            logger.info(f"Démarrage de la boucle de trading avec intervalle de {interval} secondes")
            engine.run_trading_loop(interval)
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du trading live: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
