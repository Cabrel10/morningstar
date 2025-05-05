#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'initialisation pour l'environnement de trading Morningstar.

Ce script permet de configurer facilement l'environnement de trading
en cru00e9ant les ru00e9pertoires nu00e9cessaires et en gu00e9nu00e9rant les fichiers
de configuration par du00e9faut.

Utilisation:
    python init_trading_env.py
"""

import os
import sys
import yaml
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Ru00e9pertoires u00e0 cru00e9er
DIRECTORIES = [
    "data/raw",
    "data/processed",
    "data/llm_cache/news",
    "data/llm_cache/instruments",
    "logs/trading",
    "logs/backtest",
    "model/saved_model",
    "results/backtest",
    "results/trading",
    "config",
]

# Fichiers de configuration u00e0 gu00e9nu00e9rer
CONFIG_FILES = {
    "config/trading_live.yaml": {
        "exchange": {
            "exchange_id": "bitget",
            "testnet": True,
            "api_key": "",
            "api_secret": "",
            "max_retries": 3,
            "retry_delay": 2.0
        },
        "model_path": "model/saved_model/morningstar_final.h5",
        "hmm_model_path": "model/saved_model/market_regime_detector.joblib",
        "trading_params": {
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "live_trading": False,
            "interval_seconds": 3600
        },
        "risk_params": {
            "risk_per_trade": 0.02,
            "max_position_size": 0.1,
            "min_confidence": 0.7,
            "use_sl_tp": True
        },
        "log_dir": "logs/trading"
    },
    "config/secrets.env.example": """
# Fichier d'exemple pour les secrets (u00e0 copier vers secrets.env)

# API Keys pour les exchanges
BITGET_API_KEY=your_bitget_api_key_here
BITGET_API_SECRET=your_bitget_api_secret_here

BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

KUCOIN_API_KEY=your_kucoin_api_key_here
KUCOIN_API_SECRET=your_kucoin_api_secret_here
KUCOIN_API_PASSPHRASE=your_kucoin_api_passphrase_here

# API Keys pour les services LLM
GEMINI_API_KEY=your_gemini_api_key_here
""",
    "config/backtest_config.yaml": {
        "data": {
            "data_dir": "data/processed",
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01"
        },
        "model": {
            "model_path": "model/saved_model/morningstar_final.h5",
            "hmm_model_path": "model/saved_model/market_regime_detector.joblib"
        },
        "backtest": {
            "initial_capital": 10000.0,
            "commission": 0.001,
            "slippage": 0.0005,
            "signal_threshold": 0.6,
            "use_sl_tp": True,
            "default_sl_pct": 0.02,
            "default_tp_pct": 0.04,
            "risk_per_trade": 0.02
        },
        "output": {
            "results_dir": "results/backtest",
            "save_trades": True,
            "plot_results": True
        }
    }
}

def create_directories():
    """
    Cru00e9e les ru00e9pertoires nu00e9cessaires pour le projet.
    """
    logger.info("Cru00e9ation des ru00e9pertoires...")
    for directory in DIRECTORIES:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ru00e9pertoire cru00e9u00e9: {directory}")
        else:
            logger.info(f"Ru00e9pertoire existant: {directory}")

def generate_config_files():
    """
    Gu00e9nu00e8re les fichiers de configuration par du00e9faut.
    """
    logger.info("Gu00e9nu00e9ration des fichiers de configuration...")
    for file_path, content in CONFIG_FILES.items():
        path = Path(file_path)
        if not path.exists():
            with open(path, "w") as f:
                if isinstance(content, dict):
                    yaml.dump(content, f, default_flow_style=False, sort_keys=False)
                else:
                    f.write(content)
            logger.info(f"Fichier de configuration cru00e9u00e9: {file_path}")
        else:
            logger.info(f"Fichier de configuration existant: {file_path}")

def create_sample_data():
    """
    Cru00e9e des exemples de donnu00e9es pour les tests.
    """
    logger.info("Cru00e9ation d'exemples de donnu00e9es pour les tests...")
    
    # Exemple de donnu00e9es OHLCV pour les tests
    sample_data_path = Path("data/raw/sample_btcusdt_1h.csv")
    if not sample_data_path.exists():
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            # Gu00e9nu00e9rer des donnu00e9es OHLCV fictives
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 31)
            dates = [start_date + timedelta(hours=i) for i in range(int((end_date - start_date).total_seconds() / 3600) + 1)]
            
            # Gu00e9nu00e9rer des prix fictifs
            base_price = 20000.0
            prices = [base_price + np.random.normal(0, 200) for _ in range(len(dates))]
            
            # Cru00e9er le DataFrame
            df = pd.DataFrame({
                "timestamp": dates,
                "open": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": [np.random.uniform(100, 1000) for _ in range(len(dates))]
            })
            
            # Sauvegarder les donnu00e9es
            df.to_csv(sample_data_path, index=False)
            logger.info(f"Donnu00e9es d'exemple cru00e9u00e9es: {sample_data_path}")
        except ImportError:
            logger.warning("Impossible de cru00e9er des donnu00e9es d'exemple: pandas ou numpy non installu00e9s")
    else:
        logger.info(f"Donnu00e9es d'exemple existantes: {sample_data_path}")

def main():
    parser = argparse.ArgumentParser(description="Initialisation de l'environnement de trading Morningstar")
    parser.add_argument("--with-sample-data", action="store_true", help="Cru00e9er des donnu00e9es d'exemple pour les tests")
    args = parser.parse_args()
    
    logger.info("Initialisation de l'environnement de trading Morningstar...")
    
    # Cru00e9er les ru00e9pertoires
    create_directories()
    
    # Gu00e9nu00e9rer les fichiers de configuration
    generate_config_files()
    
    # Cru00e9er des donnu00e9es d'exemple si demandu00e9
    if args.with_sample_data:
        create_sample_data()
    
    logger.info("Initialisation terminu00e9e avec succu00e8s!")
    logger.info("\nProchaines u00e9tapes:")
    logger.info("1. Copiez config/secrets.env.example vers config/secrets.env et remplissez vos clu00e9s API")
    logger.info("2. Configurez vos paramu00e8tres de trading dans config/trading_live.yaml")
    logger.info("3. Tu00e9lu00e9chargez des donnu00e9es avec utils/api_manager.py ou utilisez les donnu00e9es d'exemple")
    logger.info("4. Exu00e9cutez un backtest avec run_backtest.py")
    logger.info("5. Lancez le trading live avec run_trading_live.py")

if __name__ == "__main__":
    main()
