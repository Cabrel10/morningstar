#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour exécuter un backtest détaillé du modèle Morningstar.
Génère des visualisations et des métriques de performance complètes.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import time
from datetime import datetime, timedelta
import traceback

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(BASE_DIR))

# Imports du projet
from backtesting.backtest_engine import BacktestEngine
from backtesting.visualization import BacktestVisualizer
from utils.data_preparation import load_and_prepare_data

# Configuration du logging
log_dir = Path("logs/backtest")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Charge la configuration depuis un fichier JSON ou YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"Format de fichier de configuration non supporté: {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return {}

def load_data(data_path: str, sentiment_path: str = None) -> tuple:
    """
    Charge les données de marché et de sentiment.
    
    Args:
        data_path: Chemin vers le fichier de données de marché
        sentiment_path: Chemin vers le fichier de données de sentiment (optionnel)
        
    Returns:
        Tuple (données de marché, données de sentiment)
    """
    try:
        # Charger les données de marché
        data_path = Path(data_path)
        if data_path.suffix == '.csv':
            market_data = pd.read_csv(data_path)
        elif data_path.suffix == '.parquet':
            market_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Format de fichier non supporté: {data_path}")
        
        # Convertir la colonne timestamp en datetime si nécessaire
        if 'timestamp' in market_data.columns and not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        
        # Charger les données de sentiment si fournies
        sentiment_data = None
        if sentiment_path:
            sentiment_path = Path(sentiment_path)
            if sentiment_path.suffix == '.csv':
                sentiment_data = pd.read_csv(sentiment_path)
            elif sentiment_path.suffix == '.parquet':
                sentiment_data = pd.read_parquet(sentiment_path)
            else:
                logger.warning(f"Format de fichier de sentiment non supporté: {sentiment_path}")
            
            # Convertir la colonne timestamp en datetime si nécessaire
            if sentiment_data is not None and 'timestamp' in sentiment_data.columns and not pd.api.types.is_datetime64_any_dtype(sentiment_data['timestamp']):
                sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
        
        return market_data, sentiment_data
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        traceback.print_exc()
        raise

def run_backtest(args):
    """
    Exécute le backtest avec les arguments fournis.
    
    Args:
        args: Arguments de ligne de commande
    """
    start_time = time.time()
    logger.info("Démarrage du backtest détaillé")
    
    try:
        # Charger la configuration
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # Mettre à jour la configuration avec les arguments de ligne de commande
        backtest_config = {
            'initial_capital': args.initial_capital,
            'position_size': args.position_size,
            'commission': args.commission,
            'slippage': args.slippage,
            'use_sl_tp': args.use_sl_tp
        }
        
        # Fusionner avec la configuration du fichier
        if 'backtest' in config:
            config['backtest'].update(backtest_config)
        else:
            config['backtest'] = backtest_config
        
        # Charger les données
        logger.info(f"Chargement des données depuis {args.data_file}")
        market_data, sentiment_data = load_data(args.data_file, args.sentiment_file)
        
        logger.info(f"Données chargées: {len(market_data)} points de marché, " + 
                   (f"{len(sentiment_data)} points de sentiment" if sentiment_data is not None else "pas de données de sentiment"))
        
        # Créer le répertoire de sortie
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le moteur de backtest
        logger.info(f"Initialisation du moteur de backtest avec le modèle: {args.model_dir}")
        engine = BacktestEngine(Path(args.model_dir), config['backtest'])
        
        # Exécuter le backtest
        logger.info("Exécution du backtest")
        metrics = engine.run_backtest(
            market_data, 
            sentiment_data, 
            args.symbol, 
            args.timeframe
        )
        
        # Sauvegarder les résultats
        logger.info(f"Sauvegarde des résultats dans {output_dir}")
        result_paths = engine.save_results(output_dir)
        
        # Générer les visualisations
        logger.info("Génération des visualisations")
        visualizer = BacktestVisualizer(output_dir)
        visualization_paths = visualizer.generate_performance_report(output_dir / 'visualizations')
        
        # Afficher les métriques principales
        print("\n=== Résultats du Backtest ===")
        print(f"Rendement total: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Rendement annualisé: {metrics.get('annual_return_pct', 0):.2f}%")
        print(f"Ratio de Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Drawdown maximum: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"Nombre de trades: {metrics.get('total_trades', 0)}")
        print(f"Taux de réussite: {metrics.get('win_rate_pct', 0):.2f}%")
        print(f"Rapport profit/perte: {metrics.get('profit_loss_ratio', 0):.2f}")
        print(f"Profit net: {metrics.get('net_profit', 0):.2f} ({metrics.get('net_profit_pct', 0):.2f}%)")
        
        # Afficher les chemins des résultats
        print("\n=== Fichiers générés ===")
        print(f"Résultats: {result_paths['results']}")
        print(f"Trades: {result_paths['trades']}")
        print(f"Métriques: {result_paths['metrics']}")
        print(f"Rapport de performance: {visualization_paths['report']}")
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        logger.info(f"Backtest terminé en {execution_time:.2f} secondes")
        print(f"\nBacktest terminé en {execution_time:.2f} secondes")
        
        return 0
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du backtest: {e}")
        traceback.print_exc()
        return 1

def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description="Backtest détaillé du modèle Morningstar")
    
    # Arguments obligatoires
    parser.add_argument("--model-dir", required=True, help="Répertoire du modèle")
    parser.add_argument("--data-file", required=True, help="Fichier de données de marché (CSV ou Parquet)")
    
    # Arguments optionnels
    parser.add_argument("--sentiment-file", help="Fichier de données de sentiment (CSV ou Parquet)")
    parser.add_argument("--output-dir", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--config", help="Fichier de configuration (JSON ou YAML)")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole de la paire")
    parser.add_argument("--timeframe", default="1h", help="Timeframe des données")
    
    # Paramètres de backtest
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--position-size", type=float, default=0.1, help="Taille de position (% du capital)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission (en %)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage (en %)")
    parser.add_argument("--use-sl-tp", action="store_true", help="Utiliser les Stop Loss et Take Profit")
    
    args = parser.parse_args()
    
    return run_backtest(args)

if __name__ == "__main__":
    sys.exit(main())
