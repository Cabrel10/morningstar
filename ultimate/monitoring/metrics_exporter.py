#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exporteur de métriques pour le modèle monolithique Morningstar.

Ce script expose un endpoint HTTP pour les métriques au format Prometheus.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import threading
import logging
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from prometheus_client.core import CollectorRegistry, REGISTRY
import argparse

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metrics_exporter")

# Définition des métriques Prometheus
PREDICTIONS_COUNT = Counter('morningstar_predictions_count', 'Nombre total de prédictions')
TRADES_EXECUTED = Counter('morningstar_trades_executed', 'Nombre de trades exécutés')
TRADES_SUCCESSFUL = Counter('morningstar_trades_successful', 'Nombre de trades réussis')
TRADES_FAILED = Counter('morningstar_trades_failed', 'Nombre de trades échoués')

PREDICTION_LATENCY = Histogram('morningstar_prediction_latency_milliseconds', 
                              'Latence des prédictions en ms',
                              buckets=[10, 20, 50, 100, 200, 500, 1000])

EQUITY_PERCENTAGE = Gauge('morningstar_equity_percentage', 'Équité en pourcentage du capital initial')
EQUITY_VALUE = Gauge('morningstar_equity_value', 'Valeur actuelle de l\'équité')
DRAWDOWN = Gauge('morningstar_drawdown_percentage', 'Drawdown en pourcentage')
SL_HIT_RATE = Gauge('morningstar_sl_hit_rate', 'Taux de hit des stop loss (%)')
TP_HIT_RATE = Gauge('morningstar_tp_hit_rate', 'Taux de hit des take profit (%)')

API_ERRORS = Counter('morningstar_api_errors', 'Nombre d\'erreurs API', ['exchange', 'endpoint', 'error_type'])
MODEL_ERRORS = Counter('morningstar_model_errors', 'Nombre d\'erreurs du modèle', ['component', 'error_type'])

SIGNAL_COUNTS = Counter('morningstar_signals', 'Nombre de signaux générés', ['signal_type'])
POSITIONS = Gauge('morningstar_positions', 'Nombre de positions ouvertes', ['symbol', 'direction'])


class MetricsCollector:
    """Collecteur de métriques pour le modèle monolithique."""

    def __init__(self, metrics_dir: str, update_interval: int = 60):
        """
        Initialise le collecteur de métriques.
        
        Args:
            metrics_dir: Répertoire où sont stockées les métriques
            update_interval: Intervalle de mise à jour des métriques en secondes
        """
        self.metrics_dir = metrics_dir
        self.update_interval = update_interval
        self.last_equity = 100.0  # Valeur initiale supposée (100%)
        self.max_equity = 100.0   # Maximum historique
        self.trading_stats_file = os.path.join(metrics_dir, "trading_stats.json")
        self.performance_file = os.path.join(metrics_dir, "performance.json")
        self.positions_file = os.path.join(metrics_dir, "positions.json")
        
        # Créer le répertoire si nécessaire
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Démarrer le thread de collecte
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics)
        self.collection_thread.daemon = True
        self.collection_thread.start()
    
    def stop(self):
        """Arrête le collecteur de métriques."""
        self.running = False
        if self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
    
    def _collect_metrics(self):
        """Méthode principale de collecte des métriques."""
        while self.running:
            try:
                # Collecter et mettre à jour les métriques
                self._update_trading_stats()
                self._update_performance()
                self._update_positions()
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des métriques: {e}")
            
            # Attendre le prochain cycle
            time.sleep(self.update_interval)
    
    def _update_trading_stats(self):
        """Met à jour les métriques de trading."""
        if not os.path.exists(self.trading_stats_file):
            logger.warning(f"Fichier de statistiques non trouvé: {self.trading_stats_file}")
            return
        
        try:
            with open(self.trading_stats_file, 'r') as f:
                stats = json.load(f)
            
            # Mettre à jour les compteurs
            TRADES_EXECUTED._value.set(stats.get("trades_executed", 0))
            TRADES_SUCCESSFUL._value.set(stats.get("trades_successful", 0))
            TRADES_FAILED._value.set(stats.get("trades_failed", 0))
            PREDICTIONS_COUNT._value.set(stats.get("predictions_count", 0))
            
            # Mettre à jour les taux
            SL_HIT_RATE.set(stats.get("sl_hit_rate", 0.0))
            TP_HIT_RATE.set(stats.get("tp_hit_rate", 0.0))
            
            # Extraire et mettre à jour les compteurs de signaux
            for signal_type, count in stats.get("signals", {}).items():
                SIGNAL_COUNTS.labels(signal_type=signal_type)._value.set(count)
            
            logger.debug("Statistiques de trading mises à jour")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Erreur lors de la lecture des statistiques de trading: {e}")
    
    def _update_performance(self):
        """Met à jour les métriques de performance."""
        if not os.path.exists(self.performance_file):
            logger.warning(f"Fichier de performance non trouvé: {self.performance_file}")
            return
        
        try:
            with open(self.performance_file, 'r') as f:
                perf = json.load(f)
            
            # Extraire les métriques
            equity_pct = perf.get("equity_percentage", 100.0)
            equity_value = perf.get("equity_value", 0.0)
            
            # Mettre à jour l'équité
            EQUITY_PERCENTAGE.set(equity_pct)
            EQUITY_VALUE.set(equity_value)
            
            # Calculer et mettre à jour le drawdown
            self.max_equity = max(self.max_equity, equity_pct)
            current_drawdown = ((self.max_equity - equity_pct) / self.max_equity) * 100.0
            DRAWDOWN.set(current_drawdown)
            
            # Mettre à jour la latence des prédictions si disponible
            latency_data = perf.get("prediction_latency_ms", [])
            for latency in latency_data:
                PREDICTION_LATENCY.observe(latency)
            
            logger.debug("Métriques de performance mises à jour")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Erreur lors de la lecture des métriques de performance: {e}")
    
    def _update_positions(self):
        """Met à jour les métriques des positions ouvertes."""
        if not os.path.exists(self.positions_file):
            logger.warning(f"Fichier de positions non trouvé: {self.positions_file}")
            return
        
        try:
            with open(self.positions_file, 'r') as f:
                positions = json.load(f)
            
            # Réinitialiser les métriques de positions (pour supprimer les positions fermées)
            for symbol, direction in POSITIONS._metrics.keys():
                POSITIONS.labels(symbol=symbol, direction=direction).set(0)
            
            # Mettre à jour avec les positions actuelles
            for position in positions:
                symbol = position.get("symbol", "unknown")
                direction = position.get("direction", "neutral")
                size = position.get("size", 0.0)
                POSITIONS.labels(symbol=symbol, direction=direction).set(size)
            
            logger.debug("Métriques de positions mises à jour")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Erreur lors de la lecture des positions: {e}")


def record_prediction(latency_ms: float, is_success: bool = True):
    """
    Enregistre une prédiction effectuée par le modèle.
    
    Args:
        latency_ms: Latence de la prédiction en millisecondes
        is_success: Si la prédiction a réussi ou échoué
    """
    PREDICTIONS_COUNT.inc()
    PREDICTION_LATENCY.observe(latency_ms)
    
    if not is_success:
        MODEL_ERRORS.labels(component="prediction", error_type="inference_failure").inc()


def record_trade(symbol: str, direction: str, is_success: bool = True, sl_hit: bool = False, 
                tp_hit: bool = False):
    """
    Enregistre un trade exécuté.
    
    Args:
        symbol: Symbole du trade
        direction: Direction du trade (buy/sell)
        is_success: Si le trade a réussi ou échoué
        sl_hit: Si le stop loss a été atteint
        tp_hit: Si le take profit a été atteint
    """
    TRADES_EXECUTED.inc()
    
    if is_success:
        TRADES_SUCCESSFUL.inc()
        SIGNAL_COUNTS.labels(signal_type=f"{direction}_{symbol}").inc()
    else:
        TRADES_FAILED.inc()
    
    if sl_hit:
        SIGNAL_COUNTS.labels(signal_type="sl_hit").inc()
    
    if tp_hit:
        SIGNAL_COUNTS.labels(signal_type="tp_hit").inc()


def record_api_error(exchange: str, endpoint: str, error_type: str):
    """
    Enregistre une erreur d'API.
    
    Args:
        exchange: Nom de l'exchange
        endpoint: Endpoint de l'API
        error_type: Type d'erreur
    """
    API_ERRORS.labels(exchange=exchange, endpoint=endpoint, error_type=error_type).inc()


def record_model_error(component: str, error_type: str):
    """
    Enregistre une erreur du modèle.
    
    Args:
        component: Composant qui a généré l'erreur
        error_type: Type d'erreur
    """
    MODEL_ERRORS.labels(component=component, error_type=error_type).inc()


def start_metrics_server(port: int = 8888, metrics_dir: str = None, update_interval: int = 60):
    """
    Démarre le serveur d'exportation des métriques.
    
    Args:
        port: Port sur lequel exposer les métriques
        metrics_dir: Répertoire où sont stockées les métriques
        update_interval: Intervalle de mise à jour des métriques en secondes
    
    Returns:
        Le collecteur de métriques
    """
    # Démarrer le serveur HTTP pour les métriques
    start_http_server(port)
    logger.info(f"Serveur de métriques démarré sur le port {port}")
    
    # Créer et démarrer le collecteur si un répertoire est spécifié
    collector = None
    if metrics_dir:
        collector = MetricsCollector(metrics_dir, update_interval)
        logger.info(f"Collecteur de métriques démarré avec intervalle de {update_interval}s")
    
    return collector


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Exporter de métriques Prometheus pour Morningstar")
    parser.add_argument("--port", type=int, default=8888, help="Port sur lequel exposer les métriques")
    parser.add_argument("--metrics-dir", type=str, default="./ultimate/monitoring/metrics", 
                      help="Répertoire où sont stockées les métriques")
    parser.add_argument("--update-interval", type=int, default=60, 
                      help="Intervalle de mise à jour des métriques (secondes)")
    
    args = parser.parse_args()
    
    # Créer le répertoire des métriques s'il n'existe pas
    os.makedirs(args.metrics_dir, exist_ok=True)
    
    try:
        # Démarrer le serveur de métriques
        collector = start_metrics_server(
            port=args.port,
            metrics_dir=args.metrics_dir,
            update_interval=args.update_interval
        )
        
        # Continuer à exécuter tant que le programme n'est pas interrompu
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Arrêt du serveur de métriques")
        if collector:
            collector.stop()


if __name__ == "__main__":
    main() 