# live/monitoring.py

import time
import logging
from typing import Dict, Any, Optional
# Utiliser Gauge principalement comme dans le guide
from prometheus_client import Gauge, start_http_server, Counter, Histogram

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Collecte et expose les métriques de performance et d'état simplifiées
    du système de trading live via Prometheus, basé sur le guide fourni.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le logger de métriques et démarre le serveur HTTP Prometheus.

        Args:
            config: Dictionnaire de configuration chargé depuis config.yaml.
                    Utilisé pour obtenir le port Prometheus.
        """
        self.config = config
        # Récupérer le port depuis la section live_trading comme dans config mise à jour
        live_config = self.config.get('live_trading', {})
        self.prometheus_port = live_config.get('prometheus_port', 8000) # Port par défaut
        self.symbol = live_config.get('symbol', 'UNKNOWN') # Récupérer le symbole pour les labels

        # --- Définition des Métriques Prometheus (selon le guide) ---
        self.latency_gauge = Gauge(
            'morningstar_ws_latency_ms', # Nom simplifié
            'WS round-trip latency in ms',
            ['symbol'] # Ajouter le label symbol
        )
        # Utiliser un Counter pour le succès des ordres semble plus logique qu'un Gauge
        self.order_success_counter = Counter(
            'morningstar_order_success_total', # Nom simplifié
            'Number of successful orders executed',
            ['symbol'] # Ajouter le label symbol
        )
        self.order_error_counter = Counter(
             'morningstar_order_error_total',
             'Number of failed orders',
             ['symbol', 'error_type']
        )
        self.pnl_gauge = Gauge(
            'morningstar_current_pnl', # Nom simplifié
            'Estimated profit & loss',
            ['symbol'] # Ajouter le label symbol
        )
        # Ajouter une jauge pour le statut de connexion WS, utile pour le monitoring
        self.websocket_connection_status = Gauge(
             'morningstar_websocket_connection_status',
             'Statut de la connexion WebSocket (1=connecté, 0=déconnecté)',
             ['symbol']
        )
        # Ajouter un compteur pour les tentatives de trade
        self.trade_attempts = Counter(
            'morningstar_trade_attempts_total',
            'Nombre total de tentatives de trade',
            ['symbol', 'side']
        )
        # Ajouter un compteur pour les prédictions
        self.predictions_received = Counter(
            'morningstar_predictions_received_total',
            'Nombre total de prédictions reçues du modèle',
            ['symbol']
        )
        # Ajouter un compteur pour les erreurs générales
        self.errors_logged = Counter(
            'morningstar_errors_logged_total',
            'Nombre total d\'erreurs système logguées',
            ['component', 'error_type']
        )
        # Ajouter un histogramme pour le temps de prédiction
        self.prediction_time = Histogram(
            'morningstar_prediction_time_seconds',
            'Temps pris pour obtenir une prédiction du modèle',
            ['symbol'],
            buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        )


        # Démarrer le serveur HTTP pour Prometheus
        try:
            start_http_server(self.prometheus_port)
            logger.info(f"Serveur Prometheus démarré sur le port {self.prometheus_port}")
            # Initialiser le statut de connexion à 0 (déconnecté)
            self.websocket_connection_status.labels(symbol=self.symbol).set(0)
        except OSError as e:
             logger.error(f"Impossible de démarrer le serveur Prometheus sur le port {self.prometheus_port}: {e}. Le port est peut-être déjà utilisé.")
        except Exception as e:
            logger.exception(f"Erreur inattendue lors du démarrage du serveur Prometheus: {e}")

    # --- Méthodes d'enregistrement (adaptées du guide et de la version précédente) ---

    def log_websocket_latency(self, event_timestamp_ms: Optional[float], symbol: str):
        """Enregistre la latence du WebSocket en ms et met à jour le statut."""
        if event_timestamp_ms:
            try:
                current_time_ms = time.time() * 1000
                latency_ms = current_time_ms - event_timestamp_ms
                if latency_ms >= 0:
                    self.latency_gauge.labels(symbol=symbol).set(latency_ms) # Enregistrer en ms
                    self.websocket_connection_status.labels(symbol=symbol).set(1) # Marquer comme connecté
                else:
                     logger.warning(f"Latence WebSocket négative détectée ({latency_ms}ms), ignorée.")
                     self.websocket_connection_status.labels(symbol=symbol).set(1) # Marquer connecté quand même
            except Exception as e:
                logger.warning(f"Impossible de calculer/logguer la latence WebSocket: {e}")
        else:
             # Si pas de timestamp, on ne peut pas calculer la latence, mais on est connecté
             self.websocket_connection_status.labels(symbol=symbol).set(1)

    def log_prediction(self, predictions: Dict[str, Any], symbol: str):
         """Enregistre la réception d'une prédiction."""
         self.predictions_received.labels(symbol=symbol).inc()

    def log_trade_attempt(self, side: str, symbol: str):
         """Enregistre une tentative de trade."""
         self.trade_attempts.labels(symbol=symbol, side=side).inc()

    def log_trade_result(self, success: bool, side: str, symbol: str, error_type: Optional[str] = None):
        """Enregistre le résultat d'une tentative de trade (succès ou échec)."""
        if success:
            self.order_success_counter.labels(symbol=symbol).inc()
        else:
            error_label = error_type if error_type else 'unknown'
            self.order_error_counter.labels(symbol=symbol, error_type=error_label).inc()

    def log_error(self, component: str, error_type: str = 'unknown'):
         """Enregistre une erreur système générale."""
         self.errors_logged.labels(component=component, error_type=error_type).inc()
         # Si l'erreur concerne le websocket, marquer comme déconnecté
         if component.startswith('websocket'):
              self.websocket_connection_status.labels(symbol=self.symbol).set(0)

    def update_pnl(self, pnl_value: float, symbol: str):
        """Met à jour la jauge P&L."""
        self.pnl_gauge.labels(symbol=symbol).set(pnl_value)

    # Ajouter la méthode pour le temps de prédiction si nécessaire
    # def log_prediction_time(self, time_seconds: float, symbol: str):
    #      self.prediction_time.labels(symbol=symbol).observe(time_seconds)