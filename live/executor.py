# live/executor.py

import os
import asyncio
import ccxt # Ajout pour le client REST
import ccxt.pro as ccxtpro
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple

# --- Imports du projet ---
from utils.helpers import load_config # Supposons qu'une telle fonction existe
from model.architecture.enhanced_hybrid_model import MorningstarModel # Assurez-vous que le nom est correct
# Renommer ou créer cette fonction/classe pour traiter les données live (OHLCV ici)
from utils.live_preprocessing import LiveDataPreprocessor # Ou une fonction preprocess_live_data
from live.monitoring import MetricsLogger

# Configuration du Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
RECONNECT_DELAY = 5
ORDER_EXECUTION_DELAY = 0.5

class LiveExecutor:
    """
    Orchestre le trading en direct: connexion exchange, réception données temps réel,
    prédiction modèle, exécution ordres et monitoring.
    """

    def __init__(self, config_path: str = 'config/config.yaml', exchange_id_override: Optional[str] = None, dry_run: bool = False):
        """
        Initialise l'exécuteur live.

        Args:
            config_path: Chemin vers le fichier de configuration principal.
            exchange_id_override: ID de l'exchange à utiliser, surcharge la config si fourni.
            dry_run: Si True, simule les ordres sans les exécuter réellement.
        """
        logger.info(f"Initialisation de LiveExecutor (Dry Run: {dry_run})...")
        self.dry_run = dry_run
        self.trading_active = True  # Nouvel attribut pour activer/désactiver le trading
        self.max_consecutive_errors = 5  # Nombre maximum d'erreurs consécutives avant pause
        self.consecutive_errors = 0  # Compteur d'erreurs consécutives
        self.error_pause_duration = 300  # Pause de 5 minutes après trop d'erreurs
        self.last_error_time = None  # Horodatage de la dernière erreur
        
        try:
            self.config = load_config(config_path)
            live_config = self.config.get('live_trading', {})
            exchange_params_config = self.config.get('exchange_params', {})
            model_config = self.config.get('model', {})
            data_pipeline_config = self.config.get('data_pipeline', {})

            # --- Configuration de base ---
            self.exchange_id = (exchange_id_override or live_config.get('default_exchange', 'binance')).lower()
            self.symbol = live_config.get('symbol', 'BTC/USDT')
            self.timeframe = live_config.get('timeframe', '1m')
            self.use_websocket = live_config.get('websocket', True)

            # --- Paramètres de Trading ---
            self.risk_per_trade_pct: float = live_config.get('risk_per_trade_pct', 0.01) # Ex: 1%
            self.atr_sl_multiplier: float = live_config.get('atr_sl_multiplier', 1.5)
            self.rr_ratio_tp: float = live_config.get('rr_ratio_tp', 2.0)

            logger.info(f"Configuration Live: Exchange={self.exchange_id}, Symbol={self.symbol}, Timeframe={self.timeframe}, WebSocket={self.use_websocket}")
            logger.info(f"Paramètres Trading: Risk={self.risk_per_trade_pct*100}%, SL Mult={self.atr_sl_multiplier}, TP Ratio={self.rr_ratio_tp}")

            # --- État de la Position ---
            self.current_position_size: float = 0.0
            self.entry_price: Optional[float] = None
            self.position_side: Optional[str] = None # 'long' ou None pour l'instant
            self.active_sl_order_id: Optional[str] = None
            self.active_tp_order_id: Optional[str] = None
            self.last_known_balance: Dict[str, Dict[str, float]] = {} # Ex: {'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0}}

            # --- Composants ---
            # Charger le modèle
            model_path = model_config.get('save_path', 'model/saved_model/morningstar_model')
            self.model = MorningstarModel.load_model(model_path) # Assumer que load_model est une méthode de classe
            logger.info(f"Modèle chargé depuis: {model_path}")

            # Initialiser le preprocessor
            preprocessor_window = data_pipeline_config.get('indicator_window_size', 100)
            self.preprocessor = LiveDataPreprocessor(window_size=preprocessor_window, symbol=self.symbol)

            # Initialiser le logger de métriques (passer la config entière)
            self.metrics = MetricsLogger(config=self.config) # Assumer que MetricsLogger existe et est importé

            # Initialiser les clients REST et WebSocket
            self.client: Optional[ccxt.Exchange] = None # Client REST (synchrone ou asynchrone)
            self.ws_client: Optional[ccxtpro.Exchange] = None # Client WebSocket
            # Note: _init_exchange_clients doit être appelé dans un contexte async si on utilise des clients async
            # Pour l'instant, on suppose qu'il est appelé avant la boucle async run()
            # Ou qu'il initialise des clients synchrones pour les opérations REST initiales comme fetch_balance
            self._init_exchange_clients() # Peut lever une exception
            logger.info(f"Clients CCXT initialisés pour: {self.exchange_id}")

            # Récupérer le solde initial (si pas en dry run)
            # Note: Cet appel pourrait être bloquant s'il est synchrone. Idéalement, le faire async.
            # Pour simplifier, on le laisse synchrone ici, mais à revoir si performance critique.
            if not self.dry_run:
            # On encapsule dans une méthode pour le rendre potentiellement async plus tard
                self._update_balance_sync() # Appel synchrone pour l'initialisation

        except FileNotFoundError:
            logger.error(f"Erreur: Fichier de configuration non trouvé à {config_path}")
            raise
        except KeyError as e:
            logger.error(f"Erreur: Clé manquante dans la configuration: {e}")
            raise
        except Exception as e:
            logger.exception(f"Erreur inattendue lors de l'initialisation de LiveExecutor: {e}")
            raise

    def _init_exchange_clients(self):
        """Initialise les instances ccxt (REST) et ccxt.pro (WebSocket) avec authentification."""
        logger.info(f"Configuration des clients ccxt/ccxt.pro pour {self.exchange_id}...")
        api_key_env = f'{self.exchange_id.upper()}_API_KEY'
        secret_env = f'{self.exchange_id.upper()}_API_SECRET'
        passphrase_env = f'{self.exchange_id.upper()}_PASSPHRASE'

        api_key = os.getenv(api_key_env)
        secret = os.getenv(secret_env)
        password = os.getenv(passphrase_env) # Sera None si non défini

        if not api_key or not secret:
            logger.error(f"Clés API ({api_key_env}, {secret_env}) non trouvées dans les variables d'environnement.")
            raise ValueError("Clés API manquantes.")

        # Paramètres communs extraits de la config
        common_params = self.config.get('exchange_params', {}).get(self.exchange_id, {})
        # S'assurer que enableRateLimit est bien dans les params si défini
        if 'enableRateLimit' not in common_params:
            common_params['enableRateLimit'] = True # Valeur par défaut sûre

        # Configuration spécifique pour les clients
        client_config = {
            'apiKey': api_key,
            'secret': secret,
            **common_params # Fusionne les paramètres de config.yaml
        }
        ws_client_config = {
            'apiKey': api_key,
            'secret': secret,
            **common_params
        }

        # Ajouter la passphrase si elle existe et est requise
        if password and self.exchange_id in ['kucoin', 'bitget']:
            client_config['password'] = password
            ws_client_config['password'] = password
        elif not password and self.exchange_id in ['kucoin', 'bitget']:
            logger.error(f"Passphrase ({passphrase_env}) requise mais non trouvée pour {self.exchange_id}.")
            raise ValueError("Passphrase manquante.")

        # Instanciation
        try:
            rest_cls = getattr(ccxt, self.exchange_id)
            self.client = rest_cls(client_config)
            logger.info(f"Client REST ccxt pour {self.exchange_id} créé.")

            if self.use_websocket:
                ws_cls = getattr(ccxtpro, self.exchange_id)
                self.ws_client = ws_cls(ws_client_config)
                logger.info(f"Client WebSocket ccxt.pro pour {self.exchange_id} créé.")
            else:
                logger.warning("WebSocket désactivé dans la configuration.")

        except AttributeError as e:
            logger.error(f"Exchange '{self.exchange_id}' non supporté par ccxt ou ccxt.pro: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des clients ccxt: {e}")
            raise

    def _update_balance_sync(self):
        """Récupère et met à jour le solde du compte de manière synchrone."""
        if self.dry_run:
            logger.info("[DRY RUN] Mise à jour du solde ignorée.")
            # Simuler un solde pour le dry run si nécessaire
            quote_currency = self.symbol.split('/')[1]
            if not self.last_known_balance:
                self.last_known_balance = {quote_currency: {'free': 1000.0, 'used': 0.0, 'total': 1000.0}}
            return

        if not self.client:
            logger.error("Client CCXT non initialisé, impossible de récupérer le solde.")
            return

        try:
            logger.info("Récupération du solde du compte...")
            balance = self.client.fetch_balance()
            # Garder uniquement les informations 'free', 'used', 'total'
            self.last_known_balance = { 
                currency: {
                    'free': data.get('free'), 
                    'used': data.get('used'), 
                    'total': data.get('total')
                } 
                for currency, data in balance.items() if isinstance(data, dict)
            }
            quote_currency = self.symbol.split('/')[1]
            logger.info(f"Solde mis à jour. Disponible {quote_currency}: {self.last_known_balance.get(quote_currency, {}).get('free', 'N/A')}")
        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau lors de la récupération du solde: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur d'exchange lors de la récupération du solde: {e}")
        except Exception as e:
            logger.exception(f"Erreur inattendue lors de la récupération du solde: {e}")

    def _calculate_trade_details(self, current_price: float, atr: float) -> Optional[Dict[str, float]]:
        """
        Calcule les détails d'un trade potentiel (taille, SL, TP).

        Args:
            current_price: Le prix actuel du marché.
            atr: La valeur actuelle de l'Average True Range.

        Returns:
            Un dictionnaire avec 'sl_price', 'tp_price', 'order_size', ou None si le calcul échoue.
        """
        if atr <= 1e-8: # Éviter division par zéro ou ATR invalide
            logger.warning(f"ATR invalide ou nul ({atr}), impossible de calculer les détails du trade.")
            return None

        quote_currency = self.symbol.split('/')[1]
        # Utiliser le solde 'free' pour le calcul
        available_balance = self.last_known_balance.get(quote_currency, {}).get('free')

        if available_balance is None or available_balance <= 0:
            logger.warning(f"Solde disponible insuffisant ou inconnu en {quote_currency} ({available_balance}), impossible de calculer la taille de l'ordre.")
            return None

        # Calcul SL
        sl_distance_points = atr * self.atr_sl_multiplier
        sl_price = current_price - sl_distance_points

        # Calcul TP
        tp_distance_points = sl_distance_points * self.rr_ratio_tp
        tp_price = current_price + tp_distance_points

        # Calcul Taille Ordre
        risk_amount_quote = available_balance * self.risk_per_trade_pct
        # La distance SL est la différence entre prix actuel et SL
        sl_distance_for_size = current_price - sl_price # = sl_distance_points
        if sl_distance_for_size < 1e-8:
            logger.warning(f"Distance SL trop petite ({sl_distance_for_size}), impossible de calculer la taille de l'ordre.")
            return None
        
        order_size_base = risk_amount_quote / sl_distance_for_size

        # TODO: Vérifier les limites de l'exchange (taille min/max, précision)
        # market_info = self.client.market(self.symbol) if self.client else None
        # if market_info:
        #     min_size = market_info.get('limits', {}).get('amount', {}).get('min')
        #     max_size = market_info.get('limits', {}).get('amount', {}).get('max')
        #     precision = market_info.get('precision', {}).get('amount')
        #     if min_size and order_size_base < min_size:
        #         logger.warning(f"Taille calculée {order_size_base} inférieure au min {min_size}. Ajustement nécessaire ou annulation.")
        #         # return None ou ajuster
        #     # ... autres vérifications ...

        logger.debug(f"Détails Trade Calculés: Taille={order_size_base:.6f}, SL={sl_price:.4f}, TP={tp_price:.4f}")
        return {
            'sl_price': sl_price,
            'tp_price': tp_price,
            'order_size': order_size_base
        }

    async def _handle_signal(self, prediction: Dict, current_price: float, atr: float):
        """
        Gère un signal de trading: décide d'entrer, sortir ou ne rien faire,
        calcule la taille, place les ordres (entrée, SL, TP) et met à jour l'état.
        Logique actuelle: Long/Flat uniquement.
        """
        # Assurer que le client REST est disponible (et potentiellement asynchrone)
        # Note: Idéalement, utiliser un client REST asynchrone si disponible pour ccxt
        # Pour l'instant, on suppose que self.client peut gérer les appels await (via aiohttp par ex)
        if not self.client:
            logger.error("Client REST non disponible pour gérer le signal.")
            return

        # Extraire le signal (adapter si la sortie du modèle est différente)
        # Supposons 0=Hold, 1=Buy, 2=Sell
        signal = prediction.get('signal')
        if signal is None:
            logger.warning("Signal non trouvé dans la prédiction.")
            return

        signal_label = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(signal, f"UNKNOWN({signal})")
        logger.debug(f"Signal reçu: {signal_label}, État actuel: {self.position_side}, Prix: {current_price:.4f}, ATR: {atr:.4f}")

        # --- Logique d'Entrée (Achat si Flat) ---
        if signal == 1 and self.position_side is None:
            trade_details = self._calculate_trade_details(current_price, atr)
            if not trade_details:
                logger.warning("Impossible de calculer les détails du trade pour l'entrée.")
                return # Ne pas trader si les détails ne sont pas valides

            order_size = trade_details['order_size']
            sl_price = trade_details['sl_price']
            tp_price = trade_details['tp_price']

            # Vérification supplémentaire de sécurité (ex: taille > 0)
            if order_size <= 0:
                logger.warning(f"Taille d'ordre calculée invalide ({order_size}). Annulation de l'entrée.")
                return

            logger.info(f"[{'DRY RUN' if self.dry_run else 'LIVE'}] Signal BUY détecté (état Flat). Tentative d'entrée:")
            logger.info(f"  Taille={order_size:.6f}, SL={sl_price:.4f}, TP={tp_price:.4f}")
            self.metrics.log_trade_attempt(side='buy', symbol=self.symbol, amount=order_size)

            entry_order_info = None
            sl_order_id = None
            tp_order_id = None
            entry_success = False

            try:
                # 1. Passer Ordre d'Entrée MARKET BUY
                if not self.dry_run:
                    # Utiliser await car on s'attend à un client asynchrone
                    entry_order = await self.client.create_market_buy_order(self.symbol, order_size)
                    entry_order_info = entry_order # Garder l'info complète
                    # Utiliser le prix moyen d'exécution si disponible, sinon le prix demandé
                    filled_price = entry_order.get('average') or entry_order.get('price') or current_price
                    filled_amount = entry_order.get('filled') or order_size # Utiliser la taille remplie si disponible
                    logger.info(f"Ordre d'entrée BUY exécuté: ID={entry_order['id']}, Prix Rempli={filled_price:.4f}, Taille Remplie={filled_amount:.6f}")
                    self.entry_price = float(filled_price)
                    self.current_position_size = float(filled_amount) # Utiliser la taille réelle
                else:
                    self.entry_price = current_price # Simuler prix d'entrée
                    self.current_position_size = order_size
                    logger.info(f"[DRY RUN] Ordre MARKET BUY simulé. Taille={order_size:.6f}, Prix Entrée={self.entry_price:.4f}")

                entry_success = True # Marquer l'entrée comme réussie (ou simulée)

                # 2. Placer Ordre STOP_MARKET SELL (SL) - Uniquement si entrée réussie
                # Utiliser la taille réellement exécutée (self.current_position_size)
                if not self.dry_run:
                    sl_order = await self.client.create_order(
                        self.symbol, 'stop_market', 'sell', self.current_position_size,
                        params={'stopPrice': sl_price, 'reduceOnly': True} # reduceOnly est crucial si supporté
                    )
                    sl_order_id = sl_order['id']
                    logger.info(f"Ordre SL placé: ID={sl_order_id}, Trigger={sl_price:.4f}")
                else:
                    sl_order_id = f"dry_sl_{int(time.time())}"
                    logger.info(f"[DRY RUN] Ordre STOP_MARKET SELL (SL) simulé à {sl_price:.4f}, ID={sl_order_id}")
                self.active_sl_order_id = sl_order_id

                # 3. Placer Ordre TAKE_PROFIT_MARKET SELL (TP) - Uniquement si SL réussi
                # Utiliser la taille réellement exécutée (self.current_position_size)
                if not self.dry_run:
                    tp_order = await self.client.create_order(
                        self.symbol, 'take_profit_market', 'sell', self.current_position_size,
                        params={'stopPrice': tp_price, 'reduceOnly': True} # reduceOnly est crucial
                    )
                    tp_order_id = tp_order['id']
                    logger.info(f"Ordre TP placé: ID={tp_order_id}, Trigger={tp_price:.4f}")
                else:
                    tp_order_id = f"dry_tp_{int(time.time())}"
                    logger.info(f"[DRY RUN] Ordre TAKE_PROFIT_MARKET SELL (TP) simulé à {tp_price:.4f}, ID={tp_order_id}")
                self.active_tp_order_id = tp_order_id

                # 4. Mettre à jour l'état final et logguer succès
                self.position_side = 'long'
                self.metrics.log_trade_result(success=True, side='buy', symbol=self.symbol, order_info=entry_order_info)
                self.metrics.update_position_size(self.current_position_size, symbol=self.symbol)
                logger.info(f"Position LONG ouverte avec succès. Taille: {self.current_position_size:.6f}, Entrée: {self.entry_price:.4f}")

            except ccxt.InsufficientFunds as e:
                logger.error(f"Fonds insuffisants pour l'ordre d'entrée BUY: {e}")
                self.metrics.log_trade_result(success=False, side='buy', symbol=self.symbol, error_type='insufficient_funds')
                # Pas besoin de compensation car rien n'a été placé
            except ccxt.NetworkError as e:
                logger.error(f"Erreur réseau lors de l'entrée en position: {e}")
                self.metrics.log_trade_result(success=False, side='buy', symbol=self.symbol, error_type='network_error')
                # Gérer la compensation si l'erreur survient APRES l'ordre d'entrée mais AVANT SL/TP
                await self._compensate_failed_entry(entry_success, sl_order_id, tp_order_id)
            except ccxt.ExchangeError as e:
                logger.error(f"Erreur d'exchange lors de l'entrée en position: {e}")
                self.metrics.log_trade_result(success=False, side='buy', symbol=self.symbol, error_type='exchange_error')
                await self._compensate_failed_entry(entry_success, sl_order_id, tp_order_id)
            except Exception as e:
                logger.exception(f"Erreur inattendue lors de l'entrée en position longue: {e}")
                self.metrics.log_trade_result(success=False, side='buy', symbol=self.symbol, error_type='unexpected')
                await self._compensate_failed_entry(entry_success, sl_order_id, tp_order_id)

        # --- Logique de Sortie (Vente si Long) ---
        elif signal == 2 and self.position_side == 'long':
            logger.info(f"[{'DRY RUN' if self.dry_run else 'LIVE'}] Signal SELL détecté (état Long). Tentative de clôture:")
            self.metrics.log_trade_attempt(side='sell', symbol=self.symbol, amount=self.current_position_size)

            original_sl_id = self.active_sl_order_id
            original_tp_id = self.active_tp_order_id
            close_order_info = None
            close_success = False

            try:
                # 1. Annuler Ordre SL Actif
                if original_sl_id:
                    logger.debug(f"Tentative d'annulation de l'ordre SL: {original_sl_id}")
                    if not self.dry_run:
                        try:
                            await self.client.cancel_order(original_sl_id, self.symbol)
                            logger.info(f"Ordre SL {original_sl_id} annulé avec succès.")
                        except ccxt.OrderNotFound:
                            logger.warning(f"Ordre SL {original_sl_id} déjà clôturé ou introuvable.")
                        except Exception as cancel_e:
                            logger.error(f"Échec de l'annulation de l'ordre SL {original_sl_id}: {cancel_e}")
                            # Continuer quand même pour tenter de clôturer la position
                    else:
                        logger.info(f"[DRY RUN] Annulation Ordre SL {original_sl_id} simulée.")
                    self.active_sl_order_id = None # Marquer comme annulé même si échec API

                # 2. Annuler Ordre TP Actif
                if original_tp_id:
                    logger.debug(f"Tentative d'annulation de l'ordre TP: {original_tp_id}")
                    if not self.dry_run:
                        try:
                            await self.client.cancel_order(original_tp_id, self.symbol)
                            logger.info(f"Ordre TP {original_tp_id} annulé avec succès.")
                        except ccxt.OrderNotFound:
                            logger.warning(f"Ordre TP {original_tp_id} déjà clôturé ou introuvable.")
                        except Exception as cancel_e:
                            logger.error(f"Échec de l'annulation de l'ordre TP {original_tp_id}: {cancel_e}")
                    else:
                        logger.info(f"[DRY RUN] Annulation Ordre TP {original_tp_id} simulée.")
                    self.active_tp_order_id = None # Marquer comme annulé

                # 3. Passer Ordre de Clôture MARKET SELL
                logger.info(f"Passage de l'ordre MARKET SELL pour clôturer {self.current_position_size:.6f} {self.symbol.split('/')[0]}")
                if not self.dry_run:
                    close_order = await self.client.create_market_sell_order(self.symbol, self.current_position_size)
                    close_order_info = close_order
                    close_price = close_order.get('average') or close_order.get('price') or current_price
                    logger.info(f"Ordre de clôture SELL exécuté: ID={close_order['id']}, Prix={close_price:.4f}")
                    # Calcul P&L
                    pnl = (close_price - self.entry_price) * self.current_position_size if self.entry_price else 0
                    self.metrics.update_pnl(pnl, symbol=self.symbol)
                    logger.info(f"P&L réalisé: {pnl:.4f} {self.symbol.split('/')[1]}")
                else:
                    close_price = current_price
                    pnl = (close_price - self.entry_price) * self.current_position_size if self.entry_price else 0
                    logger.info(f"[DRY RUN] Ordre MARKET SELL (Clôture) simulé. Prix={close_price:.4f}, P&L={pnl:.4f}")
                    self.metrics.update_pnl(pnl, symbol=self.symbol) # Log P&L simulé

                close_success = True

                # 4. Réinitialiser l'état final
                self.position_side = None
                self.entry_price = None
                self.current_position_size = 0.0
                self.active_sl_order_id = None # Assurer qu'ils sont bien None
                self.active_tp_order_id = None
                self.metrics.log_trade_result(success=True, side='sell', symbol=self.symbol, order_info=close_order_info)
                self.metrics.update_position_size(0.0, symbol=self.symbol)
                logger.info("Position LONG clôturée avec succès.")

            except ccxt.InsufficientFunds as e: # Peut arriver si la position a déjà été réduite/clôturée
                logger.error(f"Fonds insuffisants pour l'ordre de clôture SELL (position peut-être déjà clôturée?): {e}")
                self.metrics.log_trade_result(success=False, side='sell', symbol=self.symbol, error_type='insufficient_funds')
                # Tenter de réinitialiser l'état si on pense que la position est fermée
                self._reset_position_state()
            except ccxt.NetworkError as e:
                logger.error(f"Erreur réseau lors de la clôture de position: {e}")
                self.metrics.log_trade_result(success=False, side='sell', symbol=self.symbol, error_type='network_error')
                # L'état est incertain ici, une vérification manuelle ou périodique peut être nécessaire
            except ccxt.ExchangeError as e:
                logger.error(f"Erreur d'exchange lors de la clôture de position: {e}")
                self.metrics.log_trade_result(success=False, side='sell', symbol=self.symbol, error_type='exchange_error')
                # État incertain
            except Exception as e:
                logger.exception(f"Erreur inattendue lors de la clôture de la position longue: {e}")
                self.metrics.log_trade_result(success=False, side='sell', symbol=self.symbol, error_type='unexpected')
                # État incertain

        # else: # Autres cas (Hold, Buy si Long, Sell si Flat) -> Ignorer
        #     logger.debug(f"Signal {signal_label} ignoré dans l'état actuel {self.position_side}.")

    async def _compensate_failed_entry(self, entry_success: bool, sl_order_id: Optional[str], tp_order_id: Optional[str]):
        """Tente d'annuler les ordres SL/TP si l'entrée a échoué après leur placement."""
        if entry_success and (sl_order_id or tp_order_id): # Si l'entrée a réussi mais une erreur est survenue après
            logger.warning("Erreur après l'ordre d'entrée. Tentative d'annulation des ordres SL/TP placés.")
            if sl_order_id and not self.dry_run:
                try: await self.client.cancel_order(sl_order_id, self.symbol)
                except Exception as e: logger.error(f"Échec annulation SL ({sl_order_id}) après entrée échouée: {e}")
            if tp_order_id and not self.dry_run:
                try: await self.client.cancel_order(tp_order_id, self.symbol)
                except Exception as e: logger.error(f"Échec annulation TP ({tp_order_id}) après entrée échouée: {e}")
            # Réinitialiser l'état car l'entrée est considérée comme échouée globalement
            self._reset_position_state()

    def _reset_position_state(self):
        """Réinitialise l'état de la position à 'flat'."""
        logger.warning("Réinitialisation de l'état de la position à FLAT.")
        self.position_side = None
        self.entry_price = None
        self.current_position_size = 0.0
        self.active_sl_order_id = None
        self.active_tp_order_id = None
        self.metrics.update_position_size(0.0, symbol=self.symbol)


    async def run(self):
        """Boucle principale pour écouter les données WebSocket et exécuter le trading."""
        logger.info("Démarrage de la boucle principale de trading...")
        self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(0) # Déconnecté au départ
        
        while True:
            try:
                # Vérifier si nous sommes en pause après trop d'erreurs
                if self.consecutive_errors >= self.max_consecutive_errors:
                    if self.last_error_time is None or (time.time() - self.last_error_time) < self.error_pause_duration:
                        logger.warning(f"Trop d'erreurs consécutives ({self.consecutive_errors}), pause de {self.error_pause_duration/60} minutes")
                        await asyncio.sleep(30)  # Vérifier toutes les 30 secondes
                        continue
                    else:
                        # Réinitialiser le compteur d'erreurs après la pause
                        logger.info("Reprise après pause d'erreur")
                        self.consecutive_errors = 0
                
                # Initialiser le client WebSocket si nécessaire
                if not hasattr(self, 'ws_client') or self.ws_client is None:
                    logger.info(f"Initialisation du client WebSocket pour {self.exchange_id}...")
                    exchange_class = getattr(ccxtpro, self.exchange_id)
                    self.ws_client = exchange_class({
                        'apiKey': self.config.get('exchange_params', {}).get('api_key'),
                        'secret': self.config.get('exchange_params', {}).get('api_secret'),
                        'enableRateLimit': True
                    })
                
                # Mise à jour du solde au démarrage
                self._update_balance_sync()
                
                logger.info(f"Démarrage de l'écoute des données OHLCV pour {self.symbol}...")
                
                # Boucle principale d'écoute WebSocket
                while True:
                    # Récupérer les données OHLCV via WebSocket
                    ohlcv = await self.ws_client.watch_ohlcv(self.symbol, self.timeframe)
                    
                    # Marquer la connexion comme active
                    self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(1)
                    
                    # Calculer et logger la latence
                    if 'timestamp' in ohlcv[-1]:
                        self.metrics.log_websocket_latency(ohlcv[-1]['timestamp'], symbol=self.symbol)
                    
                    # Convertir en DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Ajouter l'ATR pour le calcul des SL/TP
                    atr_period = 14
                    if len(df) >= atr_period:
                        tr1 = df['high'] - df['low']
                        tr2 = abs(df['high'] - df['close'].shift(1))
                        tr3 = abs(df['low'] - df['close'].shift(1))
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        df['atr'] = tr.rolling(window=atr_period).mean()
                        last_atr = df['atr'].iloc[-1]
                        current_price = df['close'].iloc[-1]
                        
                        # Préparation des données pour le modèle
                        try:
                            # Prétraiter les données pour le modèle
                            model_input = self.preprocessor.preprocess_live_data(df)
                            
                            # Si le prétraitement a réussi, faire une prédiction
                            if model_input is not None:
                                # Faire une prédiction avec le modèle
                                raw_predictions = self.model.predict(model_input)
                                
                                # Traiter les prédictions pour obtenir un signal final
                                # Exemple simplifié - à adapter selon votre modèle
                                if 'signal' in raw_predictions:
                                    signal_probs = raw_predictions['signal'][0]
                                    final_signal = np.argmax(signal_probs)
                                else:
                                    final_signal = 0

                                formatted_predictions = {'signal': final_signal}
                                
                                # Logguer la prédiction formatée
                                self.metrics.log_prediction(formatted_predictions, symbol=self.symbol)
                                logger.debug(f"Prédiction traitée: {formatted_predictions}, Prix Actuel: {current_price:.4f}, ATR: {last_atr:.4f}")
                                
                                # Vérifier si le trading est actif avant d'exécuter les ordres
                                if self.trading_active:
                                    # Appeler la fonction pour gérer le signal
                                    await self._handle_signal(formatted_predictions, current_price, last_atr)
                                else:
                                    logger.info(f"Trading désactivé, signal ignoré: {formatted_predictions}")
                                
                                # Réinitialiser le compteur d'erreurs après un cycle réussi
                                self.consecutive_errors = 0
                            
                        except Exception as e:
                            logger.exception(f"Erreur lors de la prédiction ou de la gestion du signal: {e}")
                            self.metrics.log_error(component='prediction_execution', error_type=type(e).__name__)
                            self.consecutive_errors += 1
                            self.last_error_time = time.time()

            except ccxtpro.NetworkError as e:
                logger.warning(f"Erreur réseau WebSocket: {e}. Tentative de reconnexion dans {RECONNECT_DELAY}s...")
                self.metrics.log_error(component='websocket', error_type='network_error')
                self.consecutive_errors += 1
                self.last_error_time = time.time()
            except ccxtpro.ExchangeError as e:
                logger.error(f"Erreur d'exchange WebSocket: {e}. Tentative de reconnexion dans {RECONNECT_DELAY}s...")
                self.metrics.log_error(component='websocket', error_type='exchange_error')
                self.consecutive_errors += 1
                self.last_error_time = time.time()
            except asyncio.CancelledError:
                logger.info("Boucle de trading annulée.")
                break # Sortir de la boucle while
            except Exception as e:
                logger.exception(f"Erreur inattendue dans la boucle principale: {e}. Tentative de reconnexion dans {RECONNECT_DELAY}s...")
                self.metrics.log_error(component='main_loop', error_type='unexpected')
                self.consecutive_errors += 1
                self.last_error_time = time.time()

            # Attendre avant de retenter la connexion/boucle en cas d'erreur
            logger.info(f"Attente de {RECONNECT_DELAY}s avant reconnexion...")
            await asyncio.sleep(RECONNECT_DELAY)
            # Fermer l'ancienne connexion WS avant de retenter (important)
            await self.close_ws_client()
            # Réinitialiser les clients peut être plus sûr
            logger.info("Tentative de réinitialisation des clients CCXT...")
            try:
                self._init_exchange_clients()
            except Exception as init_e:
                logger.error(f"Échec de la réinitialisation des clients après erreur: {init_e}. Nouvel essai dans {RECONNECT_DELAY}s.")
                await asyncio.sleep(RECONNECT_DELAY) # Attente supplémentaire si réinit échoue

    async def close_ws_client(self):
        """Ferme proprement la connexion WebSocket."""
        if hasattr(self, 'ws_client') and self.ws_client:
            logger.info("Fermeture de la connexion WebSocket ccxt.pro...")
            try:
                await self.ws_client.close()
                logger.info("Connexion WebSocket ccxt.pro fermée.")
                self.metrics.websocket_connection_status.labels(symbol=self.symbol).set(0) # Marquer déconnecté
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture de la connexion WebSocket: {e}")
        self.ws_client = None # S'assurer qu'il est None après fermeture

    async def close(self):
        """Ferme proprement toutes les connexions."""
        await self.close_ws_client()
        # Le client REST ccxt standard n'a pas de méthode close() asynchrone ou synchrone standard.
        # La fermeture des sessions HTTP sous-jacentes est généralement gérée par la lib (requests/aiohttp).
        logger.info("Client REST ccxt ne nécessite pas de fermeture explicite.")
        self.client = None
        
    def activate_trading(self):
        """Active l'exécution des ordres de trading."""
        if not self.trading_active:
            self.trading_active = True
            logger.info("Trading activé - les signaux seront exécutés")
            return True
        return False
        
    def deactivate_trading(self):
        """Désactive l'exécution des ordres de trading."""
        if self.trading_active:
            self.trading_active = False
            logger.info("Trading désactivé - les signaux seront ignorés")
            return True
        return False
        
    def get_trading_status(self):
        """Retourne le statut actuel du trading."""
        return {
            "active": self.trading_active,
            "consecutive_errors": self.consecutive_errors,
            "max_consecutive_errors": self.max_consecutive_errors,
            "error_pause_active": self.consecutive_errors >= self.max_consecutive_errors,
            "last_error_time": self.last_error_time
        }