#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de trading en direct pour l'application Streamlit Morningstar.

Ce module permet d'exécuter des stratégies de trading en temps réel
en utilisant les modèles entraînés et les API de trading.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import os
import sys
import subprocess
import threading
import ccxt
import pickle

# Ajouter le répertoire parent au path pour les importations absolues
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from app_modules.utils import (
    BASE_DIR, DATA_DIR, MODEL_DIR, REPORTS_DIR,
    get_available_models, load_dataset, plot_price_chart,
    format_number, create_metric_card
)
from live.exchange_manager import ExchangeManager
from config.api_keys import EXCHANGE_API_KEYS

# Liste des paires de crypto-monnaies disponibles
AVAILABLE_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT", "LINK/USDT", 
    "DOGE/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "BCH/USDT"
]

# Liste des échanges disponibles
AVAILABLE_EXCHANGES = [
    "binance", "bitget", "kucoin", "bybit", "huobi", "okx", "bitfinex", "kraken", "coinbase"
]

# Modes de trading
TRADING_MODES = [
    "Simulation", "Papier", "Réel"
]

# Créer un répertoire pour les checkpoints si nécessaire
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

class RecoveryManager:
    """
    Classe pour gérer la sauvegarde et la récupération de l'état du bot de trading.
    """
    def __init__(self, trading_bot, checkpoint_dir=CHECKPOINT_DIR, max_checkpoints=5):
        """
        Initialise le gestionnaire de récupération.
        
        Args:
            trading_bot: Instance du bot de trading à sauvegarder
            checkpoint_dir (Path): Répertoire où stocker les checkpoints
            max_checkpoints (int): Nombre maximum de checkpoints à conserver
        """
        self.trading_bot = trading_bot
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoint_interval = 300  # 5 minutes par défaut
        self.last_checkpoint_time = time.time()
        
        # Créer le répertoire des checkpoints s'il n'existe pas
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    def save_checkpoint(self, force=False):
        """
        Sauvegarde l'état actuel du bot de trading.
        
        Args:
            force (bool): Si True, force la sauvegarde même si l'intervalle n'est pas écoulé
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        # Vérifier si l'intervalle de temps est écoulé ou si la sauvegarde est forcée
        current_time = time.time()
        if not force and (current_time - self.last_checkpoint_time) < self.checkpoint_interval:
            return False
        
        try:
            # Créer un dictionnaire avec l'état du bot
            state = {
                'exchange_id': self.trading_bot.exchange_id,
                'api_key': self.trading_bot.api_key,
                'api_secret': self.trading_bot.api_secret,
                'model_path': self.trading_bot.model_path,
                'symbols': self.trading_bot.symbols,
                'mode': self.trading_bot.mode,
                'balance': self.trading_bot.balance,
                'positions': self.trading_bot.positions,
                'orders': self.trading_bot.orders,
                'trades': self.trading_bot.trades,
                'performance': self.trading_bot.performance,
                'trading_active': self.trading_bot.trading_active,
                'error_count': self.trading_bot.error_count,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Générer un nom de fichier pour le checkpoint
            checkpoint_file = self.checkpoint_dir / f"trading_bot_checkpoint_{int(current_time)}.pkl"
            
            # Sauvegarder l'état dans un fichier
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Mettre à jour le temps de la dernière sauvegarde
            self.last_checkpoint_time = current_time
            
            # Nettoyer les anciens checkpoints
            self._cleanup_old_checkpoints()
            
            print(f"Checkpoint sauvegardé: {checkpoint_file}")
            return True
        
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du checkpoint: {str(e)}")
            return False
    
    def load_latest_checkpoint(self):
        """
        Charge le dernier checkpoint sauvegardé.
        
        Returns:
            dict: État du bot chargé depuis le checkpoint, ou None si aucun checkpoint n'est trouvé
        """
        try:
            # Trouver tous les fichiers de checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("trading_bot_checkpoint_*.pkl"))
            
            if not checkpoint_files:
                print("Aucun checkpoint trouvé.")
                return None
            
            # Trier les fichiers par date de modification (le plus récent en premier)
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            # Charger le checkpoint le plus récent
            latest_checkpoint = checkpoint_files[0]
            
            with open(latest_checkpoint, 'rb') as f:
                state = pickle.load(f)
            
            print(f"Checkpoint chargé: {latest_checkpoint}")
            return state
        
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint: {str(e)}")
            return None
    
    def restore_trading_bot(self):
        """
        Restaure l'état du bot de trading à partir du dernier checkpoint.
        
        Returns:
            bool: True si la restauration a réussi, False sinon
        """
        state = self.load_latest_checkpoint()
        
        if not state:
            return False
        
        try:
            # Restaurer l'état du bot
            self.trading_bot.exchange_id = state['exchange_id']
            self.trading_bot.api_key = state['api_key']
            self.trading_bot.api_secret = state['api_secret']
            self.trading_bot.model_path = state['model_path']
            self.trading_bot.symbols = state['symbols']
            self.trading_bot.mode = state['mode']
            self.trading_bot.balance = state['balance']
            self.trading_bot.positions = state['positions']
            self.trading_bot.orders = state['orders']
            self.trading_bot.trades = state['trades']
            self.trading_bot.performance = state['performance']
            self.trading_bot.trading_active = state['trading_active']
            self.trading_bot.error_count = state['error_count']
            
            # Réinitialiser l'échange si nécessaire
            self.trading_bot._init_exchange()
            
            print(f"Bot de trading restauré à partir du checkpoint du {state['timestamp']}")
            return True
        
        except Exception as e:
            print(f"Erreur lors de la restauration du bot: {str(e)}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """
        Supprime les checkpoints les plus anciens si le nombre maximum est dépassé.
        """
        try:
            # Trouver tous les fichiers de checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("trading_bot_checkpoint_*.pkl"))
            
            if len(checkpoint_files) <= self.max_checkpoints:
                return
            
            # Trier les fichiers par date de modification (le plus ancien en premier)
            checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
            
            # Supprimer les fichiers les plus anciens
            for i in range(len(checkpoint_files) - self.max_checkpoints):
                os.remove(checkpoint_files[i])
                print(f"Ancien checkpoint supprimé: {checkpoint_files[i]}")
        
        except Exception as e:
            print(f"Erreur lors du nettoyage des checkpoints: {str(e)}")

class TradingBot:
    """
    Classe pour gérer le trading en direct.
    """
    
    def __init__(self, exchange_id, api_key=None, api_secret=None, model_path=None, symbols=None, mode="Simulation", use_testnet=False):
        """
        Initialise le bot de trading.
        
        Args:
            exchange_id (str): ID de l'échange (ex: binance, bitget)
            api_key (str, optional): Clé API pour l'échange
            api_secret (str, optional): Secret API pour l'échange
            model_path (str, optional): Chemin vers le modèle à utiliser
            symbols (list, optional): Liste des symboles à trader
            mode (str, optional): Mode de trading (Simulation, Papier, Réel)
            use_testnet (bool, optional): Utiliser le testnet au lieu du mainnet
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.model_path = model_path
        self.symbols = symbols or ["BTC/USDT"]
        self.mode = mode
        self.use_testnet = use_testnet
        
        # État du bot
        self.trading_active = False
        self.thread = None
        self.stop_event = threading.Event()
        self.model = None
        self.scaler = None
        self.exchange = None
        self.exchange_manager = None
        self.balance = {}
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.performance = {}
        self.last_update_time = time.time()
        self.update_interval = 60  # Intervalle de mise à jour en secondes
        self.error_count = 0
        self.max_errors = 5  # Nombre maximum d'erreurs avant pause
        self.recovery_manager = RecoveryManager(self)
        
        # Initialiser l'échange
        self._init_exchange()
        
        # Charger le modèle
        if model_path:
            self._load_model()
    
    def _init_exchange(self):
        """
        Initialise la connexion à l'échange.
        """
        try:
            if self.mode == "Simulation":
                # En mode simulation, pas besoin de connexion réelle
                self.exchange = None
                self.exchange_manager = None
                self.balance = {"USDT": 10000.0}
                for symbol in self.symbols:
                    base_currency = symbol.split("/")[0]
                    self.balance[base_currency] = 0.0
            else:
                # Initialiser le gestionnaire d'échange avec les clés API
                config = {
                    'api_key': self.api_key,
                    'api_secret': self.api_secret,
                    'testnet': self.use_testnet
                }
                
                # Si c'est Bitget, ajouter le mot de passe si disponible
                if self.exchange_id.lower() == 'bitget' and 'password' in EXCHANGE_API_KEYS.get('bitget', {}):
                    config['password'] = EXCHANGE_API_KEYS['bitget']['password']
                
                # Créer le gestionnaire d'échange
                self.exchange_manager = ExchangeManager(
                    exchange_id=self.exchange_id,
                    use_testnet=self.use_testnet
                )
                
                # Accéder à l'instance d'échange sous-jacente
                self.exchange = self.exchange_manager.exchange.exchange if self.exchange_manager else None
                
                # Charger les soldes si en mode papier ou réel
                if self.mode == "Papier":
                    # En mode papier, simuler un solde initial
                    self.balance = {"USDT": 10000.0}
                    for symbol in self.symbols:
                        base_currency = symbol.split("/")[0]
                        self.balance[base_currency] = 0.0
                elif self.mode == "Réel":
                    # En mode réel, charger les soldes réels
                    if self.exchange_manager:
                        self._update_balance()
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation de l'échange: {str(e)}")

    def _execute_trade(self, symbol, action, amount=None, price=None):
        """
        Exécute un trade.
        
        Args:
            symbol (str): Symbole à trader
            action (str): Action à exécuter (buy, sell)
            amount (float, optional): Montant à trader
            price (float, optional): Prix d'exécution
        
        Returns:
            dict: Résultat du trade
        """
        # Vérifier si le trading est actif
        if not self.trading_active:
            return {
                "success": False,
                "message": "Trading désactivé. Aucun ordre n'a été exécuté."
            }
            
        try:
            current_price = self._get_current_price(symbol)
            
            if not current_price:
                self._handle_error(f"Impossible d'obtenir le prix actuel pour {symbol}")
                return {
                    "success": False,
                    "message": f"Impossible d'obtenir le prix actuel pour {symbol}"
                }
            
            # Utiliser le prix fourni ou le prix actuel
            price = price or current_price
            
            # Si aucun montant n'est spécifié, utiliser un pourcentage du solde disponible
            if not amount:
                if action == "buy":
                    quote_currency = symbol.split("/")[1]
                    available_balance = self.balance.get(quote_currency, 0)
                    amount = (available_balance * 0.1) / price  # Utiliser 10% du solde
                else:  # sell
                    base_currency = symbol.split("/")[0]
                    amount = self.balance.get(base_currency, 0) * 0.1  # Vendre 10% de la position
            
            # Exécuter le trade en fonction du mode
            if self.mode == "Simulation":
                # Simuler l'exécution du trade
                trade_result = {
                    "symbol": symbol,
                    "action": action,
                    "amount": amount,
                    "price": price,
                    "value": amount * price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": f"sim_{int(time.time())}"
                }
                
                # Mettre à jour les soldes simulés
                if action == "buy":
                    base_currency = symbol.split("/")[0]
                    quote_currency = symbol.split("/")[1]
                    
                    # Vérifier si nous avons assez de fonds
                    if self.balance.get(quote_currency, 0) >= trade_result["value"]:
                        self.balance[quote_currency] = self.balance.get(quote_currency, 0) - trade_result["value"]
                        self.balance[base_currency] = self.balance.get(base_currency, 0) + amount
                        success = True
                    else:
                        return {
                            "success": False,
                            "message": f"Fonds insuffisants en {quote_currency}"
                        }
                else:  # sell
                    base_currency = symbol.split("/")[0]
                    quote_currency = symbol.split("/")[1]
                    
                    # Vérifier si nous avons assez d'actifs
                    if self.balance.get(base_currency, 0) >= amount:
                        self.balance[base_currency] = self.balance.get(base_currency, 0) - amount
                        self.balance[quote_currency] = self.balance.get(quote_currency, 0) + trade_result["value"]
                        success = True
                    else:
                        return {
                            "success": False,
                            "message": f"Actifs insuffisants en {base_currency}"
                        }
            
            elif self.mode == "Papier":
                # Similaire à la simulation, mais peut utiliser l'API pour obtenir des prix réels
                trade_result = {
                    "symbol": symbol,
                    "action": action,
                    "amount": amount,
                    "price": price,
                    "value": amount * price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "id": f"paper_{int(time.time())}"
                }
                
                # Mettre à jour les soldes simulés (même logique que pour la simulation)
                if action == "buy":
                    base_currency = symbol.split("/")[0]
                    quote_currency = symbol.split("/")[1]
                    
                    if self.balance.get(quote_currency, 0) >= trade_result["value"]:
                        self.balance[quote_currency] = self.balance.get(quote_currency, 0) - trade_result["value"]
                        self.balance[base_currency] = self.balance.get(base_currency, 0) + amount
                        success = True
                    else:
                        return {
                            "success": False,
                            "message": f"Fonds insuffisants en {quote_currency}"
                        }
                else:  # sell
                    base_currency = symbol.split("/")[0]
                    quote_currency = symbol.split("/")[1]
                    
                    if self.balance.get(base_currency, 0) >= amount:
                        self.balance[base_currency] = self.balance.get(base_currency, 0) - amount
                        self.balance[quote_currency] = self.balance.get(quote_currency, 0) + trade_result["value"]
                        success = True
                    else:
                        return {
                            "success": False,
                            "message": f"Actifs insuffisants en {base_currency}"
                        }
            
            elif self.mode == "Réel":
                # Exécuter un ordre réel sur l'échange via le gestionnaire d'échange
                if not self.exchange_manager:
                    return {
                        "success": False,
                        "message": "Gestionnaire d'échange non initialisé"
                    }
                
                try:
                    # Utiliser le gestionnaire d'échange pour exécuter l'ordre
                    order_type = "market"  # Par défaut, utiliser des ordres au marché
                    side = action  # 'buy' ou 'sell'
                    
                    # Récupérer les paramètres spécifiques à l'échange
                    exchange_params = self.exchange_manager.get_exchange_parameters()
                    
                    # Créer l'ordre via le gestionnaire d'échange
                    order = self.exchange_manager.create_order(
                        symbol=symbol,
                        order_type=order_type,
                        side=side,
                        amount=amount,
                        price=None  # Prix non nécessaire pour les ordres au marché
                    )
                    
                    trade_result = {
                        "symbol": symbol,
                        "action": action,
                        "amount": amount,
                        "price": price,
                        "value": amount * price,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "id": order.get("id", f"real_{int(time.time())}")
                    }
                    
                    success = True
                    
                    # Mettre à jour les soldes
                    self._update_balance()
                except Exception as e:
                    self._handle_error(f"Erreur lors de l'exécution de l'ordre: {str(e)}")
                    return {
                        "success": False,
                        "message": f"Erreur lors de l'exécution de l'ordre: {str(e)}"
                    }
            
            # Si le trade a réussi, l'ajouter à l'historique
            if success:
                self.trades.append(trade_result)
                
                # Mettre à jour les performances
                self._update_performance()
                
                return {
                    "success": True,
                    "trade": trade_result
                }
            
            return {
                "success": False,
                "message": "Échec de l'exécution du trade pour une raison inconnue"
            }
        except Exception as e:
            self._handle_error(f"Erreur lors de l'exécution du trade: {str(e)}")
            return {
                "success": False,
                "message": f"Erreur lors de l'exécution du trade: {str(e)}"
            }

    def _generate_signals(self):
        """
        Génère des signaux de trading pour chaque symbole.
        
        Returns:
            dict: Signaux de trading pour chaque symbole
        """
        signals = {}
        
        if not self.model:
            return signals
        
        try:
            for symbol in self.symbols:
                # Récupérer les données techniques
                technical_data = self._get_technical_data(symbol)
                
                if technical_data is None or len(technical_data) == 0:
                    continue
                
                # Récupérer les caractéristiques spécifiques à l'échange si disponible
                exchange_features = {}
                if self.exchange_manager and self.mode != "Simulation":
                    exchange_features = self.exchange_manager.get_exchange_specific_features(symbol)
                
                # Préparer les entrées du modèle
                model_inputs = {
                    'technical_input': np.expand_dims(technical_data, axis=0),
                    # Ajouter d'autres entrées si le modèle les utilise
                }
                
                # Faire la prédiction
                predictions = self.model.predict(model_inputs, verbose=0)
                
                # Traiter les prédictions
                signal = self._process_predictions(predictions, symbol)
                
                # Ajouter les caractéristiques spécifiques à l'échange au signal
                signal['exchange_features'] = exchange_features
                
                # Ajouter le signal au dictionnaire
                signals[symbol] = signal
                
                # Gérer les ordres existants si en mode réel
                if self.mode == "Réel" and self.exchange_manager:
                    # Utiliser le gestionnaire d'échange pour gérer les ordres existants
                    order_management = self.exchange_manager.manage_existing_orders(symbol, {symbol: signal})
                    signal['orders_kept'] = len(order_management.get('kept_orders', []))
                    signal['orders_cancelled'] = len(order_management.get('cancelled_orders', []))
            
            return signals
        except Exception as e:
            self._handle_error(f"Erreur lors de la génération des signaux: {str(e)}")
            return {}

def show_live_trading():
    """
    Affiche l'interface de trading en direct.
    """
    st.title("Trading en Direct")
    
    # Vérifier si un bot de trading existe déjà
    if 'trading_bot' not in st.session_state:
        st.session_state.trading_bot = None
    
    # Afficher l'interface de configuration ou de contrôle
    if not st.session_state.trading_bot:
        # Vérifier s'il existe un checkpoint à restaurer
        recovery_manager = RecoveryManager(None)
        latest_checkpoint = recovery_manager.load_latest_checkpoint()
        
        if latest_checkpoint and not st.session_state.trading_bot:
            st.info(f"Un checkpoint du {latest_checkpoint['timestamp']} a été trouvé. Voulez-vous restaurer l'état précédent du bot de trading?")
            
            if st.button("Restaurer à partir du checkpoint"):
                # Créer un bot temporaire pour la restauration
                temp_bot = TradingBot(
                    exchange_id=latest_checkpoint['exchange_id'],
                    api_key=latest_checkpoint['api_key'],
                    api_secret=latest_checkpoint['api_secret'],
                    model_path=latest_checkpoint['model_path'],
                    symbols=latest_checkpoint['symbols'],
                    mode=latest_checkpoint['mode']
                )
                
                # Attacher le recovery manager au bot temporaire
                temp_bot.recovery_manager = RecoveryManager(temp_bot)
                
                # Restaurer l'état
                if temp_bot.recovery_manager.restore_trading_bot():
                    st.session_state.trading_bot = temp_bot
                    st.success("Bot de trading restauré avec succès!")
                    st.experimental_rerun()
                else:
                    st.error("Erreur lors de la restauration du bot de trading.")
                    
        with st.form("trading_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Sélection de l'échange
                exchange_id = st.selectbox(
                    "Échange",
                    AVAILABLE_EXCHANGES,
                    index=0
                )
                
                # Sélection du modèle
                models = get_available_models()
                selected_model = st.selectbox(
                    "Modèle à utiliser",
                    models if models else ["Aucun modèle disponible"],
                    key="trading_model"
                )
                
                # Sélection du mode de trading
                trading_mode = st.selectbox(
                    "Mode de trading",
                    TRADING_MODES,
                    index=0
                )
                
                # Option pour utiliser le testnet
                use_testnet = st.checkbox("Utiliser le testnet", value=True)
            
            with col2:
                # Sélection des symboles
                selected_symbols = st.multiselect(
                    "Symboles à trader",
                    AVAILABLE_SYMBOLS,
                    default=["BTC/USDT"]
                )
                
                # Clés API (uniquement pour les modes Papier et Réel)
                api_key = st.text_input(
                    "Clé API",
                    type="password",
                    disabled=trading_mode == "Simulation"
                )
                
                api_secret = st.text_input(
                    "Secret API",
                    type="password",
                    disabled=trading_mode == "Simulation"
                )
                
                # Mot de passe/Passphrase (pour certains échanges comme Bitget)
                if exchange_id.lower() == "bitget":
                    api_passphrase = st.text_input(
                        "Passphrase API",
                        type="password",
                        disabled=trading_mode == "Simulation"
                    )
                    # Mettre à jour la configuration des clés API
                    if api_passphrase and trading_mode != "Simulation":
                        EXCHANGE_API_KEYS["bitget"]["password"] = api_passphrase
            
            # Bouton de soumission
            submit_button = st.form_submit_button("Configurer le bot de trading")
        
        if submit_button:
            if selected_model == "Aucun modèle disponible":
                st.error("Aucun modèle disponible pour le trading.")
            elif not selected_symbols:
                st.error("Veuillez sélectionner au moins un symbole.")
            else:
                # Mettre à jour les clés API dans la configuration
                if trading_mode != "Simulation" and api_key and api_secret:
                    EXCHANGE_API_KEYS[exchange_id.lower()]["api_key"] = api_key
                    EXCHANGE_API_KEYS[exchange_id.lower()]["api_secret"] = api_secret
                
                # Créer le bot de trading
                st.session_state.trading_bot = TradingBot(
                    exchange_id=exchange_id,
                    api_key=api_key if api_key and trading_mode != "Simulation" else None,
                    api_secret=api_secret if api_secret and trading_mode != "Simulation" else None,
                    model_path=selected_model,
                    symbols=selected_symbols,
                    mode=trading_mode,
                    use_testnet=use_testnet
                )
                
                st.success(f"Bot de trading configuré avec succès pour {len(selected_symbols)} symboles en mode {trading_mode}.")
