"""
Module de gestion des échanges pour le trading live.
Fournit une interface unifiée pour interagir avec différents exchanges (Binance, Bitget).
"""

import ccxt
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import os
import json
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les importations absolues
sys.path.append(str(Path(__file__).resolve().parent.parent))

from live.exchange_integration import ExchangeBase, BinanceExchange, BitgetExchange, ExchangeFactory
from config.api_keys import EXCHANGE_API_KEYS

# Configuration du logging
logger = logging.getLogger(__name__)

class ExchangeManager:
    """
    Gestionnaire d'échanges pour le trading live.
    Gère les connexions aux différents exchanges et fournit une interface unifiée.
    """
    
    def __init__(self, exchange_id: str = "binance", use_testnet: bool = False):
        """
        Initialise le gestionnaire d'échanges.
        
        Args:
            exchange_id: Identifiant de l'exchange ('binance', 'bitget')
            use_testnet: Utiliser le testnet au lieu du mainnet
        """
        self.exchange_id = exchange_id.lower()
        self.use_testnet = use_testnet
        self.exchange = None
        self.api_config = None
        self.orders_cache = {}
        self.position_cache = {}
        self.last_balance_update = 0
        self.balance_cache = {}
        self.balance_update_interval = 60  # Secondes
        
        # Initialiser l'échange
        self._init_exchange()
    
    def _init_exchange(self):
        """
        Initialise la connexion à l'échange avec les clés API configurées.
        """
        # Récupérer la configuration de l'échange
        if self.exchange_id not in EXCHANGE_API_KEYS:
            raise ValueError(f"Configuration pour l'échange {self.exchange_id} non trouvée")
        
        # Copier la configuration pour ne pas modifier l'original
        self.api_config = EXCHANGE_API_KEYS[self.exchange_id].copy()
        
        # Forcer l'utilisation du testnet si demandé
        if self.use_testnet:
            self.api_config['testnet'] = True
        
        # Créer l'instance d'échange via la factory
        try:
            self.exchange = ExchangeFactory.create_exchange(self.exchange_id, self.api_config)
            logger.info(f"Connexion à l'échange {self.exchange_id} initialisée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'échange {self.exchange_id}: {e}")
            raise
    
    def get_exchange_parameters(self) -> Dict[str, Any]:
        """
        Récupère les paramètres spécifiques à l'échange.
        
        Returns:
            Dictionnaire des paramètres spécifiques à l'échange
        """
        params = {
            'exchange_id': self.exchange_id,
            'testnet': self.use_testnet
        }
        
        # Ajouter les paramètres spécifiques à l'échange
        if 'params' in self.api_config:
            params.update(self.api_config['params'])
        
        # Ajouter des paramètres spécifiques selon l'échange
        if self.exchange_id == 'binance':
            # Paramètres spécifiques à Binance
            params.update({
                'order_type_mapping': {
                    'market': 'MARKET',
                    'limit': 'LIMIT',
                    'stop_loss': 'STOP_LOSS',
                    'take_profit': 'TAKE_PROFIT'
                },
                'leverage_modes': ['cross', 'isolated'],
                'default_leverage': 1
            })
        elif self.exchange_id == 'bitget':
            # Paramètres spécifiques à Bitget
            params.update({
                'order_type_mapping': {
                    'market': 'market',
                    'limit': 'limit',
                    'stop_loss': 'stop_loss',
                    'take_profit': 'take_profit'
                },
                'leverage_modes': ['cross', 'fixed'],
                'default_leverage': 1
            })
        
        return params
    
    def get_balance(self, force_update: bool = False) -> Dict[str, float]:
        """
        Récupère le solde du compte.
        
        Args:
            force_update: Forcer la mise à jour du cache
            
        Returns:
            Dictionnaire des soldes par devise
        """
        current_time = time.time()
        
        # Vérifier si une mise à jour du cache est nécessaire
        if force_update or (current_time - self.last_balance_update) > self.balance_update_interval:
            try:
                balance_data = self.exchange.fetch_balance()
                
                # Formater les données de solde
                self.balance_cache = {}
                
                # Extraire les soldes disponibles
                if 'free' in balance_data:
                    for currency, amount in balance_data['free'].items():
                        if amount > 0:
                            self.balance_cache[currency] = amount
                
                # Si 'free' n'est pas disponible, utiliser le format alternatif
                else:
                    for currency, data in balance_data.items():
                        if isinstance(data, dict) and 'free' in data and data['free'] > 0:
                            self.balance_cache[currency] = data['free']
                
                self.last_balance_update = current_time
                logger.debug(f"Solde mis à jour: {self.balance_cache}")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du solde: {e}")
                # En cas d'erreur, utiliser le cache existant
        
        return self.balance_cache
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            # Mettre à jour le cache des ordres
            if symbol:
                self.orders_cache[symbol] = orders
            else:
                # Regrouper les ordres par symbole
                for order in orders:
                    order_symbol = order.get('symbol')
                    if order_symbol:
                        if order_symbol not in self.orders_cache:
                            self.orders_cache[order_symbol] = []
                        self.orders_cache[order_symbol].append(order)
            
            return orders
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {e}")
            
            # En cas d'erreur, utiliser le cache si disponible
            if symbol and symbol in self.orders_cache:
                return self.orders_cache[symbol]
            elif not symbol:
                # Fusionner tous les ordres en cache
                all_orders = []
                for orders_list in self.orders_cache.values():
                    all_orders.extend(orders_list)
                return all_orders
            
            return []
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Annule tous les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Liste des ordres annulés
        """
        cancelled_orders = []
        
        try:
            # Récupérer les ordres ouverts
            open_orders = self.get_open_orders(symbol)
            
            # Annuler chaque ordre
            for order in open_orders:
                order_id = order.get('id')
                order_symbol = order.get('symbol')
                
                if order_id and order_symbol:
                    try:
                        result = self.exchange.cancel_order(order_id, order_symbol)
                        cancelled_orders.append(result)
                        logger.info(f"Ordre {order_id} sur {order_symbol} annulé avec succès")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'annulation de l'ordre {order_id} sur {order_symbol}: {e}")
            
            # Mettre à jour le cache des ordres
            if symbol:
                self.orders_cache[symbol] = []
            else:
                self.orders_cache = {}
            
            return cancelled_orders
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation des ordres: {e}")
            return []
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                    price: Optional[float] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Crée un ordre sur l'échange.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            order_type: Type d'ordre ('market', 'limit', 'stop_loss', 'take_profit')
            side: Côté de l'ordre ('buy', 'sell')
            amount: Quantité à acheter/vendre
            price: Prix pour les ordres limit (optionnel)
            params: Paramètres supplémentaires spécifiques à l'échange
            
        Returns:
            Informations sur l'ordre créé
        """
        try:
            # Adapter les paramètres selon l'échange
            exchange_params = self.get_exchange_parameters()
            order_type_mapping = exchange_params.get('order_type_mapping', {})
            
            # Convertir le type d'ordre selon l'échange
            exchange_order_type = order_type_mapping.get(order_type, order_type)
            
            # Fusionner les paramètres
            merged_params = {}
            if params:
                merged_params.update(params)
            
            # Créer l'ordre
            order_result = self.exchange.create_order(
                symbol=symbol,
                order_type=exchange_order_type,
                side=side,
                amount=amount,
                price=price,
                params=merged_params
            )
            
            # Mettre à jour le cache des ordres
            if symbol not in self.orders_cache:
                self.orders_cache[symbol] = []
            
            self.orders_cache[symbol].append(order_result)
            
            logger.info(f"Ordre créé sur {symbol}: {side} {amount} à {price or 'prix du marché'}")
            return order_result
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre sur {symbol}: {e}")
            raise
    
    def should_keep_order(self, order: Dict[str, Any], current_prediction: Dict[str, Any]) -> bool:
        """
        Détermine si un ordre existant doit être conservé en fonction des prédictions actuelles.
        
        Args:
            order: Informations sur l'ordre
            current_prediction: Prédictions actuelles du modèle
            
        Returns:
            True si l'ordre doit être conservé, False sinon
        """
        # Extraire les informations de l'ordre
        order_symbol = order.get('symbol')
        order_side = order.get('side')
        order_type = order.get('type')
        order_price = order.get('price')
        
        # Extraire les prédictions pour ce symbole
        symbol_prediction = current_prediction.get(order_symbol, {})
        predicted_signal = symbol_prediction.get('signal')
        predicted_stop_loss = symbol_prediction.get('stop_loss')
        predicted_take_profit = symbol_prediction.get('take_profit')
        
        # Logique pour déterminer si l'ordre doit être conservé
        if order_type == 'limit':
            # Pour les ordres limit, vérifier si le signal est toujours valide
            if order_side == 'buy' and predicted_signal in ['buy', 'strong_buy']:
                return True
            elif order_side == 'sell' and predicted_signal in ['sell', 'strong_sell']:
                return True
            return False
        
        elif order_type in ['stop_loss', 'stop']:
            # Pour les stop loss, vérifier si le nouveau stop loss est proche
            if abs(order_price - predicted_stop_loss) / order_price < 0.05:  # 5% de différence
                return True
            return False
        
        elif order_type in ['take_profit', 'limit']:
            # Pour les take profit, vérifier si le nouveau take profit est proche
            if abs(order_price - predicted_take_profit) / order_price < 0.05:  # 5% de différence
                return True
            return False
        
        # Par défaut, conserver les ordres non reconnus
        return True
    
    def manage_existing_orders(self, symbol: str, current_prediction: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Gère les ordres existants en fonction des prédictions actuelles.
        
        Args:
            symbol: Symbole de la paire
            current_prediction: Prédictions actuelles du modèle
            
        Returns:
            Dictionnaire contenant les ordres conservés et annulés
        """
        result = {
            'kept_orders': [],
            'cancelled_orders': []
        }
        
        try:
            # Récupérer les ordres ouverts pour ce symbole
            open_orders = self.get_open_orders(symbol)
            
            for order in open_orders:
                # Vérifier si l'ordre doit être conservé
                if self.should_keep_order(order, current_prediction):
                    result['kept_orders'].append(order)
                else:
                    # Annuler l'ordre
                    order_id = order.get('id')
                    try:
                        cancelled = self.exchange.cancel_order(order_id, symbol)
                        result['cancelled_orders'].append(cancelled)
                        logger.info(f"Ordre {order_id} sur {symbol} annulé car ne correspond plus aux prédictions")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'annulation de l'ordre {order_id} sur {symbol}: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la gestion des ordres existants pour {symbol}: {e}")
            return result
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Récupère les données de marché pour un symbole et un timeframe donnés.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            timeframe: Intervalle de temps (ex: '1m', '5m', '1h', '1d')
            limit: Nombre de bougies à récupérer
            
        Returns:
            DataFrame pandas avec les données OHLCV
        """
        try:
            # Récupérer les données OHLCV
            ohlcv_data = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            
            if not ohlcv_data:
                logger.warning(f"Aucune donnée OHLCV récupérée pour {symbol}")
                return pd.DataFrame()
            
            # Convertir en DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir les timestamps en datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de marché pour {symbol}: {e}")
            return pd.DataFrame()
    
    def get_exchange_specific_features(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère des caractéristiques spécifiques à l'échange pour un symbole donné.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            
        Returns:
            Dictionnaire des caractéristiques spécifiques à l'échange
        """
        features = {}
        
        try:
            # Caractéristiques communes
            ticker = self.exchange.fetch_ticker(symbol)
            features['bid'] = ticker.get('bid', 0)
            features['ask'] = ticker.get('ask', 0)
            features['spread'] = features['ask'] - features['bid']
            features['volume_24h'] = ticker.get('volume', 0)
            
            # Caractéristiques spécifiques à Binance
            if self.exchange_id == 'binance':
                if isinstance(self.exchange, BinanceExchange):
                    # Récupérer le taux de financement pour les contrats à terme
                    try:
                        features['funding_rate'] = self.exchange.fetch_funding_rate(symbol)
                    except:
                        features['funding_rate'] = 0
            
            # Caractéristiques spécifiques à Bitget
            elif self.exchange_id == 'bitget':
                # Ajouter des caractéristiques spécifiques à Bitget si nécessaire
                pass
            
            return features
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des caractéristiques pour {symbol}: {e}")
            return features
    
    def save_trading_data(self, data: Dict[str, Any], file_path: str):
        """
        Sauvegarde les données de trading dans un fichier JSON.
        
        Args:
            data: Données à sauvegarder
            file_path: Chemin du fichier de sauvegarde
        """
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convertir les objets non sérialisables
            def json_serial(obj):
                if isinstance(obj, (datetime, np.datetime64)):
                    return obj.isoformat()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} non sérialisable")
            
            # Sauvegarder les données
            with open(file_path, 'w') as f:
                json.dump(data, f, default=json_serial, indent=2)
            
            logger.info(f"Données de trading sauvegardées dans {file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données de trading: {e}")
