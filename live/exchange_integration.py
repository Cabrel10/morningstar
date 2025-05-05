import ccxt
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import os

# Configuration du logging
logger = logging.getLogger(__name__)

class ExchangeBase:
    """
    Classe de base pour l'intégration avec les exchanges crypto.
    Fournit une interface commune pour tous les exchanges supportés.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la connexion à l'exchange.
        
        Args:
            config: Dictionnaire de configuration contenant:
                - exchange_id: Identifiant de l'exchange (ex: 'bitget', 'binance', 'kucoin')
                - api_key: Clé API (optionnel pour les données publiques)
                - api_secret: Secret API (optionnel pour les données publiques)
                - password: Mot de passe (requis pour certains exchanges comme KuCoin)
                - testnet: Booléen indiquant si on utilise le testnet
        """
        self.config = config
        self.exchange_id = config.get('exchange_id', 'binance')
        self.testnet = config.get('testnet', False)
        self.exchange = self._init_exchange()
        self.markets = {}
        self.retry_count = 0
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2.0)
        
    def _init_exchange(self) -> ccxt.Exchange:
        """
        Initialise la connexion à l'exchange avec la configuration appropriée.
        
        Returns:
            Instance ccxt de l'exchange configuré
        
        Raises:
            ValueError: Si l'exchange n'est pas supporté ou si la configuration est invalide
        """
        try:
            # Vérifier si l'exchange est supporté par ccxt
            if not hasattr(ccxt, self.exchange_id):
                raise ValueError(f"Exchange '{self.exchange_id}' non supporté par ccxt")
            
            exchange_class = getattr(ccxt, self.exchange_id)
            
            # Configuration de base
            exchange_config = {
                'enableRateLimit': True,  # Respecter les limites de l'API
                'timeout': 30000,  # 30 secondes de timeout
                'options': {
                    'adjustForTimeDifference': True
                }
            }
            
            # Ajouter les clés API si fournies
            if 'api_key' in self.config and 'api_secret' in self.config:
                exchange_config['apiKey'] = self.config['api_key']
                exchange_config['secret'] = self.config['api_secret']
                
                # Certains exchanges comme KuCoin nécessitent un mot de passe
                if 'password' in self.config and self.config['password']:
                    exchange_config['password'] = self.config['password']
            
            # Configuration spécifique pour le testnet
            if self.testnet:
                if self.exchange_id == 'binance':
                    exchange_config['options']['defaultType'] = 'future'
                    exchange_config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binancefuture.com/fapi/v1',
                            'private': 'https://testnet.binancefuture.com/fapi/v1',
                        }
                    }
                elif self.exchange_id == 'bitget':
                    # Configuration pour Bitget testnet
                    exchange_config['urls'] = {
                        'api': 'https://api-testnet.bitget.com'
                    }
                elif self.exchange_id == 'kucoin':
                    # Configuration pour KuCoin testnet
                    exchange_config['urls'] = {
                        'api': 'https://openapi-sandbox.kucoin.com'
                    }
            
            # Créer l'instance de l'exchange
            exchange = exchange_class(exchange_config)
            
            # Charger les marchés pour validation des symboles
            try:
                exchange.load_markets()
                logger.info(f"Marchés chargés pour {self.exchange_id}: {len(exchange.markets)} paires disponibles")
                self.markets = exchange.markets
            except Exception as e:
                logger.warning(f"Impossible de charger les marchés pour {self.exchange_id}: {e}")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange {self.exchange_id}: {e}")
            raise
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Formate le symbole selon les conventions de l'exchange.
        
        Args:
            symbol: Symbole à formater (ex: 'BTC/USDT', 'BTCUSDT')
            
        Returns:
            Symbole formaté selon les conventions de l'exchange
        """
        # Si le symbole est déjà dans le format correct pour cet exchange
        if symbol in self.markets:
            return symbol
        
        # Essayer différents formats
        formats_to_try = []
        
        # Format avec /
        if '/' not in symbol:
            # Essayer de deviner la paire de base/quote
            for quote in ['USDT', 'USD', 'BUSD', 'USDC', 'BTC', 'ETH']:
                if symbol.endswith(quote):
                    base = symbol[:-len(quote)]
                    formats_to_try.append(f"{base}/{quote}")
                    break
        else:
            formats_to_try.append(symbol)  # Déjà au format base/quote
        
        # Format sans /
        if '/' in symbol:
            base, quote = symbol.split('/')
            formats_to_try.append(f"{base}{quote}")
        else:
            formats_to_try.append(symbol)  # Déjà au format sans séparateur
        
        # Format avec - (pour KuCoin)
        if '/' in symbol and self.exchange_id == 'kucoin':
            base, quote = symbol.split('/')
            formats_to_try.append(f"{base}-{quote}")
        
        # Essayer chaque format
        for fmt in formats_to_try:
            if fmt in self.markets:
                logger.info(f"Symbole '{symbol}' formaté en '{fmt}' pour {self.exchange_id}")
                return fmt
        
        # Si aucun format ne correspond, utiliser le format original
        logger.warning(f"Aucun format valide trouvé pour le symbole '{symbol}' sur {self.exchange_id}. Utilisation du format original.")
        return symbol
    
    def _handle_rate_limit(self, e: Exception) -> bool:
        """
        Gère les erreurs de rate limit en implémentant un backoff exponentiel.
        
        Args:
            e: Exception levée par l'API
            
        Returns:
            bool: True si l'erreur a été gérée et qu'une nouvelle tentative peut être faite,
                  False si le nombre maximum de tentatives a été atteint
        """
        self.retry_count += 1
        
        if self.retry_count <= self.max_retries:
            # Backoff exponentiel
            delay = self.retry_delay * (2 ** (self.retry_count - 1))
            logger.warning(f"Rate limit atteint. Attente de {delay:.2f}s avant nouvelle tentative ({self.retry_count}/{self.max_retries})")
            time.sleep(delay)
            return True
        else:
            logger.error(f"Nombre maximum de tentatives atteint ({self.max_retries}). Abandon.")
            self.retry_count = 0  # Réinitialiser pour les prochains appels
            return False
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, since: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère les données OHLCV pour un symbole et un timeframe donnés.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            timeframe: Intervalle de temps (ex: '1m', '5m', '1h', '1d')
            limit: Nombre maximum de bougies à récupérer
            since: Timestamp Unix en millisecondes pour la date de début
            
        Returns:
            DataFrame pandas avec les colonnes [timestamp, open, high, low, close, volume]
        """
        formatted_symbol = self._format_symbol(symbol)
        self.retry_count = 0  # Réinitialiser le compteur de tentatives
        
        while True:
            try:
                # Vérifier si l'exchange supporte fetchOHLCV
                if not self.exchange.has['fetchOHLCV']:
                    raise ValueError(f"L'exchange {self.exchange_id} ne supporte pas fetchOHLCV")
                
                # Récupérer les données OHLCV
                ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, timeframe, since, limit)
                
                if not ohlcv:
                    logger.warning(f"Aucune donnée OHLCV retournée pour {formatted_symbol} sur {self.exchange_id}")
                    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Convertir en DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                logger.info(f"Récupération de {len(df)} bougies pour {formatted_symbol} sur {self.exchange_id}")
                return df
                
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            except ccxt.NetworkError as e:
                logger.error(f"Erreur réseau lors de la récupération des données OHLCV: {e}")
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    time.sleep(self.retry_delay)
                    continue
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données OHLCV: {e}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les informations de ticker pour un symbole donné.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            
        Returns:
            Dictionnaire contenant les informations du ticker
        """
        formatted_symbol = self._format_symbol(symbol)
        self.retry_count = 0
        
        while True:
            try:
                ticker = self.exchange.fetch_ticker(formatted_symbol)
                return ticker
            
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return {}
            
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du ticker pour {formatted_symbol}: {e}")
                return {}
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Crée un ordre sur l'exchange.
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            order_type: Type d'ordre ('market', 'limit')
            side: Côté de l'ordre ('buy', 'sell')
            amount: Quantité à acheter/vendre
            price: Prix pour les ordres limit (None pour les ordres market)
            
        Returns:
            Dictionnaire contenant les informations de l'ordre créé
        """
        if not self.exchange.has['createOrder']:
            raise ValueError(f"L'exchange {self.exchange_id} ne supporte pas la création d'ordres")
        
        if 'api_key' not in self.config or 'api_secret' not in self.config:
            raise ValueError("Clés API non configurées. Impossible de créer un ordre.")
        
        formatted_symbol = self._format_symbol(symbol)
        self.retry_count = 0
        
        while True:
            try:
                # Créer l'ordre
                if order_type == 'market':
                    order = self.exchange.create_market_order(formatted_symbol, side, amount)
                elif order_type == 'limit' and price is not None:
                    order = self.exchange.create_limit_order(formatted_symbol, side, amount, price)
                else:
                    raise ValueError(f"Type d'ordre '{order_type}' non supporté ou prix manquant pour un ordre limit")
                
                logger.info(f"Ordre {order_type} {side} créé pour {amount} {formatted_symbol} sur {self.exchange_id}")
                return order
            
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return {}
            
            except ccxt.InsufficientFunds as e:
                logger.error(f"Fonds insuffisants pour créer l'ordre: {e}")
                return {'error': 'insufficient_funds', 'message': str(e)}
            
            except ccxt.InvalidOrder as e:
                logger.error(f"Ordre invalide: {e}")
                return {'error': 'invalid_order', 'message': str(e)}
            
            except Exception as e:
                logger.error(f"Erreur lors de la création de l'ordre: {e}")
                return {'error': 'unknown', 'message': str(e)}
    
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Annule un ordre existant.
        
        Args:
            order_id: Identifiant de l'ordre à annuler
            symbol: Symbole de la paire (requis par certains exchanges)
            
        Returns:
            Dictionnaire contenant les informations de l'ordre annulé
        """
        if not self.exchange.has['cancelOrder']:
            raise ValueError(f"L'exchange {self.exchange_id} ne supporte pas l'annulation d'ordres")
        
        if 'api_key' not in self.config or 'api_secret' not in self.config:
            raise ValueError("Clés API non configurées. Impossible d'annuler un ordre.")
        
        formatted_symbol = self._format_symbol(symbol) if symbol else None
        self.retry_count = 0
        
        while True:
            try:
                result = self.exchange.cancel_order(order_id, formatted_symbol)
                logger.info(f"Ordre {order_id} annulé sur {self.exchange_id}")
                return result
            
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return {}
            
            except ccxt.OrderNotFound as e:
                logger.error(f"Ordre {order_id} non trouvé: {e}")
                return {'error': 'order_not_found', 'message': str(e)}
            
            except Exception as e:
                logger.error(f"Erreur lors de l'annulation de l'ordre {order_id}: {e}")
                return {'error': 'unknown', 'message': str(e)}
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Récupère le solde du compte.
        
        Returns:
            Dictionnaire contenant les soldes pour chaque devise
        """
        if not self.exchange.has['fetchBalance']:
            raise ValueError(f"L'exchange {self.exchange_id} ne supporte pas la récupération du solde")
        
        if 'api_key' not in self.config or 'api_secret' not in self.config:
            raise ValueError("Clés API non configurées. Impossible de récupérer le solde.")
        
        self.retry_count = 0
        
        while True:
            try:
                balance = self.exchange.fetch_balance()
                return balance
            
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return {}
            
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du solde: {e}")
                return {}
    
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole de la paire (optionnel)
            
        Returns:
            Liste des ordres ouverts
        """
        if not self.exchange.has['fetchOpenOrders']:
            raise ValueError(f"L'exchange {self.exchange_id} ne supporte pas la récupération des ordres ouverts")
        
        if 'api_key' not in self.config or 'api_secret' not in self.config:
            raise ValueError("Clés API non configurées. Impossible de récupérer les ordres ouverts.")
        
        formatted_symbol = self._format_symbol(symbol) if symbol else None
        self.retry_count = 0
        
        while True:
            try:
                open_orders = self.exchange.fetch_open_orders(formatted_symbol)
                return open_orders
            
            except ccxt.RateLimitExceeded as e:
                if not self._handle_rate_limit(e):
                    logger.error(f"Erreur de rate limit non récupérable: {e}")
                    return []
            
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des ordres ouverts: {e}")
                return []


class BitgetExchange(ExchangeBase):
    """
    Implémentation spécifique pour l'exchange Bitget.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la connexion à Bitget.
        
        Args:
            config: Dictionnaire de configuration
        """
        # S'assurer que l'exchange_id est correctement défini
        config['exchange_id'] = 'bitget'
        super().__init__(config)
    
    def get_btc_test_pair(self) -> str:
        """
        Retourne la paire de test #BTC sur Bitget.
        
        Returns:
            Symbole de la paire de test
        """
        return "#BTC/USDT"  # Paire de test spécifique à Bitget
    
    def place_test_order(self, side: str, amount: float) -> Dict[str, Any]:
        """
        Place un ordre de test sur la paire #BTC.
        
        Args:
            side: Côté de l'ordre ('buy', 'sell')
            amount: Quantité à acheter/vendre
            
        Returns:
            Résultat de l'ordre
        """
        test_pair = self.get_btc_test_pair()
        return self.create_order(test_pair, 'market', side, amount)


class BinanceExchange(ExchangeBase):
    """
    Implémentation spécifique pour l'exchange Binance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la connexion à Binance.
        
        Args:
            config: Dictionnaire de configuration
        """
        config['exchange_id'] = 'binance'
        super().__init__(config)
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Récupère le taux de financement pour un symbole donné (spécifique aux contrats à terme).
        
        Args:
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            
        Returns:
            Taux de financement actuel
        """
        formatted_symbol = self._format_symbol(symbol)
        
        try:
            funding_rate = self.exchange.fetch_funding_rate(formatted_symbol)
            return funding_rate.get('fundingRate', 0.0)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du taux de financement pour {formatted_symbol}: {e}")
            return 0.0


class KuCoinExchange(ExchangeBase):
    """
    Implémentation spécifique pour l'exchange KuCoin.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la connexion à KuCoin.
        
        Args:
            config: Dictionnaire de configuration
        """
        config['exchange_id'] = 'kucoin'
        super().__init__(config)
    
    def fetch_currencies(self) -> Dict[str, Any]:
        """
        Récupère les informations sur les devises disponibles sur KuCoin.
        
        Returns:
            Dictionnaire des devises disponibles
        """
        try:
            currencies = self.exchange.fetch_currencies()
            return currencies
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des devises sur KuCoin: {e}")
            return {}


class ExchangeFactory:
    """
    Factory pour créer des instances d'exchanges.
    """
    
    @staticmethod
    def create_exchange(exchange_id: str, config: Dict[str, Any]) -> ExchangeBase:
        """
        Crée une instance d'exchange basée sur l'identifiant.
        
        Args:
            exchange_id: Identifiant de l'exchange ('bitget', 'binance', 'kucoin')
            config: Configuration de l'exchange
            
        Returns:
            Instance de l'exchange
            
        Raises:
            ValueError: Si l'exchange n'est pas supporté
        """
        config['exchange_id'] = exchange_id.lower()
        
        if exchange_id.lower() == 'bitget':
            return BitgetExchange(config)
        elif exchange_id.lower() == 'binance':
            return BinanceExchange(config)
        elif exchange_id.lower() == 'kucoin':
            return KuCoinExchange(config)
        else:
            raise ValueError(f"Exchange '{exchange_id}' non supporté")
