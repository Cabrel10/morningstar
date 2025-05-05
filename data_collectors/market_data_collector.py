#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collecteur de donnu00e9es de marchu00e9 pour le modu00e8le Morningstar.
Ce module utilise l'API CCXT pour collecter des donnu00e9es OHLCV et des mu00e9triques de volume.
"""

import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """
    Classe pour collecter des donnu00e9es de marchu00e9 u00e0 partir d'exchanges de crypto-monnaies.
    """
    def __init__(self, exchange_id='binance', rate_limit_ms=1000):
        """
        Initialise le collecteur de donnu00e9es de marchu00e9.
        
        Args:
            exchange_id: ID de l'exchange u00e0 utiliser (par du00e9faut: 'binance')
            rate_limit_ms: Du00e9lai entre les requ00eates en millisecondes
        """
        self.exchange_id = exchange_id
        self.rate_limit_ms = rate_limit_ms
        
        # Initialiser l'exchange
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
            })
            logger.info(f"Exchange {exchange_id} initialisu00e9 avec succu00e8s")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'exchange {exchange_id}: {e}")
            self.exchange = None
    
    def fetch_ohlcv(self, symbol, timeframe='1d', since=None, until=None, limit=1000):
        """
        Ru00e9cupu00e8re les donnu00e9es OHLCV pour un symbole donnu00e9.
        
        Args:
            symbol: Paire de trading (ex: 'BTC/USDT')
            timeframe: Intervalle de temps ('1d', '1h', etc.)
            since: Date de du00e9but (format: 'YYYY-MM-DD' ou timestamp)
            until: Date de fin (format: 'YYYY-MM-DD' ou timestamp)
            limit: Nombre maximum d'entrées par requ00eate
            
        Returns:
            DataFrame contenant les donnu00e9es OHLCV
        """
        if self.exchange is None:
            logger.error("Exchange non initialisu00e9")
            return pd.DataFrame()
        
        # Convertir les dates en timestamps si nu00e9cessaire
        if since is not None and isinstance(since, str):
            since = int(datetime.strptime(since, '%Y-%m-%d').timestamp() * 1000)
        
        if until is not None and isinstance(until, str):
            until = int(datetime.strptime(until, '%Y-%m-%d').timestamp() * 1000)
        
        logger.info(f"Ru00e9cupu00e9ration des donnu00e9es OHLCV pour {symbol} de {since} u00e0 {until}")
        
        all_ohlcv = []
        current_since = since
        
        try:
            while True:
                # Ru00e9cupu00e9rer un lot de donnu00e9es
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
                
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                logger.debug(f"Ru00e9cupu00e9ru00e9 {len(ohlcv)} entrées pour {symbol}")
                
                # Mettre u00e0 jour le timestamp de du00e9part pour la prochaine requ00eate
                current_since = ohlcv[-1][0] + 1
                
                # Arru00eater si on a atteint la date de fin
                if until is not None and current_since >= until:
                    break
                
                # Respecter la limite de taux
                time.sleep(self.rate_limit_ms / 1000)
            
            # Convertir en DataFrame
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                
                # Filtrer par date de fin si spu00e9cifiu00e9e
                if until is not None:
                    until_dt = datetime.fromtimestamp(until / 1000)
                    df = df[df['timestamp'] <= until_dt]
                
                logger.info(f"Donnu00e9es OHLCV ru00e9cupu00e9ru00e9es pour {symbol}: {len(df)} entrées")
                return df
            else:
                logger.warning(f"Aucune donnu00e9e OHLCV ru00e9cupu00e9ru00e9e pour {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Erreur lors de la ru00e9cupu00e9ration des donnu00e9es OHLCV pour {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_order_book(self, symbol, limit=100):
        """
        Ru00e9cupu00e8re le carnet d'ordres pour un symbole donnu00e9.
        
        Args:
            symbol: Paire de trading (ex: 'BTC/USDT')
            limit: Profondeur du carnet d'ordres
            
        Returns:
            Dictionnaire contenant le carnet d'ordres
        """
        if self.exchange is None:
            logger.error("Exchange non initialisu00e9")
            return {}
        
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            logger.info(f"Carnet d'ordres ru00e9cupu00e9ru00e9 pour {symbol} avec {limit} niveaux")
            return order_book
        except Exception as e:
            logger.error(f"Erreur lors de la ru00e9cupu00e9ration du carnet d'ordres pour {symbol}: {e}")
            return {}
    
    def fetch_trades(self, symbol, since=None, limit=1000):
        """
        Ru00e9cupu00e8re les transactions ru00e9centes pour un symbole donnu00e9.
        
        Args:
            symbol: Paire de trading (ex: 'BTC/USDT')
            since: Timestamp de du00e9but
            limit: Nombre maximum de transactions
            
        Returns:
            Liste des transactions
        """
        if self.exchange is None:
            logger.error("Exchange non initialisu00e9")
            return []
        
        try:
            trades = self.exchange.fetch_trades(symbol, since, limit)
            logger.info(f"Transactions ru00e9cupu00e9ru00e9es pour {symbol}: {len(trades)}")
            return trades
        except Exception as e:
            logger.error(f"Erreur lors de la ru00e9cupu00e9ration des transactions pour {symbol}: {e}")
            return []
    
    def add_volume_metrics(self, df, symbol):
        """
        Ajoute des mu00e9triques de volume au DataFrame.
        
        Args:
            df: DataFrame contenant les donnu00e9es OHLCV
            symbol: Paire de trading
            
        Returns:
            DataFrame enrichi avec des mu00e9triques de volume
        """
        if df.empty:
            return df
        
        logger.info(f"Ajout de mu00e9triques de volume pour {symbol}")
        
        # Volume moyen sur diffu00e9rentes pu00e9riodes
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        
        # Ratio de volume
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        
        # Volume relatif (par rapport au volume moyen sur 20 jours)
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Accumulation/Distribution
        df['money_flow_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
        df['adl'] = df['money_flow_volume'].cumsum()
        
        # On-Balance Volume (OBV)
        df['price_change'] = df['close'].diff()
        df['obv'] = np.where(df['price_change'] > 0, df['volume'], 
                           np.where(df['price_change'] < 0, -df['volume'], 0)).cumsum()
        
        # Chaikin Money Flow (CMF)
        df['cmf'] = df['money_flow_volume'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Remplir les valeurs manquantes
        df = df.fillna(0)
        
        return df
    
    def add_technical_indicators(self, df):
        """
        Ajoute des indicateurs techniques au DataFrame.
        
        Args:
            df: DataFrame contenant les donnu00e9es OHLCV
            
        Returns:
            DataFrame enrichi avec des indicateurs techniques
        """
        if df.empty:
            return df
        
        logger.info("Ajout d'indicateurs techniques")
        
        # Moyennes mobiles
        df['sma_5'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=5).mean())
        df['sma_10'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=10).mean())
        df['sma_20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
        df['sma_50'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50).mean())
        df['sma_100'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=100).mean())
        df['sma_200'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=200).mean())
        
        # Moyennes mobiles exponentielles
        df['ema_5'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
        df['ema_10'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
        df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
        df['ema_100'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
        df['ema_200'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())
        
        # RSI
        def calculate_rsi(series, window=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = df.groupby('symbol')['close'].transform(lambda x: calculate_rsi(x, 14))
        
        # MACD
        df['macd'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean() - x.ewm(span=26, adjust=False).mean())
        df['macd_signal'] = df.groupby('symbol')['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bollinger_middle'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).mean())
        df['bollinger_std'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20).std())
        df['bollinger_upper'] = df['bollinger_middle'] + 2 * df['bollinger_std']
        df['bollinger_lower'] = df['bollinger_middle'] - 2 * df['bollinger_std']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
        
        # Stochastique - correction pour éviter l'erreur de longueur
        df['stoch_k'] = np.nan
        df['stoch_d'] = np.nan
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) > 14:  # Vérifier qu'il y a assez de données
                lowest_low = symbol_data['low'].rolling(window=14).min()
                highest_high = symbol_data['high'].rolling(window=14).max()
                k = 100 * ((symbol_data['close'] - lowest_low) / (highest_high - lowest_low))
                d = k.rolling(window=3).mean()
                df.loc[df['symbol'] == symbol, 'stoch_k'] = k
                df.loc[df['symbol'] == symbol, 'stoch_d'] = d
        
        # ATR - correction pour éviter l'erreur de longueur
        df['atr_14'] = np.nan
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) > 1:  # Vérifier qu'il y a assez de données
                tr1 = symbol_data['high'] - symbol_data['low']
                tr2 = abs(symbol_data['high'] - symbol_data['close'].shift())
                tr3 = abs(symbol_data['low'] - symbol_data['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                df.loc[df['symbol'] == symbol, 'atr_14'] = atr
        
        # Ichimoku Cloud - correction pour éviter l'erreur de longueur
        df['ichimoku_tenkan'] = np.nan
        df['ichimoku_kijun'] = np.nan
        df['ichimoku_senkou_a'] = np.nan
        df['ichimoku_senkou_b'] = np.nan
        df['ichimoku_chikou'] = np.nan
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            if len(symbol_data) > 52:  # Vérifier qu'il y a assez de données
                # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
                tenkan_high = symbol_data['high'].rolling(window=9).max()
                tenkan_low = symbol_data['low'].rolling(window=9).min()
                tenkan_sen = (tenkan_high + tenkan_low) / 2
                
                # Kijun-sen (Base Line): (26-period high + 26-period low)/2
                kijun_high = symbol_data['high'].rolling(window=26).max()
                kijun_low = symbol_data['low'].rolling(window=26).min()
                kijun_sen = (kijun_high + kijun_low) / 2
                
                # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                
                # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
                senkou_high = symbol_data['high'].rolling(window=52).max()
                senkou_low = symbol_data['low'].rolling(window=52).min()
                senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
                
                # Chikou Span (Lagging Span): Close price shifted -26 periods
                chikou_span = symbol_data['close'].shift(-26)
                
                df.loc[df['symbol'] == symbol, 'ichimoku_tenkan'] = tenkan_sen
                df.loc[df['symbol'] == symbol, 'ichimoku_kijun'] = kijun_sen
                df.loc[df['symbol'] == symbol, 'ichimoku_senkou_a'] = senkou_span_a
                df.loc[df['symbol'] == symbol, 'ichimoku_senkou_b'] = senkou_span_b
                df.loc[df['symbol'] == symbol, 'ichimoku_chikou'] = chikou_span
        
        # Remplir les valeurs manquantes
        df = df.fillna(0)
        
        return df
