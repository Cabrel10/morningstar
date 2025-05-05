#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour exu00e9cuter des backtests du modu00e8le Morningstar en ligne de commande.

Ce script permet d'automatiser les backtests et de les intu00e9grer dans des pipelines
de test sans avoir u00e0 exu00e9cuter le notebook interactif.

Utilisation:
    python run_backtest.py --pair BTC/USDT --model model/saved_model/morningstar_final.h5
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import json
import yaml
from datetime import datetime

# Imports pour backtrader
import backtrader as bt

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

# Du00e9sactiver les avertissements TensorFlow
tf.get_logger().setLevel('ERROR')

# Constantes par du00e9faut
DEFAULT_DATA_DIR = Path("data")
DEFAULT_RESULTS_DIR = Path("results/backtest")
DEFAULT_MODEL_PATH = Path("model/saved_model/morningstar_final.h5")
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_TRANSACTION_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
DEFAULT_SIGNAL_THRESHOLD = 0.6

# Stratu00e9gie de trading pour backtrader
class MorningstarStrategy(bt.Strategy):
    params = (
        ('default_sl_pct', 0.02),  # Stop loss par du00e9faut: 2%
        ('default_tp_pct', 0.04),  # Take profit par du00e9faut: 4%
        ('use_sl_tp', True),       # Utiliser SL/TP
        ('risk_per_trade', 0.02),  # Risque par trade: 2% du capital
    )
    
    def __init__(self):
        # Lignes de donnu00e9es
        self.data_signal = self.datas[0].signal
        self.data_sl_level = self.datas[0].sl_level
        self.data_tp_level = self.datas[0].tp_level
        
        # Variables de suivi
        self.order = None
        self.entry_price = None
        self.sl_order = None
        self.tp_order = None
        
        # Indicateurs
        self.sma20 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=50)
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                
                # Placer SL/TP si activu00e9s
                if self.params.use_sl_tp and not self.sl_order and not self.tp_order:
                    self.place_sl_tp()
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Annuler les ordres SL/TP si position fermu00e9e
                if self.sl_order:
                    self.cancel(self.sl_order)
                    self.sl_order = None
                if self.tp_order:
                    self.cancel(self.tp_order)
                    self.tp_order = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
        
        self.order = None
    
    def place_sl_tp(self):
        # Utiliser les niveaux du modu00e8le si disponibles, sinon utiliser les valeurs par du00e9faut
        if self.data_sl_level[0] > 0:
            sl_level = self.data_sl_level[0]
            tp_level = self.data_tp_level[0]
        else:
            # Utiliser les niveaux par du00e9faut
            sl_level = self.entry_price * (1 - self.params.default_sl_pct)
            tp_level = self.entry_price * (1 + self.params.default_tp_pct)
        
        # Placer les ordres SL/TP
        size = self.position.size
        self.sl_order = self.sell(exectype=bt.Order.Stop, price=sl_level, size=size)
        self.tp_order = self.sell(exectype=bt.Order.Limit, price=tp_level, size=size)
        
        self.log(f'SL plau00e9 u00e0 {sl_level:.2f}, TP plau00e9 u00e0 {tp_level:.2f}')
    
    def next(self):
        # Ne pas prendre de nouvelles positions si un ordre est en attente
        if self.order:
            return
        
        # Signal d'achat
        if not self.position and self.data_signal[0] > 0:
            # Calculer la taille de position basu00e9e sur le risque
            cash = self.broker.getcash()
            risk_amount = cash * self.params.risk_per_trade
            price = self.datas[0].close[0]
            size = risk_amount / price
            
            self.log(f'BUY CREATE, {size:.6f} @ {price:.2f}')
            self.order = self.buy(size=size)
        
        # Signal de vente
        elif self.position and self.data_signal[0] < 0:
            self.log(f'SELL CREATE, {self.position.size:.6f} @ {self.datas[0].close[0]:.2f}')
            self.order = self.sell(size=self.position.size)

# Classe pour les donnu00e9es avec signaux
class SignalData(bt.feeds.PandasData):
    lines = ('signal', 'sl_level', 'tp_level')
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', None),
        ('signal', -1),
        ('sl_level', -1),
        ('tp_level', -1)
    )

def load_data(pair, data_dir=DEFAULT_DATA_DIR):
    """
    Charge les donnu00e9es d'une paire spu00e9cifique.
    
    Args:
        pair: Nom de la paire (ex: 'BTC/USDT')
        data_dir: Ru00e9pertoire des donnu00e9es
        
    Returns:
        DataFrame pandas avec les donnu00e9es OHLCV
    """
    try:
        # Convertir le format de la paire pour le nom de fichier
        pair_filename = pair.replace('/', '').lower()
        
        # Essayer diffu00e9rents formats de nom de fichier
        possible_paths = [
            data_dir / f"{pair_filename}_data.csv",
            data_dir / f"{pair_filename}_data.parquet",
            data_dir / "processed" / f"{pair_filename}_data.csv",
            data_dir / "processed" / f"{pair_filename}_data.parquet"
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Chargement des donnu00e9es depuis {path}")
                
                if path.suffix == '.csv':
                    df = pd.read_csv(path)
                elif path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                else:
                    continue
                
                # Vu00e9rifier et convertir la colonne timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                logger.info(f"Donnu00e9es chargu00e9es avec succu00e8s: {len(df)} lignes")
                return df
        
        logger.error(f"Aucun fichier de donnu00e9es trouvu00e9 pour {pair}")
        return None
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des donnu00e9es: {e}")
        return None

def prepare_features(df, pair):
    """
    Pru00e9pare les features pour le modu00e8le.
    
    Args:
        df: DataFrame avec les donnu00e9es OHLCV
        pair: Nom de la paire
        
    Returns:
        Dictionnaire des features pour le modu00e8le
    """
    try:
        # Pru00e9parer les features techniques
        from utils.feature_engineering import apply_feature_pipeline
        df_features = apply_feature_pipeline(df.reset_index())
        
        # Pru00e9parer les features pour le modu00e8le
        technical_features = df_features.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
        
        # Placeholder pour les embeddings LLM (u00e0 remplacer par les vrais embeddings si disponibles)
        llm_embedding = np.zeros(768, dtype=np.float32)
        
        # Placeholder pour l'input instrument (u00e0 adapter selon votre modu00e8le)
        instrument_input = np.array([0])
        
        # Construire le dictionnaire de features
        features = {
            'technical_input': technical_features.values.astype(np.float32),
            'llm_input': np.tile(llm_embedding, (len(df), 1)),
            'instrument_input': np.tile(instrument_input, (len(df), 1))
        }
        
        return features, df_features
        
    except Exception as e:
        logger.error(f"Erreur lors de la pru00e9paration des features: {e}")
        return None, None

def generate_signals(df, model, features, threshold=DEFAULT_SIGNAL_THRESHOLD):
    """
    Gu00e9nu00e8re les signaux de trading u00e0 partir des pru00e9dictions du modu00e8le.
    
    Args:
        df: DataFrame avec les donnu00e9es OHLCV
        model: Modu00e8le TensorFlow chargu00e9
        features: Features pru00e9paru00e9es pour le modu00e8le
        threshold: Seuil de confiance pour les signaux
        
    Returns:
        DataFrame avec les signaux et niveaux SL/TP
    """
    try:
        # Pru00e9parer les inputs pour le modu00e8le
        model_inputs = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                model_inputs[key] = value
        
        # Faire les pru00e9dictions
        predictions = model.predict(model_inputs)
        
        # Interpru00e9ter les pru00e9dictions selon la structure du modu00e8le
        if isinstance(predictions, list):
            signal_pred = predictions[0]
            sl_tp_pred = predictions[1] if len(predictions) > 1 else None
        else:
            signal_pred = predictions
            sl_tp_pred = None
        
        # Convertir les pru00e9dictions en signaux
        signal_classes = np.argmax(signal_pred, axis=1)
        signal_probs = np.max(signal_pred, axis=1)
        
        # Mapper les classes aux signaux (-1: vente, 0: hold, 1: achat)
        signal_map = {0: -1, 1: 0, 2: 1}  # Adapter selon votre modu00e8le
        signals = np.array([signal_map.get(cls, 0) for cls in signal_classes])
        
        # Appliquer le seuil de confiance
        signals = np.where(signal_probs >= threshold, signals, 0)
        
        # Cru00e9er un DataFrame avec les signaux
        df_signals = df.copy()
        df_signals['signal'] = signals
        
        # Ajouter les niveaux SL/TP si disponibles
        if sl_tp_pred is not None:
            df_signals['sl_level'] = sl_tp_pred[:, 0]
            df_signals['tp_level'] = sl_tp_pred[:, 1]
        else:
            df_signals['sl_level'] = 0.0
            df_signals['tp_level'] = 0.0
        
        return df_signals
        
    except Exception as e:
        logger.error(f"Erreur lors de la gu00e9nu00e9ration des signaux: {e}")
        return None

def run_backtest(df_signals, pair, initial_capital=DEFAULT_INITIAL_CAPITAL, 
                commission=DEFAULT_TRANSACTION_FEE, slippage=DEFAULT_SLIPPAGE):
    """
    Exu00e9cute un backtest avec backtrader.
    
    Args:
        df_signals: DataFrame avec les signaux de trading
        pair: Nom de la paire
        initial_capital: Capital initial
        commission: Frais de transaction
        slippage: Slippage
        
    Returns:
        Ru00e9sultats du backtest
    """
    try:
        # Pru00e9parer les donnu00e9es pour backtrader
        df_bt = df_signals.reset_index().copy()
        df_bt.rename(columns={'timestamp': 'datetime'}, inplace=True)
        
        # Cru00e9er une instance Cerebro
        cerebro = bt.Cerebro()
        
        # Ajouter les donnu00e9es
        data = SignalData(
            dataname=df_bt,
            datetime='datetime',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            signal='signal',
            sl_level='sl_level',
            tp_level='tp_level'
        )
        cerebro.adddata(data)
        
        # Ajouter la stratu00e9gie
        cerebro.addstrategy(MorningstarStrategy)
        
        # Configurer le broker
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=commission)  # 0.1%
        
        # Ajouter les analyseurs
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Exu00e9cuter le backtest
        logger.info(f"Exu00e9cution du backtest pour {pair}...")
        results = cerebro.run()
        strat = results[0]
        
        # Ru00e9cupu00e9rer les ru00e9sultats
        final_value = cerebro.broker.getvalue()
        pnl = final_value - initial_capital
        roi = (final_value / initial_capital - 1) * 100
        
        # Ru00e9cupu00e9rer les mu00e9triques des analyseurs
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)
        trades = strat.analyzers.trades.get_analysis()
        
        # Calculer les statistiques de trading
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0.0
        
        # Afficher les ru00e9sultats
        logger.info(f"\nRu00e9sultats du backtest pour {pair}:")
        logger.info(f"Capital initial: {initial_capital:.2f}")
        logger.info(f"Valeur finale: {final_value:.2f}")
        logger.info(f"P&L: {pnl:.2f} ({roi:.2f}%)")
        logger.info(f"Ratio de Sharpe: {sharpe:.2f}")
        logger.info(f"Drawdown maximum: {drawdown:.2f}%")
        logger.info(f"Nombre total de trades: {total_trades}")
        logger.info(f"Trades gagnants: {won_trades} ({win_rate*100:.2f}%)")
        logger.info(f"Trades perdants: {lost_trades}")
        
        # Ru00e9sultats du backtest
        backtest_results = {
            'pair': pair,
            'initial_capital': initial_capital,
            'final_value': float(final_value),
            'pnl': float(pnl),
            'roi': float(roi),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(drawdown),
            'total_trades': int(total_trades),
            'won_trades': int(won_trades),
            'lost_trades': int(lost_trades),
            'win_rate': float(win_rate),
            'timestamp': datetime.now().isoformat()
        }
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exu00e9cution du backtest: {e}")
        return None

def save_results(results, pair, results_dir=DEFAULT_RESULTS_DIR):
    """
    Sauvegarde les ru00e9sultats du backtest.
    
    Args:
        results: Ru00e9sultats du backtest
        pair: Nom de la paire
        results_dir: Ru00e9pertoire des ru00e9sultats
    """
    try:
        # Cru00e9er le ru00e9pertoire si nu00e9cessaire
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Nom du fichier de ru00e9sultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{pair.replace('/', '')}_backtest_{timestamp}.json"
        
        # Sauvegarder les ru00e9sultats
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ru00e9sultats sauvegardus dans {results_file}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des ru00e9sultats: {e}")

def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description="Backtest du modu00e8le Morningstar")
    parser.add_argument("--pair", type=str, required=True, help="Paire u00e0 tester (ex: BTC/USDT)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Chemin vers le modu00e8le")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Ru00e9pertoire des donnu00e9es")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Ru00e9pertoire des ru00e9sultats")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="Capital initial")
    parser.add_argument("--commission", type=float, default=DEFAULT_TRANSACTION_FEE, help="Frais de transaction")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE, help="Slippage")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIGNAL_THRESHOLD, help="Seuil de confiance pour les signaux")
    
    args = parser.parse_args()
    
    try:
        # Charger le modu00e8le
        logger.info(f"Chargement du modu00e8le depuis {args.model}...")
        model = tf.keras.models.load_model(args.model)
        logger.info("Modu00e8le chargu00e9 avec succu00e8s")
        
        # Charger les donnu00e9es
        df = load_data(args.pair, Path(args.data_dir))
        if df is None:
            logger.error(f"Impossible de charger les donnu00e9es pour {args.pair}")
            return 1
        
        # Pru00e9parer les features
        features, df_features = prepare_features(df, args.pair)
        if features is None:
            logger.error(f"Impossible de pru00e9parer les features pour {args.pair}")
            return 1
        
        # Gu00e9nu00e9rer les signaux
        df_signals = generate_signals(df, model, features, args.threshold)
        if df_signals is None:
            logger.error(f"Impossible de gu00e9nu00e9rer les signaux pour {args.pair}")
            return 1
        
        # Exu00e9cuter le backtest
        results = run_backtest(
            df_signals, 
            args.pair, 
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage
        )
        if results is None:
            logger.error(f"Erreur lors de l'exu00e9cution du backtest pour {args.pair}")
            return 1
        
        # Sauvegarder les ru00e9sultats
        save_results(results, args.pair, args.results_dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exu00e9cution du backtest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
