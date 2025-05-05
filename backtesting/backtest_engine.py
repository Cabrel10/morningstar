#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Moteur de backtesting pour le modèle Morningstar.
Permet de simuler le trading sur des données historiques avec les mêmes
conditions que le trading en direct.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import joblib
import time
from datetime import datetime, timedelta
import json
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Imports du projet
from model.architecture.enhanced_hybrid_model import MorningstarModel
from live.live_data_processor import LiveDataProcessor
from utils.data_preparation import CryptoBERTEmbedder
from model.reasoning.reasoning_module import ReasoningModule

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Moteur de backtesting pour simuler le trading avec le modèle Morningstar.
    """
    
    def __init__(self, model_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialise le moteur de backtesting.
        
        Args:
            model_path: Chemin vers le modèle entraîné
            config: Configuration optionnelle pour le backtesting
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        
        # Paramètres de backtesting par défaut
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.position_size = self.config.get('position_size', 0.1)  # 10% du capital par défaut
        self.commission = self.config.get('commission', 0.001)  # 0.1% par défaut
        self.slippage = self.config.get('slippage', 0.0005)  # 0.05% par défaut
        self.use_sl_tp = self.config.get('use_sl_tp', True)  # Utiliser SL/TP par défaut
        
        # Charger le modèle
        logger.info(f"Chargement du modèle depuis {self.model_path}")
        self.model = MorningstarModel.load_model(self.model_path)
        
        # Initialiser le processeur de données
        self.data_processor = LiveDataProcessor(
            model_dir=self.model_path,
            config=self.config,
            cache_dir=self.model_path / 'cache'
        )
        
        # Initialiser le module de raisonnement si disponible
        reasoning_module_path = self.model_path / 'metadata' / 'reasoning_module.pkl'
        if reasoning_module_path.exists():
            logger.info(f"Chargement du module de raisonnement depuis {reasoning_module_path}")
            self.reasoning_module = joblib.load(reasoning_module_path)
        else:
            logger.info("Module de raisonnement non trouvé, initialisation d'un nouveau module")
            self.reasoning_module = ReasoningModule()
        
        # Résultats du backtesting
        self.results = None
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        
        logger.info("Moteur de backtesting initialisé")
    
    def run_backtest(self, data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None, 
                     symbol: str = "BTC/USDT", timeframe: str = "1h") -> Dict[str, Any]:
        """
        Exécute un backtest sur les données fournies.
        
        Args:
            data: DataFrame des données de marché (OHLCV)
            sentiment_data: DataFrame optionnel des données de sentiment
            symbol: Symbole de la paire
            timeframe: Timeframe des données
            
        Returns:
            Dictionnaire des résultats du backtest
        """
        logger.info(f"Démarrage du backtest pour {symbol} sur {len(data)} points de données")
        start_time = time.time()
        
        # Vérifier que les données ont une colonne timestamp
        if 'timestamp' not in data.columns:
            if 'time' in data.columns:
                data = data.rename(columns={'time': 'timestamp'})
            else:
                raise ValueError("Les données doivent avoir une colonne 'timestamp' ou 'time'")
        
        # Convertir timestamp en datetime si ce n'est pas déjà le cas
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Trier les données par timestamp
        data = data.sort_values('timestamp')
        
        # Initialiser les variables de simulation
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        equity_curve = []
        trades = []
        current_sl = None
        current_tp = None
        
        # Créer une copie des données pour stocker les résultats
        results = data.copy()
        results['signal'] = np.nan
        results['position'] = 0.0
        results['equity'] = self.initial_capital
        results['returns'] = 0.0
        results['trade_pnl'] = 0.0
        results['reasoning'] = None
        
        # Fenêtre de données pour le traitement
        window_size = self.config.get('window_size', 100)
        
        # Boucle principale de simulation
        for i in tqdm(range(window_size, len(data)), desc="Backtesting"):
            # Extraire la fenêtre de données courante
            window = data.iloc[i-window_size:i].copy()
            current_row = data.iloc[i]
            current_price = current_row['close']
            current_time = current_row['timestamp']
            
            # Traiter les données pour le modèle
            try:
                # Obtenir les données de sentiment correspondantes si disponibles
                window_sentiment = None
                if sentiment_data is not None:
                    window_sentiment = sentiment_data[
                        (sentiment_data['timestamp'] >= window.iloc[0]['timestamp']) & 
                        (sentiment_data['timestamp'] <= current_time)
                    ].copy()
                
                # Traiter les données pour le modèle
                model_inputs = self.data_processor.process_live_data(window, window_sentiment, symbol)
                
                # Faire une prédiction avec le modèle
                predictions = self.model.predict(model_inputs)
                
                # Post-traiter les prédictions
                processed_predictions = self.data_processor.post_process_predictions(predictions, window)
                
                # Extraire le signal de trading (0=Hold, 1=Buy, 2=Sell)
                if 'signal_class' in processed_predictions:
                    signal = processed_predictions['signal_class'][-1]
                else:
                    signal = np.argmax(processed_predictions['signal'][-1]) if 'signal' in processed_predictions else 0
                
                # Extraire les niveaux SL/TP si disponibles
                sl_price = None
                tp_price = None
                if 'sl_tp' in processed_predictions:
                    sl_pct = processed_predictions['sl_tp'][-1][0]
                    tp_pct = processed_predictions['sl_tp'][-1][1]
                    sl_price = current_price * (1 + sl_pct)
                    tp_price = current_price * (1 + tp_pct)
                
                # Générer une explication si le module de raisonnement est disponible
                reasoning = None
                if self.reasoning_module is not None and hasattr(self.reasoning_module, 'generate_chain_of_thought_explanation'):
                    try:
                        reasoning = self.reasoning_module.generate_chain_of_thought_explanation(
                            window, processed_predictions, 
                            attention_scores=processed_predictions.get('attention_scores')
                        )
                    except Exception as e:
                        logger.warning(f"Erreur lors de la génération de l'explication: {e}")
                
                # Enregistrer le signal dans les résultats
                results.loc[results.index[i], 'signal'] = signal
                if reasoning:
                    results.loc[results.index[i], 'reasoning'] = reasoning
                
                # Vérifier d'abord si SL ou TP a été atteint
                if position != 0 and self.use_sl_tp:
                    # Vérifier si le prix a atteint le SL
                    if current_sl is not None:
                        if (position > 0 and current_price <= current_sl) or (position < 0 and current_price >= current_sl):
                            # SL atteint, fermer la position
                            trade_pnl = position * (current_sl - entry_price) - abs(position * current_sl) * self.commission
                            capital += trade_pnl
                            
                            # Enregistrer le trade
                            trade = {
                                'entry_time': entry_time,
                                'exit_time': current_time,
                                'entry_price': entry_price,
                                'exit_price': current_sl,
                                'position': position,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / (abs(position) * entry_price),
                                'exit_reason': 'stop_loss'
                            }
                            trades.append(trade)
                            
                            # Réinitialiser la position
                            position = 0.0
                            entry_price = 0.0
                            current_sl = None
                            current_tp = None
                    
                    # Vérifier si le prix a atteint le TP
                    elif current_tp is not None:
                        if (position > 0 and current_price >= current_tp) or (position < 0 and current_price <= current_tp):
                            # TP atteint, fermer la position
                            trade_pnl = position * (current_tp - entry_price) - abs(position * current_tp) * self.commission
                            capital += trade_pnl
                            
                            # Enregistrer le trade
                            trade = {
                                'entry_time': entry_time,
                                'exit_time': current_time,
                                'entry_price': entry_price,
                                'exit_price': current_tp,
                                'position': position,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / (abs(position) * entry_price),
                                'exit_reason': 'take_profit'
                            }
                            trades.append(trade)
                            
                            # Réinitialiser la position
                            position = 0.0
                            entry_price = 0.0
                            current_sl = None
                            current_tp = None
                
                # Traiter le signal
                if signal == 1:  # Buy
                    if position <= 0:  # Si pas de position ou short, entrer en long
                        # Fermer la position short si elle existe
                        if position < 0:
                            trade_pnl = position * (current_price - entry_price) - abs(position * current_price) * self.commission
                            capital += trade_pnl
                            
                            # Enregistrer le trade
                            trade = {
                                'entry_time': entry_time,
                                'exit_time': current_time,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'position': position,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / (abs(position) * entry_price),
                                'exit_reason': 'signal'
                            }
                            trades.append(trade)
                        
                        # Calculer la taille de la position
                        position_value = capital * self.position_size
                        position = position_value / current_price
                        
                        # Appliquer le slippage
                        entry_price = current_price * (1 + self.slippage)
                        
                        # Enregistrer l'heure d'entrée
                        entry_time = current_time
                        
                        # Définir SL/TP si disponibles
                        if sl_price is not None and tp_price is not None and self.use_sl_tp:
                            current_sl = sl_price
                            current_tp = tp_price
                
                elif signal == 2:  # Sell
                    if position >= 0:  # Si pas de position ou long, entrer en short
                        # Fermer la position long si elle existe
                        if position > 0:
                            trade_pnl = position * (current_price - entry_price) - abs(position * current_price) * self.commission
                            capital += trade_pnl
                            
                            # Enregistrer le trade
                            trade = {
                                'entry_time': entry_time,
                                'exit_time': current_time,
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'position': position,
                                'pnl': trade_pnl,
                                'pnl_pct': trade_pnl / (abs(position) * entry_price),
                                'exit_reason': 'signal'
                            }
                            trades.append(trade)
                        
                        # Calculer la taille de la position (négative pour short)
                        position_value = capital * self.position_size
                        position = -position_value / current_price
                        
                        # Appliquer le slippage
                        entry_price = current_price * (1 - self.slippage)
                        
                        # Enregistrer l'heure d'entrée
                        entry_time = current_time
                        
                        # Définir SL/TP si disponibles
                        if sl_price is not None and tp_price is not None and self.use_sl_tp:
                            current_sl = sl_price
                            current_tp = tp_price
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement à l'index {i}: {e}")
                traceback.print_exc()
            
            # Calculer la valeur du portefeuille
            if position != 0:
                # Position ouverte, calculer la valeur mark-to-market
                position_value = position * current_price
                equity = capital + position_value
            else:
                # Pas de position, la valeur est le capital
                equity = capital
            
            # Enregistrer la position et l'equity
            results.loc[results.index[i], 'position'] = position
            results.loc[results.index[i], 'equity'] = equity
            
            # Calculer les rendements
            if i > window_size:
                prev_equity = results.loc[results.index[i-1], 'equity']
                if prev_equity > 0:
                    results.loc[results.index[i], 'returns'] = (equity - prev_equity) / prev_equity
            
            # Enregistrer dans la courbe d'equity
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'position': position
            })
        
        # Calculer les métriques de performance
        self.calculate_performance_metrics(results, trades)
        
        # Enregistrer les résultats
        self.results = results
        self.trades = trades
        self.equity_curve = pd.DataFrame(equity_curve)
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        logger.info(f"Backtest terminé en {execution_time:.2f} secondes")
        
        return self.metrics
    
    def calculate_performance_metrics(self, results: pd.DataFrame, trades: List[Dict]) -> Dict[str, float]:
        """
        Calcule les métriques de performance du backtest.
        
        Args:
            results: DataFrame des résultats du backtest
            trades: Liste des trades effectués
            
        Returns:
            Dictionnaire des métriques de performance
        """
        metrics = {}
        
        # Métriques de base
        if len(results) > 0:
            initial_equity = self.initial_capital
            final_equity = results['equity'].iloc[-1]
            
            # Rendement total
            metrics['total_return'] = (final_equity - initial_equity) / initial_equity
            metrics['total_return_pct'] = metrics['total_return'] * 100
            
            # Rendement annualisé
            days = (results['timestamp'].iloc[-1] - results['timestamp'].iloc[0]).days
            if days > 0:
                years = days / 365.0
                metrics['annual_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
                metrics['annual_return_pct'] = metrics['annual_return'] * 100
            
            # Volatilité
            if 'returns' in results.columns:
                daily_returns = results['returns'].resample('D', on='timestamp').sum()
                metrics['volatility'] = daily_returns.std() * (252 ** 0.5)  # Annualisée
                
                # Ratio de Sharpe (si rendement positif)
                if metrics['annual_return'] > 0:
                    risk_free_rate = self.config.get('risk_free_rate', 0.0)
                    metrics['sharpe_ratio'] = (metrics['annual_return'] - risk_free_rate) / metrics['volatility']
                
                # Drawdown
                cumulative_returns = (1 + results['returns']).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns / running_max) - 1
                metrics['max_drawdown'] = drawdown.min()
                metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
        
        # Métriques des trades
        if len(trades) > 0:
            metrics['total_trades'] = len(trades)
            
            # Convertir en DataFrame pour faciliter les calculs
            trades_df = pd.DataFrame(trades)
            
            # Trades gagnants/perdants
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            
            if len(trades) > 0:
                metrics['win_rate'] = len(winning_trades) / len(trades)
                metrics['win_rate_pct'] = metrics['win_rate'] * 100
            
            # Profit moyen
            if len(winning_trades) > 0:
                metrics['avg_profit'] = winning_trades['pnl'].mean()
                metrics['avg_profit_pct'] = winning_trades['pnl_pct'].mean() * 100
            
            # Perte moyenne
            if len(losing_trades) > 0:
                metrics['avg_loss'] = losing_trades['pnl'].mean()
                metrics['avg_loss_pct'] = losing_trades['pnl_pct'].mean() * 100
            
            # Profit/Loss Ratio
            if len(losing_trades) > 0 and len(winning_trades) > 0:
                metrics['profit_loss_ratio'] = abs(winning_trades['pnl'].mean() / losing_trades['pnl'].mean())
            
            # Profit net
            metrics['net_profit'] = trades_df['pnl'].sum()
            metrics['net_profit_pct'] = (metrics['net_profit'] / self.initial_capital) * 100
            
            # Durée moyenne des trades
            trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600  # en heures
            metrics['avg_trade_duration_hours'] = trades_df['duration'].mean()
        
        # Enregistrer les métriques
        self.metrics = metrics
        
        return metrics
    
    def plot_equity_curve(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique de la courbe d'equity.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if self.equity_curve is None:
            raise ValueError("Aucun backtest n'a été exécuté")
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter la courbe d'equity
        fig.add_trace(go.Scatter(
            x=self.equity_curve['timestamp'],
            y=self.equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        # Ajouter les positions
        long_entries = self.equity_curve[self.equity_curve['position'] > 0].copy()
        short_entries = self.equity_curve[self.equity_curve['position'] < 0].copy()
        
        if len(long_entries) > 0:
            fig.add_trace(go.Scatter(
                x=long_entries['timestamp'],
                y=long_entries['equity'],
                mode='markers',
                name='Long',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        if len(short_entries) > 0:
            fig.add_trace(go.Scatter(
                x=short_entries['timestamp'],
                y=short_entries['equity'],
                mode='markers',
                name='Short',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
        
        # Configurer le layout
        fig.update_layout(
            title='Courbe d\'Equity du Backtest',
            xaxis_title='Date',
            yaxis_title='Equity',
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def save_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        Sauvegarde les résultats du backtest.
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire des chemins de fichiers sauvegardés
        """
        if self.results is None:
            raise ValueError("Aucun backtest n'a été exécuté")
        
        # Créer le répertoire de sortie
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les résultats
        results_path = output_dir / 'backtest_results.parquet'
        self.results.to_parquet(results_path)
        
        # Sauvegarder les trades
        trades_path = output_dir / 'backtest_trades.parquet'
        pd.DataFrame(self.trades).to_parquet(trades_path)
        
        # Sauvegarder la courbe d'equity
        equity_path = output_dir / 'equity_curve.parquet'
        self.equity_curve.to_parquet(equity_path)
        
        # Sauvegarder les métriques
        metrics_path = output_dir / 'performance_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Sauvegarder le graphique
        chart_path = output_dir / 'equity_curve.html'
        self.plot_equity_curve(chart_path)
        
        logger.info(f"Résultats du backtest sauvegardés dans {output_dir}")
        
        return {
            'results': results_path,
            'trades': trades_path,
            'equity': equity_path,
            'metrics': metrics_path,
            'chart': chart_path
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Moteur de backtesting pour Morningstar")
    parser.add_argument("--model-dir", required=True, help="Répertoire du modèle")
    parser.add_argument("--data-file", required=True, help="Fichier de données (CSV ou Parquet)")
    parser.add_argument("--sentiment-file", help="Fichier de sentiment (CSV ou Parquet)")
    parser.add_argument("--output-dir", help="Répertoire de sortie pour les résultats")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole de la paire")
    parser.add_argument("--timeframe", default="1h", help="Timeframe des données")
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--position-size", type=float, default=0.1, help="Taille de position (% du capital)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission (en %)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage (en %)")
    parser.add_argument("--use-sl-tp", action="store_true", help="Utiliser les Stop Loss et Take Profit")
    
    args = parser.parse_args()
    
    # Charger les données
    data_path = Path(args.data_file)
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    else:
        data = pd.read_parquet(data_path)
    
    # Charger les données de sentiment si fournies
    sentiment_data = None
    if args.sentiment_file:
        sentiment_path = Path(args.sentiment_file)
        if sentiment_path.suffix == '.csv':
            sentiment_data = pd.read_csv(sentiment_path)
        else:
            sentiment_data = pd.read_parquet(sentiment_path)
    
    # Configuration du backtest
    config = {
        'initial_capital': args.initial_capital,
        'position_size': args.position_size,
        'commission': args.commission,
        'slippage': args.slippage,
        'use_sl_tp': args.use_sl_tp
    }
    
    # Initialiser le moteur de backtesting
    engine = BacktestEngine(Path(args.model_dir), config)
    
    # Exécuter le backtest
    metrics = engine.run_backtest(data, sentiment_data, args.symbol, args.timeframe)
    
    # Afficher les métriques
    print("\n=== Résultats du Backtest ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Sauvegarder les résultats si un répertoire de sortie est spécifié
    if args.output_dir:
        output_paths = engine.save_results(Path(args.output_dir))
        print(f"\nRésultats sauvegardés dans {args.output_dir}")
