#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de visualisation pour les résultats de backtesting du modèle Morningstar.
Génère des graphiques détaillés et des analyses des performances.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Ajouter le répertoire du projet au PYTHONPATH
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(BASE_DIR))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestVisualizer:
    """
    Classe pour visualiser les résultats de backtesting.
    """
    
    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialise le visualiseur avec le répertoire des résultats de backtesting.
        
        Args:
            results_dir: Répertoire contenant les résultats du backtest
        """
        self.results_dir = Path(results_dir)
        
        # Charger les résultats
        self.results = self._load_results()
        self.trades = self._load_trades()
        self.equity_curve = self._load_equity_curve()
        self.metrics = self._load_metrics()
        
        # Définir les couleurs
        self.colors = {
            'equity': '#1f77b4',
            'price': '#2ca02c',
            'long': '#2ca02c',
            'short': '#d62728',
            'sl': '#ff7f0e',
            'tp': '#9467bd',
            'signal': '#8c564b'
        }
        
        logger.info(f"Visualiseur initialisé avec les résultats de {self.results_dir}")
    
    def _load_results(self) -> pd.DataFrame:
        """
        Charge les résultats du backtest.
        
        Returns:
            DataFrame des résultats
        """
        results_path = self.results_dir / 'backtest_results.parquet'
        if not results_path.exists():
            raise FileNotFoundError(f"Fichier de résultats non trouvé: {results_path}")
        
        return pd.read_parquet(results_path)
    
    def _load_trades(self) -> pd.DataFrame:
        """
        Charge les trades du backtest.
        
        Returns:
            DataFrame des trades
        """
        trades_path = self.results_dir / 'backtest_trades.parquet'
        if not trades_path.exists():
            raise FileNotFoundError(f"Fichier de trades non trouvé: {trades_path}")
        
        return pd.read_parquet(trades_path)
    
    def _load_equity_curve(self) -> pd.DataFrame:
        """
        Charge la courbe d'equity du backtest.
        
        Returns:
            DataFrame de la courbe d'equity
        """
        equity_path = self.results_dir / 'equity_curve.parquet'
        if not equity_path.exists():
            raise FileNotFoundError(f"Fichier de courbe d'equity non trouvé: {equity_path}")
        
        return pd.read_parquet(equity_path)
    
    def _load_metrics(self) -> Dict[str, Any]:
        """
        Charge les métriques de performance du backtest.
        
        Returns:
            Dictionnaire des métriques
        """
        metrics_path = self.results_dir / 'performance_metrics.json'
        if not metrics_path.exists():
            raise FileNotFoundError(f"Fichier de métriques non trouvé: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    def plot_equity_curve(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique de la courbe d'equity.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter la courbe d'equity
        fig.add_trace(go.Scatter(
            x=self.equity_curve['timestamp'],
            y=self.equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color=self.colors['equity'], width=2)
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
                marker=dict(color=self.colors['long'], size=8, symbol='triangle-up')
            ))
        
        if len(short_entries) > 0:
            fig.add_trace(go.Scatter(
                x=short_entries['timestamp'],
                y=short_entries['equity'],
                mode='markers',
                name='Short',
                marker=dict(color=self.colors['short'], size=8, symbol='triangle-down')
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
    
    def plot_drawdown(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique du drawdown.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        # Calculer le drawdown
        equity = self.equity_curve['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        
        # Créer un DataFrame pour le drawdown
        drawdown_df = pd.DataFrame({
            'timestamp': self.equity_curve['timestamp'],
            'drawdown': drawdown
        })
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter la courbe de drawdown
        fig.add_trace(go.Scatter(
            x=drawdown_df['timestamp'],
            y=drawdown_df['drawdown'] * 100,  # En pourcentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        # Configurer le layout
        fig.update_layout(
            title='Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(tickformat='.2f')
        )
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def plot_trades_distribution(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique de la distribution des trades.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if len(self.trades) == 0:
            logger.warning("Aucun trade à afficher")
            return None
        
        # Créer la figure avec 2 sous-graphiques
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Distribution des Profits/Pertes', 'Histogramme des Rendements'),
            vertical_spacing=0.2
        )
        
        # Distribution des P&L
        fig.add_trace(
            go.Histogram(
                x=self.trades['pnl'],
                name='P&L',
                marker_color=self.colors['equity'],
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=1
        )
        
        # Histogramme des rendements en pourcentage
        fig.add_trace(
            go.Histogram(
                x=self.trades['pnl_pct'] * 100,  # En pourcentage
                name='Rendement (%)',
                marker_color=self.colors['price'],
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # Configurer le layout
        fig.update_layout(
            title='Distribution des Trades',
            template='plotly_white',
            showlegend=False,
            height=800
        )
        
        fig.update_xaxes(title_text='P&L', row=1, col=1)
        fig.update_xaxes(title_text='Rendement (%)', row=2, col=1)
        
        fig.update_yaxes(title_text='Nombre de Trades', row=1, col=1)
        fig.update_yaxes(title_text='Nombre de Trades', row=2, col=1)
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def plot_monthly_returns(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique des rendements mensuels.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        # Calculer les rendements journaliers
        self.equity_curve['date'] = self.equity_curve['timestamp'].dt.date
        daily_equity = self.equity_curve.groupby('date')['equity'].last().reset_index()
        daily_equity['date'] = pd.to_datetime(daily_equity['date'])
        daily_equity['returns'] = daily_equity['equity'].pct_change()
        
        # Calculer les rendements mensuels
        daily_equity['year_month'] = daily_equity['date'].dt.strftime('%Y-%m')
        monthly_returns = daily_equity.groupby('year_month').apply(
            lambda x: (x['equity'].iloc[-1] / x['equity'].iloc[0]) - 1
        ).reset_index()
        monthly_returns.columns = ['year_month', 'returns']
        
        # Convertir en pourcentage
        monthly_returns['returns_pct'] = monthly_returns['returns'] * 100
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter les rendements mensuels
        colors = ['green' if r >= 0 else 'red' for r in monthly_returns['returns']]
        
        fig.add_trace(go.Bar(
            x=monthly_returns['year_month'],
            y=monthly_returns['returns_pct'],
            marker_color=colors,
            name='Rendement Mensuel'
        ))
        
        # Configurer le layout
        fig.update_layout(
            title='Rendements Mensuels',
            xaxis_title='Mois',
            yaxis_title='Rendement (%)',
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(tickformat='.2f')
        )
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def plot_trade_analysis(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique d'analyse des trades.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        if len(self.trades) == 0:
            logger.warning("Aucun trade à afficher")
            return None
        
        # Créer la figure avec 2 sous-graphiques
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Répartition des Trades', 
                'Raisons de Sortie',
                'Durée des Trades',
                'Évolution du P&L Cumulé'
            ),
            specs=[
                [{"type": "pie"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # Répartition des trades gagnants/perdants
        winning_trades = self.trades[self.trades['pnl'] > 0]
        losing_trades = self.trades[self.trades['pnl'] < 0]
        
        fig.add_trace(
            go.Pie(
                labels=['Gagnants', 'Perdants'],
                values=[len(winning_trades), len(losing_trades)],
                marker_colors=['green', 'red'],
                textinfo='percent+label',
                hole=0.4
            ),
            row=1, col=1
        )
        
        # Raisons de sortie
        exit_reasons = self.trades['exit_reason'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=exit_reasons.index,
                values=exit_reasons.values,
                textinfo='percent+label',
                hole=0.4
            ),
            row=1, col=2
        )
        
        # Durée des trades
        self.trades['duration_hours'] = (self.trades['exit_time'] - self.trades['entry_time']).dt.total_seconds() / 3600
        duration_bins = [0, 1, 6, 12, 24, 48, 72, 168, float('inf')]
        duration_labels = ['<1h', '1-6h', '6-12h', '12-24h', '1-2j', '2-3j', '3-7j', '>7j']
        
        self.trades['duration_category'] = pd.cut(
            self.trades['duration_hours'], 
            bins=duration_bins, 
            labels=duration_labels
        )
        
        duration_counts = self.trades['duration_category'].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=duration_counts.index,
                y=duration_counts.values,
                marker_color=self.colors['equity'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Évolution du P&L cumulé
        self.trades = self.trades.sort_values('entry_time')
        self.trades['cumulative_pnl'] = self.trades['pnl'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=self.trades['exit_time'],
                y=self.trades['cumulative_pnl'],
                mode='lines',
                line=dict(color=self.colors['equity'], width=2)
            ),
            row=2, col=2
        )
        
        # Configurer le layout
        fig.update_layout(
            title='Analyse des Trades',
            template='plotly_white',
            showlegend=False,
            height=800
        )
        
        fig.update_xaxes(title_text='Durée', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=2)
        
        fig.update_yaxes(title_text='Nombre de Trades', row=2, col=1)
        fig.update_yaxes(title_text='P&L Cumulé', row=2, col=2)
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def plot_price_with_signals(self, output_path: Optional[Path] = None) -> go.Figure:
        """
        Génère un graphique du prix avec les signaux de trading.
        
        Args:
            output_path: Chemin de sortie pour sauvegarder le graphique
            
        Returns:
            Figure Plotly
        """
        # Créer la figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Prix et Signaux', 'Position'),
            row_heights=[0.7, 0.3]
        )
        
        # Ajouter le graphique des prix
        fig.add_trace(
            go.Scatter(
                x=self.results['timestamp'],
                y=self.results['close'],
                mode='lines',
                name='Prix',
                line=dict(color=self.colors['price'], width=2)
            ),
            row=1, col=1
        )
        
        # Ajouter les signaux d'achat
        buy_signals = self.results[self.results['signal'] == 1]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals['timestamp'],
                    y=buy_signals['close'],
                    mode='markers',
                    name='Achat',
                    marker=dict(color=self.colors['long'], size=10, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Ajouter les signaux de vente
        sell_signals = self.results[self.results['signal'] == 2]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals['timestamp'],
                    y=sell_signals['close'],
                    mode='markers',
                    name='Vente',
                    marker=dict(color=self.colors['short'], size=10, symbol='triangle-down')
                ),
                row=1, col=1
            )
        
        # Ajouter les positions
        fig.add_trace(
            go.Scatter(
                x=self.results['timestamp'],
                y=self.results['position'],
                mode='lines',
                name='Position',
                line=dict(color=self.colors['signal'], width=2)
            ),
            row=2, col=1
        )
        
        # Configurer le layout
        fig.update_layout(
            title='Prix et Signaux de Trading',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=800
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Prix', row=1, col=1)
        fig.update_yaxes(title_text='Position', row=2, col=1)
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Graphique sauvegardé dans {output_path}")
        
        return fig
    
    def generate_performance_report(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Génère un rapport complet de performance avec tous les graphiques.
        
        Args:
            output_dir: Répertoire de sortie pour sauvegarder les graphiques
            
        Returns:
            Dictionnaire des chemins des graphiques générés
        """
        if output_dir is None:
            output_dir = self.results_dir / 'visualizations'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer tous les graphiques
        paths = {}
        
        # Courbe d'equity
        equity_path = output_dir / 'equity_curve.html'
        self.plot_equity_curve(equity_path)
        paths['equity_curve'] = equity_path
        
        # Drawdown
        drawdown_path = output_dir / 'drawdown.html'
        self.plot_drawdown(drawdown_path)
        paths['drawdown'] = drawdown_path
        
        # Distribution des trades
        trades_dist_path = output_dir / 'trades_distribution.html'
        self.plot_trades_distribution(trades_dist_path)
        paths['trades_distribution'] = trades_dist_path
        
        # Rendements mensuels
        monthly_returns_path = output_dir / 'monthly_returns.html'
        self.plot_monthly_returns(monthly_returns_path)
        paths['monthly_returns'] = monthly_returns_path
        
        # Analyse des trades
        trade_analysis_path = output_dir / 'trade_analysis.html'
        self.plot_trade_analysis(trade_analysis_path)
        paths['trade_analysis'] = trade_analysis_path
        
        # Prix et signaux
        price_signals_path = output_dir / 'price_signals.html'
        self.plot_price_with_signals(price_signals_path)
        paths['price_signals'] = price_signals_path
        
        # Générer un rapport HTML
        report_path = output_dir / 'performance_report.html'
        self._generate_html_report(report_path, paths)
        paths['report'] = report_path
        
        logger.info(f"Rapport de performance généré dans {output_dir}")
        
        return paths
    
    def _generate_html_report(self, output_path: Path, graph_paths: Dict[str, Path]) -> None:
        """
        Génère un rapport HTML avec tous les graphiques et métriques.
        
        Args:
            output_path: Chemin de sortie pour le rapport HTML
            graph_paths: Dictionnaire des chemins des graphiques générés
        """
        # Créer le contenu HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport de Performance Morningstar</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 20px;
                }}
                .metric-card {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    flex: 1 0 200px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.05);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .metric-name {{
                    color: #666;
                    font-size: 14px;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart-container {{
                    margin: 20px 0;
                }}
                iframe {{
                    width: 100%;
                    height: 600px;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Rapport de Performance Morningstar</h1>
                <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Métriques de Performance</h2>
                <div class="metrics-container">
        """
        
        # Ajouter les métriques
        for key, value in self.metrics.items():
            # Formater la valeur
            if isinstance(value, float):
                if key.endswith('_pct') or key.endswith('rate_pct'):
                    formatted_value = f"{value:.2f}%"
                    css_class = "positive" if value > 0 else "negative" if value < 0 else ""
                else:
                    formatted_value = f"{value:.4f}"
                    css_class = "positive" if value > 0 else "negative" if value < 0 else ""
            else:
                formatted_value = str(value)
                css_class = ""
            
            # Formater le nom
            display_name = key.replace('_', ' ').title()
            
            html_content += f"""
                    <div class="metric-card">
                        <div class="metric-name">{display_name}</div>
                        <div class="metric-value {css_class}">{formatted_value}</div>
                    </div>
            """
        
        html_content += """
                </div>
                
                <h2>Graphiques</h2>
        """
        
        # Ajouter les graphiques
        for name, path in graph_paths.items():
            if name != 'report':
                display_name = name.replace('_', ' ').title()
                relative_path = os.path.relpath(path, output_path.parent)
                
                html_content += f"""
                <div class="chart-container">
                    <h3>{display_name}</h3>
                    <iframe src="{relative_path}"></iframe>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Écrire le fichier HTML
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML généré: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualiseur de résultats de backtesting pour Morningstar")
    parser.add_argument("--results-dir", required=True, help="Répertoire des résultats de backtesting")
    parser.add_argument("--output-dir", help="Répertoire de sortie pour les visualisations")
    
    args = parser.parse_args()
    
    # Initialiser le visualiseur
    visualizer = BacktestVisualizer(Path(args.results_dir))
    
    # Générer le rapport de performance
    output_dir = args.output_dir if args.output_dir else Path(args.results_dir) / 'visualizations'
    paths = visualizer.generate_performance_report(Path(output_dir))
    
    print(f"Rapport de performance généré dans {output_dir}")
    print(f"Rapport HTML: {paths['report']}")
