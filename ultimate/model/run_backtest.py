#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de backtest pour le modèle monolithique Morningstar.

Ce script permet de:
1. Charger un modèle monolithique entraîné
2. Préparer les données pour le backtest
3. Générer des signaux de trading
4. Exécuter le backtest avec Backtrader
5. Produire et sauvegarder les métriques et visualisations
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional, Any, Union

# Ajouter le chemin du projet au PYTHONPATH
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Ajustez selon votre structure
sys.path.append(str(PROJECT_ROOT))

# Import du modèle monolithique
from monolith_model import MonolithModel

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_backtest")

# Importations spécifiques à Backtrader
try:
    import backtrader as bt
except ImportError:
    logger.warning("Backtrader non installé. Les fonctionnalités de backtest seront limitées.")


class MonolithStrategy(bt.Strategy):
    """Stratégie Backtrader basée sur les signaux du modèle monolithique."""
    
    params = (
        ('signal_file', None),  # Fichier avec les signaux pré-calculés
        ('sl_tp_file', None),   # Fichier avec les niveaux SL/TP pré-calculés
        ('risk_per_trade', 0.02),  # Risque par trade (% du capital)
        ('verbose', False),  # Affichage détaillé des trades
    )
    
    def __init__(self):
        """Initialisation de la stratégie."""
        self.orders = {}  # Suivi des ordres actifs
        self.signals_df = None
        self.sltp_df = None
        
        # Charger les signaux pré-calculés
        if self.p.signal_file:
            self.signals_df = pd.read_csv(self.p.signal_file, index_col=0, parse_dates=True)
            logger.info(f"Signaux chargés: {len(self.signals_df)} entrées")
        
        # Charger les SL/TP pré-calculés
        if self.p.sl_tp_file:
            self.sltp_df = pd.read_csv(self.p.sl_tp_file, index_col=0, parse_dates=True)
            logger.info(f"Niveaux SL/TP chargés: {len(self.sltp_df)} entrées")
    
    def next(self):
        """
        Méthode appelée pour chaque barre de données.
        Vérifie les signaux et exécute les trades correspondants.
        """
        current_date = self.data.datetime.date(0)
        
        # Sortir si pas de signaux
        if self.signals_df is None:
            return
        
        # Vérifier si nous avons un signal pour cette date
        if current_date not in self.signals_df.index:
            return
        
        # Récupérer le signal pour cette date
        signal = self.signals_df.loc[current_date, 'signal']
        
        # Récupérer les SL/TP si disponibles
        sl_level = None
        tp_level = None
        
        if self.sltp_df is not None and current_date in self.sltp_df.index:
            sl_level = self.sltp_df.loc[current_date, 'sl']
            tp_level = self.sltp_df.loc[current_date, 'tp']
        
        # Fermer les positions existantes si signal inverse
        self.close_positions_if_needed(signal)
        
        # Ouvrir de nouvelles positions si signal
        self.open_position_if_signal(signal, sl_level, tp_level)
    
    def close_positions_if_needed(self, signal):
        """Ferme les positions si le signal est inverse."""
        # Fermer position longue si signal de vente
        if self.position.size > 0 and signal < 0:
            self.close()
            if self.p.verbose:
                logger.info(f"CLOSE LONG: {self.data.datetime.date(0)}, Prix: {self.data.close[0]}")
        
        # Fermer position courte si signal d'achat
        elif self.position.size < 0 and signal > 0:
            self.close()
            if self.p.verbose:
                logger.info(f"CLOSE SHORT: {self.data.datetime.date(0)}, Prix: {self.data.close[0]}")
    
    def open_position_if_signal(self, signal, sl_level=None, tp_level=None):
        """Ouvre une position si un signal est présent."""
        # Ne rien faire si déjà en position ou pas de signal
        if self.position.size != 0 or signal == 0:
            return
        
        # Calcul de la taille de position basée sur le risque
        price = self.data.close[0]
        stop_loss = sl_level if sl_level is not None else price * 0.95 if signal > 0 else price * 1.05
        size = self.calculate_position_size(price, stop_loss)
        
        # Ouvrir une position longue si signal d'achat
        if signal > 0:
            self.buy(size=size)
            
            # Placer les ordres SL/TP si spécifiés
            if sl_level is not None:
                self.sell(size=size, price=sl_level, exectype=bt.Order.Stop)
            
            if tp_level is not None:
                self.sell(size=size, price=tp_level, exectype=bt.Order.Limit)
            
            if self.p.verbose:
                logger.info(f"BUY: {self.data.datetime.date(0)}, Prix: {price}, Size: {size}")
                if sl_level: logger.info(f"  SL: {sl_level}")
                if tp_level: logger.info(f"  TP: {tp_level}")
        
        # Ouvrir une position courte si signal de vente
        elif signal < 0:
            self.sell(size=size)
            
            # Placer les ordres SL/TP si spécifiés
            if sl_level is not None:
                self.buy(size=size, price=sl_level, exectype=bt.Order.Stop)
            
            if tp_level is not None:
                self.buy(size=size, price=tp_level, exectype=bt.Order.Limit)
            
            if self.p.verbose:
                logger.info(f"SELL: {self.data.datetime.date(0)}, Prix: {price}, Size: {size}")
                if sl_level: logger.info(f"  SL: {sl_level}")
                if tp_level: logger.info(f"  TP: {tp_level}")
    
    def calculate_position_size(self, price, stop_loss):
        """Calcule la taille de position basée sur le risque."""
        capital = self.broker.getvalue()
        risk_amount = capital * self.p.risk_per_trade
        point_risk = abs(price - stop_loss)
        
        if point_risk > 0:
            size = risk_amount / point_risk
        else:
            size = 0
        
        return size


def load_data_for_backtest(
    data_path: str,
    features_needed: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Charge les données nécessaires pour le backtest.
    
    Args:
        data_path: Chemin vers le fichier de données
        features_needed: Liste des colonnes nécessaires pour le modèle
        start_date: Date de début (optionnel)
        end_date: Date de fin (optionnel)
        
    Returns:
        DataFrame avec les données préparées pour le backtest
    """
    logger.info(f"Chargement des données depuis {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    # Filtrer par date si spécifié
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    # Vérifier que toutes les colonnes nécessaires sont présentes
    missing_cols = [col for col in features_needed if col not in df.columns]
    if missing_cols:
        logger.warning(f"Colonnes manquantes: {missing_cols}")
    
    logger.info(f"Données chargées: {df.shape[0]} échantillons, {df.shape[1]} colonnes")
    return df


def generate_signals(
    model: MonolithModel,
    df: pd.DataFrame,
    tech_cols: List[str],
    embedding_cols: List[str],
    mcp_cols: List[str],
    instrument_col: str,
    sequence_length: Optional[int] = None,
    instrument_map: Optional[Dict[str, int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Génère les signaux de trading et les niveaux SL/TP à partir du modèle.
    
    Args:
        model: Modèle monolithique
        df: DataFrame des données
        tech_cols: Liste des colonnes techniques
        embedding_cols: Liste des colonnes d'embeddings
        mcp_cols: Liste des colonnes MCP
        instrument_col: Nom de la colonne d'instrument
        sequence_length: Longueur de séquence (optionnel)
        instrument_map: Mapping des instruments vers entiers
        
    Returns:
        DataFrame des signaux et DataFrame des niveaux SL/TP
    """
    logger.info("Génération des signaux de trading")
    
    # Si pas de mapping fourni, créer un nouveau
    if instrument_map is None:
        instruments = df[instrument_col].unique()
        instrument_map = {inst: i for i, inst in enumerate(instruments)}
    
    # Encoder les instruments
    df['instrument_encoded'] = df[instrument_col].map(instrument_map)
    
    # Préparer les features
    X_tech = df[tech_cols].values
    X_emb = df[embedding_cols].values if embedding_cols else np.zeros((len(df), 1))
    X_mcp = df[mcp_cols].values if mcp_cols else np.zeros((len(df), 1))
    X_inst = df['instrument_encoded'].values.reshape(-1, 1)
    
    # Créer des séquences si nécessaire
    if sequence_length is not None:
        # Fonction pour créer des séquences
        def create_sequences(data, seq_length):
            n = len(data)
            seq_data = []
            for i in range(n - seq_length + 1):
                seq_data.append(data[i:i+seq_length])
            return np.array(seq_data)
        
        X_tech_seq = create_sequences(X_tech, sequence_length)
        
        # Ajuster les autres entrées pour correspondre aux séquences
        X_emb_seq = X_emb[sequence_length-1:]
        X_mcp_seq = X_mcp[sequence_length-1:]
        X_inst_seq = X_inst[sequence_length-1:]
        
        # Ajuster l'index du DataFrame pour correspondre aux séquences
        new_index = df.index[sequence_length-1:]
        
        # Préparer les données pour la prédiction
        inputs = {
            "technical_input": X_tech_seq,
            "embeddings_input": X_emb_seq,
            "mcp_input": X_mcp_seq,
            "instrument_input": X_inst_seq
        }
    else:
        # Cas non-séquentiel
        inputs = {
            "technical_input": X_tech,
            "embeddings_input": X_emb,
            "mcp_input": X_mcp,
            "instrument_input": X_inst
        }
        new_index = df.index
    
    # Effectuer la prédiction
    predictions = model.predict(inputs)
    
    # Traiter les signaux (signal_output)
    if "signal_output" in predictions:
        signal_probs = predictions["signal_output"]
        
        # Convertir les probabilités en classes
        signal_class = np.argmax(signal_probs, axis=1)
        
        # Convertir les classes en {-1, 0, 1} pour {sell, neutral, buy}
        signal_map = {0: -1, 1: 0, 2: 1}
        signals = np.array([signal_map[c] for c in signal_class])
        
        # Créer le DataFrame des signaux
        signals_df = pd.DataFrame(
            {"signal": signals, "sell_prob": signal_probs[:, 0], 
             "neutral_prob": signal_probs[:, 1], "buy_prob": signal_probs[:, 2]},
            index=new_index
        )
    else:
        signals_df = pd.DataFrame({"signal": np.zeros(len(new_index))}, index=new_index)
    
    # Traiter les niveaux SL/TP (sl_tp_output)
    if "sl_tp_output" in predictions:
        sl_tp_values = predictions["sl_tp_output"]
        
        # Créer le DataFrame des niveaux SL/TP
        sltp_df = pd.DataFrame(
            {"sl": sl_tp_values[:, 0], "tp": sl_tp_values[:, 1]},
            index=new_index
        )
        
        # Appliquer les niveaux SL/TP aux prix actuels
        for i in range(len(new_index)):
            current_price = df.loc[new_index[i], "close"]
            signal = signals_df.iloc[i]["signal"]
            
            # Pour signal d'achat: SL en dessous, TP au-dessus
            if signal > 0:
                sl_pct = 1 - sl_tp_values[i, 0]  # % en dessous
                tp_pct = 1 + sl_tp_values[i, 1]  # % au-dessus
            # Pour signal de vente: SL au-dessus, TP en dessous
            elif signal < 0:
                sl_pct = 1 + sl_tp_values[i, 0]  # % au-dessus
                tp_pct = 1 - sl_tp_values[i, 1]  # % en dessous
            # Pour signal neutre: pas de SL/TP
            else:
                sl_pct = tp_pct = 1.0
            
            sltp_df.iloc[i, 0] = current_price * sl_pct  # SL
            sltp_df.iloc[i, 1] = current_price * tp_pct  # TP
    else:
        sltp_df = None
    
    logger.info(f"Signaux générés: {len(signals_df)} entrées")
    if sltp_df is not None:
        logger.info(f"Niveaux SL/TP générés: {len(sltp_df)} entrées")
    
    return signals_df, sltp_df


def run_backtest(
    price_data: pd.DataFrame,
    signals_df: pd.DataFrame,
    sltp_df: Optional[pd.DataFrame] = None,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    risk_per_trade: float = 0.02,
    output_dir: str = "./backtest_results"
) -> Dict[str, Any]:
    """
    Exécute le backtest avec Backtrader.
    
    Args:
        price_data: DataFrame avec les données OHLCV
        signals_df: DataFrame avec les signaux
        sltp_df: DataFrame avec les niveaux SL/TP (optionnel)
        initial_cash: Capital initial
        commission: Commission par trade
        risk_per_trade: Risque par trade (% du capital)
        output_dir: Répertoire de sortie pour les résultats
        
    Returns:
        Dictionnaire des métriques de performance
    """
    if "backtrader" not in sys.modules:
        logger.error("Backtrader n'est pas installé. Impossible d'exécuter le backtest.")
        return {"error": "Backtrader not installed"}
    
    logger.info(f"Démarrage du backtest avec capital initial de {initial_cash}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder les signaux et SL/TP pour le backtest
    signals_path = os.path.join(output_dir, "signals.csv")
    signals_df.to_csv(signals_path)
    
    sltp_path = None
    if sltp_df is not None:
        sltp_path = os.path.join(output_dir, "sltp.csv")
        sltp_df.to_csv(sltp_path)
    
    # Préparer les données pour Backtrader
    data = bt.feeds.PandasData(
        dataname=price_data,
        datetime=None,  # Index est déjà datetime
        open=0,
        high=1,
        low=2,
        close=3,
        volume=4,
        openinterest=-1
    )
    
    # Créer le cerveau Backtrader
    cerebro = bt.Cerebro()
    
    # Ajouter les données
    cerebro.adddata(data)
    
    # Ajouter la stratégie
    cerebro.addstrategy(
        MonolithStrategy,
        signal_file=signals_path,
        sl_tp_file=sltp_path,
        risk_per_trade=risk_per_trade,
        verbose=False
    )
    
    # Configurer le broker
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Ajouter les analyseurs
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    # Exécuter le backtest
    logger.info("Exécution du backtest...")
    results = cerebro.run()
    
    # Récupérer les résultats
    strat = results[0]
    
    # Collecter les métriques
    metrics = {
        "initial_capital": initial_cash,
        "final_capital": cerebro.broker.getvalue(),
        "return_pct": (cerebro.broker.getvalue() / initial_cash - 1) * 100,
        "sharpe_ratio": getattr(strat.analyzers.sharpe, "ratio", 0),
        "max_drawdown_pct": getattr(strat.analyzers.drawdown, "maxdrawdown", 0) * 100,
        "total_trades": getattr(strat.analyzers.trades, "total", {}).get("total", 0),
        "win_trades": getattr(strat.analyzers.trades, "won", {}).get("total", 0),
        "loss_trades": getattr(strat.analyzers.trades, "lost", {}).get("total", 0)
    }
    
    if metrics["total_trades"] > 0:
        metrics["win_rate"] = metrics["win_trades"] / metrics["total_trades"] * 100
    else:
        metrics["win_rate"] = 0
    
    # Sauvegarder les métriques
    with open(os.path.join(output_dir, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Générer un plot
    try:
        plt.figure(figsize=(10, 6))
        cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
        plt.savefig(os.path.join(output_dir, "backtest_plot.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"Impossible de générer le plot: {e}")
    
    logger.info(f"Backtest terminé. Capital final: {metrics['final_capital']:.2f}, "
                f"Rendement: {metrics['return_pct']:.2f}%, "
                f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    
    return metrics


def main():
    """Fonction principale pour exécuter le backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Exécute un backtest avec le modèle monolithique")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle sauvegardé (.keras)")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers les données de prix (csv/parquet)")
    parser.add_argument("--config", type=str, help="Chemin vers le fichier de configuration (optionnel)")
    parser.add_argument("--output-dir", type=str, default="./backtest_results", help="Répertoire de sortie")
    parser.add_argument("--initial-cash", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission par trade")
    parser.add_argument("--risk", type=float, default=0.02, help="Risque par trade (fraction du capital)")
    parser.add_argument("--start-date", type=str, help="Date de début (format YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Date de fin (format YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Charger le modèle
    logger.info(f"Chargement du modèle depuis {args.model}")
    model = MonolithModel.load(args.model)
    
    # Charger la configuration si fournie
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    
    # Déterminer les features nécessaires
    if config and "metadata" in config:
        tech_cols = config["metadata"].get("tech_feature_cols", [])
        embedding_cols = config["metadata"].get("embedding_cols", [])
        mcp_cols = config["metadata"].get("mcp_cols", [])
        instrument_col = "instrument"
        instrument_map = config["metadata"].get("instrument_map", None)
        sequence_length = config["metadata"].get("sequence_length", None)
    else:
        # Déduction des colonnes basée sur les conventions de nommage
        tech_cols = []
        embedding_cols = []
        mcp_cols = []
        instrument_col = "instrument"
        instrument_map = None
        sequence_length = None
    
    # Charger les données pour le backtest
    all_needed_cols = tech_cols + embedding_cols + mcp_cols + [instrument_col, "open", "high", "low", "close", "volume"]
    df = load_data_for_backtest(args.data, all_needed_cols, args.start_date, args.end_date)
    
    # Si les colonnes n'ont pas été déterminées par la configuration, les définir
    if not tech_cols:
        tech_cols = [col for col in df.columns if col.startswith('tech_') or col in ['open', 'high', 'low', 'close', 'volume']]
    
    if not embedding_cols:
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    
    if not mcp_cols:
        mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
    
    # Générer les signaux
    signals_df, sltp_df = generate_signals(
        model=model,
        df=df,
        tech_cols=tech_cols,
        embedding_cols=embedding_cols,
        mcp_cols=mcp_cols,
        instrument_col=instrument_col,
        sequence_length=sequence_length,
        instrument_map=instrument_map
    )
    
    # Préparer les données OHLCV pour le backtest
    price_data = df[["open", "high", "low", "close", "volume"]].copy()
    
    # Exécuter le backtest
    metrics = run_backtest(
        price_data=price_data,
        signals_df=signals_df,
        sltp_df=sltp_df,
        initial_cash=args.initial_cash,
        commission=args.commission,
        risk_per_trade=args.risk,
        output_dir=args.output_dir
    )
    
    return metrics


if __name__ == "__main__":
    main() 