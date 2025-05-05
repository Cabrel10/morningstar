#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'u00e9valuation amu00e9lioru00e9 pour le modu00e8le Morningstar.
Ce script u00e9value le modu00e8le sur plusieurs mu00e9triques et visualise les ru00e9sultats.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """
    Charge les donnu00e9es normalisu00e9es.
    
    Args:
        data_path: Chemin vers le dataset normalisu00e9
    
    Returns:
        DataFrame avec les donnu00e9es de test
    """
    logger.info(f"Chargement des donnu00e9es depuis {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Filtrer pour ne garder que les donnu00e9es de test
    test_df = df[df['split'] == 'test'].drop(columns=['split'])
    logger.info(f"Ensemble de test: {len(test_df)} lignes")
    
    return test_df

def load_model(model_path):
    """
    Charge le modu00e8le entrau00eenu00e9.
    
    Args:
        model_path: Chemin vers le modu00e8le entrau00eenu00e9
    
    Returns:
        Modu00e8le Keras chargu00e9
    """
    logger.info(f"Chargement du modu00e8le depuis {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Modu00e8le chargu00e9: {model.name}")
    logger.info(f"Entru00e9es du modu00e8le: {model.input_names}")
    logger.info(f"Sorties du modu00e8le: {model.output_names}")
    
    return model

def load_scalers(scalers_path):
    """
    Charge les scalers pour la conversion inverse des pru00e9dictions.
    
    Args:
        scalers_path: Chemin vers les scalers sauvegardus
    
    Returns:
        Dictionnaire de scalers
    """
    logger.info(f"Chargement des scalers depuis {scalers_path}")
    scalers = np.load(scalers_path, allow_pickle=True)
    logger.info(f"Scalers chargu00e9s: {list(scalers.keys())}")
    
    return scalers

def prepare_features(df):
    """
    Pru00e9pare les features pour le modu00e8le.
    
    Args:
        df: DataFrame avec les donnu00e9es de test
    
    Returns:
        Dictionnaire de features
    """
    # Colonnes techniques
    technical_cols = [col for col in df.columns if col not in [
        'market_regime', 'level_sl', 'level_tp', 'instrument_type',
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
    ] and not col.startswith('llm_') and not col.startswith('mcp_')]
    
    # Colonnes LLM
    llm_cols = [col for col in df.columns if col.startswith('llm_')]
    
    # Colonnes MCP
    mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
    
    # Colonnes HMM
    hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Pru00e9parer les dictionnaires de features
    features = {
        'technical_input': df[technical_cols].values,
        'llm_input': df[llm_cols].values,
        'mcp_input': df[mcp_cols].values,
        'hmm_input': df[hmm_cols].values,
        'instrument_input': df[['instrument_type']].values
    }
    
    return features

def prepare_targets(df):
    """
    Pru00e9pare les cibles pour l'u00e9valuation.
    
    Args:
        df: DataFrame avec les donnu00e9es de test
    
    Returns:
        Dictionnaire de cibles
    """
    targets = {
        'market_regime': df['market_regime'].values,
        'sl_tp': df[['level_sl', 'level_tp']].values
    }
    
    return targets

def evaluate_model(model, features, targets, scalers=None, output_dir=None):
    """
    u00c9value le modu00e8le sur les donnu00e9es de test.
    
    Args:
        model: Modu00e8le Keras u00e0 u00e9valuer
        features: Dictionnaire de features
        targets: Dictionnaire de cibles
        scalers: Dictionnaire de scalers pour la conversion inverse
        output_dir: Ru00e9pertoire de sortie pour les visualisations
    
    Returns:
        Dictionnaire de mu00e9triques d'u00e9valuation
    """
    # Faire des pru00e9dictions
    logger.info("Pru00e9diction sur les donnu00e9es de test")
    predictions = model.predict(features)
    
    # Pru00e9parer les ru00e9sultats
    results = {}
    
    # u00c9valuer la pru00e9diction du ru00e9gime de marchu00e9
    if 'market_regime' in predictions:
        market_regime_pred = np.argmax(predictions['market_regime'], axis=1)
        market_regime_true = targets['market_regime']
        
        # Calculer les mu00e9triques
        market_regime_accuracy = np.mean(market_regime_pred == market_regime_true)
        market_regime_cm = confusion_matrix(market_regime_true, market_regime_pred)
        market_regime_report = classification_report(market_regime_true, market_regime_pred, output_dict=True)
        
        results['market_regime'] = {
            'accuracy': market_regime_accuracy,
            'confusion_matrix': market_regime_cm,
            'classification_report': market_regime_report
        }
        
        logger.info(f"Pru00e9cision du ru00e9gime de marchu00e9: {market_regime_accuracy:.4f}")
        logger.info(f"Matrice de confusion du ru00e9gime de marchu00e9:\n{market_regime_cm}")
        
        # Visualiser la matrice de confusion
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(market_regime_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matrice de confusion - Ru00e9gime de marchu00e9')
            plt.xlabel('Pru00e9diction')
            plt.ylabel('Vraie valeur')
            plt.savefig(os.path.join(output_dir, 'market_regime_confusion_matrix.png'))
            plt.close()
    
    # u00c9valuer la pru00e9diction des niveaux SL/TP
    if 'sl_tp' in predictions:
        sl_tp_pred = predictions['sl_tp']
        sl_tp_true = targets['sl_tp']
        
        # Convertir les pru00e9dictions normalisu00e9es en valeurs ru00e9elles si les scalers sont disponibles
        if scalers is not None and 'sl_tp_scaler' in scalers:
            sl_tp_scaler = scalers['sl_tp_scaler'].item()
            sl_tp_pred = sl_tp_scaler.inverse_transform(sl_tp_pred)
            sl_tp_true = sl_tp_scaler.inverse_transform(sl_tp_true)
        
        # Su00e9parer les pru00e9dictions SL et TP
        sl_pred = sl_tp_pred[:, 0]
        tp_pred = sl_tp_pred[:, 1]
        sl_true = sl_tp_true[:, 0]
        tp_true = sl_tp_true[:, 1]
        
        # Calculer les mu00e9triques
        sl_rmse = np.sqrt(mean_squared_error(sl_true, sl_pred))
        sl_mae = mean_absolute_error(sl_true, sl_pred)
        tp_rmse = np.sqrt(mean_squared_error(tp_true, tp_pred))
        tp_mae = mean_absolute_error(tp_true, tp_pred)
        
        results['sl_tp'] = {
            'sl_rmse': sl_rmse,
            'sl_mae': sl_mae,
            'tp_rmse': tp_rmse,
            'tp_mae': tp_mae,
            'sl_percentiles': np.percentile(np.abs(sl_pred - sl_true), [25, 50, 75, 90, 95, 99]),
            'tp_percentiles': np.percentile(np.abs(tp_pred - tp_true), [25, 50, 75, 90, 95, 99])
        }
        
        logger.info(f"RMSE du niveau SL: {sl_rmse:.4f}")
        logger.info(f"MAE du niveau SL: {sl_mae:.4f}")
        logger.info(f"RMSE du niveau TP: {tp_rmse:.4f}")
        logger.info(f"MAE du niveau TP: {tp_mae:.4f}")
        
        # Visualiser les erreurs SL/TP
        if output_dir:
            # Distribution des erreurs SL
            plt.figure(figsize=(12, 6))
            sns.histplot(sl_pred - sl_true, kde=True)
            plt.title('Distribution des erreurs - Niveau SL')
            plt.xlabel('Erreur (Pru00e9diction - Vraie valeur)')
            plt.ylabel('Fru00e9quence')
            plt.savefig(os.path.join(output_dir, 'sl_error_distribution.png'))
            plt.close()
            
            # Distribution des erreurs TP
            plt.figure(figsize=(12, 6))
            sns.histplot(tp_pred - tp_true, kde=True)
            plt.title('Distribution des erreurs - Niveau TP')
            plt.xlabel('Erreur (Pru00e9diction - Vraie valeur)')
            plt.ylabel('Fru00e9quence')
            plt.savefig(os.path.join(output_dir, 'tp_error_distribution.png'))
            plt.close()
            
            # Scatter plot SL
            plt.figure(figsize=(10, 10))
            plt.scatter(sl_true, sl_pred, alpha=0.5)
            plt.plot([sl_true.min(), sl_true.max()], [sl_true.min(), sl_true.max()], 'r--')
            plt.title('Pru00e9diction vs Vraie valeur - Niveau SL')
            plt.xlabel('Vraie valeur')
            plt.ylabel('Pru00e9diction')
            plt.savefig(os.path.join(output_dir, 'sl_prediction_scatter.png'))
            plt.close()
            
            # Scatter plot TP
            plt.figure(figsize=(10, 10))
            plt.scatter(tp_true, tp_pred, alpha=0.5)
            plt.plot([tp_true.min(), tp_true.max()], [tp_true.min(), tp_true.max()], 'r--')
            plt.title('Pru00e9diction vs Vraie valeur - Niveau TP')
            plt.xlabel('Vraie valeur')
            plt.ylabel('Pru00e9diction')
            plt.savefig(os.path.join(output_dir, 'tp_prediction_scatter.png'))
            plt.close()
    
    # Sauvegarder les ru00e9sultats
    if output_dir:
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            # Convertir les arrays numpy en listes pour la su00e9rialisation JSON
            results_json = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    results_json[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            results_json[key][k] = v.tolist()
                        else:
                            results_json[key][k] = v
                else:
                    if isinstance(value, np.ndarray):
                        results_json[key] = value.tolist()
                    else:
                        results_json[key] = value
            
            json.dump(results_json, f, indent=4)
    
    return results

def simulate_trading(df, market_regime_pred, sl_pred, tp_pred, output_dir=None):
    """
    Simule une stratu00e9gie de trading simple basée sur les pru00e9dictions.
    
    Args:
        df: DataFrame avec les donnu00e9es de test
        market_regime_pred: Pru00e9dictions du ru00e9gime de marchu00e9
        sl_pred: Pru00e9dictions du niveau SL
        tp_pred: Pru00e9dictions du niveau TP
        output_dir: Ru00e9pertoire de sortie pour les visualisations
    
    Returns:
        Dictionnaire de mu00e9triques de trading
    """
    # Cru00e9er un DataFrame pour la simulation
    trading_df = df.copy()
    trading_df['market_regime_pred'] = market_regime_pred
    trading_df['sl_pred'] = sl_pred
    trading_df['tp_pred'] = tp_pred
    
    # Initialiser les variables de trading
    balance = 10000  # Balance initiale
    position = None  # Position actuelle (None, 'long', 'short')
    entry_price = 0  # Prix d'entru00e9e
    trade_results = []  # Liste des ru00e9sultats des trades
    
    # Parcourir les donnu00e9es chronologiquement
    for i in range(1, len(trading_df)):
        current_row = trading_df.iloc[i]
        prev_row = trading_df.iloc[i-1]
        
        # Prix actuel
        current_price = current_row['close']
        
        # Vu00e9rifier si nous avons une position ouverte
        if position is not None:
            # Vu00e9rifier si le SL ou TP est atteint
            if position == 'long':
                # SL atteint
                if current_price <= entry_price * (1 - abs(prev_row['sl_pred']) / 100):
                    profit_pct = (current_price / entry_price - 1) * 100
                    trade_results.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': profit_pct,
                        'exit_reason': 'sl'
                    })
                    position = None
                # TP atteint
                elif current_price >= entry_price * (1 + abs(prev_row['tp_pred']) / 100):
                    profit_pct = (current_price / entry_price - 1) * 100
                    trade_results.append({
                        'type': 'long',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': profit_pct,
                        'exit_reason': 'tp'
                    })
                    position = None
            elif position == 'short':
                # SL atteint
                if current_price >= entry_price * (1 + abs(prev_row['sl_pred']) / 100):
                    profit_pct = (entry_price / current_price - 1) * 100
                    trade_results.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': profit_pct,
                        'exit_reason': 'sl'
                    })
                    position = None
                # TP atteint
                elif current_price <= entry_price * (1 - abs(prev_row['tp_pred']) / 100):
                    profit_pct = (entry_price / current_price - 1) * 100
                    trade_results.append({
                        'type': 'short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': profit_pct,
                        'exit_reason': 'tp'
                    })
                    position = None
        
        # Vu00e9rifier si nous pouvons ouvrir une nouvelle position
        if position is None:
            # Ouvrir une position longue si le ru00e9gime pru00e9dit est bullish (1)
            if prev_row['market_regime_pred'] == 1:
                position = 'long'
                entry_price = current_price
            # Ouvrir une position courte si le ru00e9gime pru00e9dit est bearish (2)
            elif prev_row['market_regime_pred'] == 2:
                position = 'short'
                entry_price = current_price
    
    # Fermer la position finale si elle est encore ouverte
    if position is not None:
        current_price = trading_df.iloc[-1]['close']
        if position == 'long':
            profit_pct = (current_price / entry_price - 1) * 100
        else:  # short
            profit_pct = (entry_price / current_price - 1) * 100
        
        trade_results.append({
            'type': position,
            'entry_price': entry_price,
            'exit_price': current_price,
            'profit_pct': profit_pct,
            'exit_reason': 'end'
        })
    
    # Calculer les mu00e9triques de trading
    if len(trade_results) > 0:
        trade_df = pd.DataFrame(trade_results)
        
        # Calculer le profit total
        total_profit_pct = trade_df['profit_pct'].sum()
        
        # Calculer le nombre de trades gagnants/perdants
        winning_trades = len(trade_df[trade_df['profit_pct'] > 0])
        losing_trades = len(trade_df[trade_df['profit_pct'] <= 0])
        win_rate = winning_trades / len(trade_df) if len(trade_df) > 0 else 0
        
        # Calculer le ratio profit/perte
        avg_win = trade_df[trade_df['profit_pct'] > 0]['profit_pct'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trade_df[trade_df['profit_pct'] <= 0]['profit_pct'].mean()) if losing_trades > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculer le drawdown maximal
        cumulative_returns = (1 + trade_df['profit_pct'] / 100).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak - 1) * 100
        max_drawdown = abs(drawdown.min())
        
        trading_metrics = {
            'total_trades': len(trade_df),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit_pct': total_profit_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown
        }
        
        logger.info(f"Nombre total de trades: {len(trade_df)}")
        logger.info(f"Trades gagnants: {winning_trades} ({win_rate:.2%})")
        logger.info(f"Profit total: {total_profit_pct:.2f}%")
        logger.info(f"Ratio profit/perte: {profit_loss_ratio:.2f}")
        logger.info(f"Drawdown maximal: {max_drawdown:.2f}%")
        
        # Visualiser les ru00e9sultats de trading
        if output_dir and len(trade_df) > 0:
            # Distribution des profits
            plt.figure(figsize=(12, 6))
            sns.histplot(trade_df['profit_pct'], kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Distribution des profits par trade')
            plt.xlabel('Profit (%)')
            plt.ylabel('Fru00e9quence')
            plt.savefig(os.path.join(output_dir, 'trade_profit_distribution.png'))
            plt.close()
            
            # u00c9volution du capital
            plt.figure(figsize=(12, 6))
            plt.plot((1 + trade_df['profit_pct'] / 100).cumprod() * balance)
            plt.title('u00c9volution du capital')
            plt.xlabel('Nombre de trades')
            plt.ylabel('Capital ($)')
            plt.savefig(os.path.join(output_dir, 'capital_evolution.png'))
            plt.close()
            
            # Sauvegarder les ru00e9sultats des trades
            trade_df.to_csv(os.path.join(output_dir, 'trade_results.csv'), index=False)
    else:
        trading_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_loss_ratio': 0,
            'max_drawdown': 0
        }
        logger.warning("Aucun trade gu00e9nu00e9ru00e9 pendant la simulation")
    
    # Sauvegarder les mu00e9triques de trading
    if output_dir:
        with open(os.path.join(output_dir, 'trading_metrics.json'), 'w') as f:
            json.dump(trading_metrics, f, indent=4)
    
    return trading_metrics

def main():
    parser = argparse.ArgumentParser(description='u00c9value le modu00e8le Morningstar sur les donnu00e9es de test.')
    parser.add_argument('--data-path', type=str, required=True, help='Chemin vers le dataset normalisu00e9')
    parser.add_argument('--model-path', type=str, required=True, help='Chemin vers le modu00e8le entrau00eenu00e9')
    parser.add_argument('--scalers-path', type=str, help='Chemin vers les scalers sauvegardus')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Ru00e9pertoire de sortie pour les visualisations')
    
    args = parser.parse_args()
    
    # Cru00e9er le ru00e9pertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Charger les donnu00e9es
    test_df = load_data(args.data_path)
    
    # Charger le modu00e8le
    model = load_model(args.model_path)
    
    # Charger les scalers si disponibles
    scalers = None
    if args.scalers_path:
        scalers = load_scalers(args.scalers_path)
    
    # Pru00e9parer les features et cibles
    features = prepare_features(test_df)
    targets = prepare_targets(test_df)
    
    # u00c9valuer le modu00e8le
    results = evaluate_model(model, features, targets, scalers, args.output_dir)
    
    # Faire des pru00e9dictions pour la simulation de trading
    predictions = model.predict(features)
    
    # Pru00e9parer les pru00e9dictions pour la simulation
    market_regime_pred = np.argmax(predictions['market_regime'], axis=1) if 'market_regime' in predictions else None
    
    sl_tp_pred = predictions['sl_tp'] if 'sl_tp' in predictions else None
    if sl_tp_pred is not None and scalers is not None and 'sl_tp_scaler' in scalers:
        sl_tp_scaler = scalers['sl_tp_scaler'].item()
        sl_tp_pred = sl_tp_scaler.inverse_transform(sl_tp_pred)
    
    sl_pred = sl_tp_pred[:, 0] if sl_tp_pred is not None else None
    tp_pred = sl_tp_pred[:, 1] if sl_tp_pred is not None else None
    
    # Simuler le trading
    if market_regime_pred is not None and sl_pred is not None and tp_pred is not None:
        trading_metrics = simulate_trading(test_df, market_regime_pred, sl_pred, tp_pred, args.output_dir)
    else:
        logger.warning("Impossible de simuler le trading: pru00e9dictions manquantes")

if __name__ == "__main__":
    main()
