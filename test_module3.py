#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour le Module 3: Apprentissage par Renforcement (RL)
Ce script teste l'intégration du module RL avec le modèle hybride existant.
"""

import os
import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configurer le logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ajouter le répertoire du projet au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importer les modules nécessaires
from model.training.reinforcement_learning import (
    TradingEnvironment,
    TradingRLAgent,
    create_trading_env_from_data,
    load_and_prepare_data,
)


def generate_synthetic_data(num_samples=1000):
    """
    Génère des données synthétiques pour tester le module RL.

    Args:
        num_samples: Nombre d'échantillons à générer

    Returns:
        DataFrame de prix et DataFrame de caractéristiques
    """
    # Générer un mouvement brownien pour le prix
    np.random.seed(42)
    price = 100.0
    prices = [price]

    for _ in range(num_samples - 1):
        # Simuler un mouvement de prix aléatoire
        change_percent = np.random.normal(0, 0.01)
        price = price * (1 + change_percent)
        prices.append(price)

    # Créer une série temporelle d'indices
    dates = pd.date_range(start="2023-01-01", periods=num_samples, freq="H")

    # Créer le DataFrame des prix
    price_data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, num_samples),
        },
        index=dates,
    )

    # Créer des caractéristiques techniques simples
    feature_data = pd.DataFrame(
        {
            # Tendance (moyenne mobile sur 10 périodes)
            "sma_10": price_data["close"].rolling(10).mean(),
            # Tendance (moyenne mobile sur 30 périodes)
            "sma_30": price_data["close"].rolling(30).mean(),
            # Momentum (% de changement sur 5 périodes)
            "momentum_5": price_data["close"].pct_change(5),
            # Momentum (% de changement sur 20 périodes)
            "momentum_20": price_data["close"].pct_change(20),
            # Volatilité (écart-type sur 10 périodes)
            "volatility_10": price_data["close"].rolling(10).std(),
            # Volatilité (écart-type sur 30 périodes)
            "volatility_30": price_data["close"].rolling(30).std(),
            # Indicateur RSI simplifié
            "rsi": np.random.uniform(0, 100, num_samples),
            # Indicateur MACD simplifié
            "macd": price_data["close"].rolling(12).mean() - price_data["close"].rolling(26).mean(),
            # Signal MACD
            "macd_signal": (price_data["close"].rolling(12).mean() - price_data["close"].rolling(26).mean())
            .rolling(9)
            .mean(),
            # Volume normalisé
            "volume_norm": price_data["volume"] / price_data["volume"].rolling(20).mean(),
        }
    )

    # Remplir les valeurs NaN
    feature_data = feature_data.fillna(0)

    # Ajouter des variables catégorielles simulées pour LLM, MCP, etc.
    feature_data["llm_sentiment"] = np.random.choice([-1, 0, 1], num_samples)
    feature_data["mcp_regime"] = np.random.choice([0, 1, 2, 3], num_samples)
    feature_data["cot_signal"] = np.random.choice([-1, 0, 1], num_samples)

    return price_data, feature_data


def test_trading_environment():
    """Teste l'environnement de trading."""
    logger.info("=== Test de l'environnement de trading ===")

    # Générer des données synthétiques
    price_data, feature_data = generate_synthetic_data()

    # Créer l'environnement de trading
    env = TradingEnvironment(
        price_data=price_data,
        feature_data=feature_data,
        window_size=30,
        max_steps=100,
        initial_balance=10000.0,
        transaction_fee=0.001,
    )

    # Tester l'initialisation
    obs, _ = env.reset()  # Avec gymnasium, reset() retourne (obs, info)
    logger.info(f"Forme de l'observation initiale: {obs.shape}")

    # Exécuter quelques étapes aléatoires
    total_reward = 0
    for i in range(20):
        action = np.random.choice([0, 1, 2])
        obs, reward, terminated, truncated, info = env.step(
            action
        )  # Avec gymnasium, step() retourne (obs, reward, terminated, truncated, info)
        done = terminated or truncated
        total_reward += reward
        logger.info(f"Étape {i+1}, Action: {action}, Récompense: {reward:.4f}, Done: {done}")

        # Afficher des informations supplémentaires toutes les 5 étapes
        if (i + 1) % 5 == 0:
            logger.info(f"Position: {env.position}, Balance: {env.balance:.2f}, Trades: {len(env.trades)}")

    logger.info(f"Récompense totale: {total_reward:.4f}")
    logger.info(f"Informations finales: {env._get_info()}")

    return env


def test_rl_agent(env=None):
    """Teste l'agent RL avec un environnement de trading."""
    logger.info("=== Test de l'agent RL ===")

    if env is None:
        # Générer des données synthétiques
        price_data, feature_data = generate_synthetic_data()

        # Créer l'environnement de trading
        env = create_trading_env_from_data(
            price_data=price_data,
            feature_data=feature_data,
            window_size=30,
            max_steps=500,
            initial_balance=10000.0,
            transaction_fee=0.001,
        )

    # Créer l'agent RL
    output_dir = os.path.join(current_dir, "output", "test_rl")
    os.makedirs(output_dir, exist_ok=True)

    agent = TradingRLAgent(
        model_path=os.path.join(output_dir, "test_rl_model.zip"),
        learning_rate=3e-4,
        n_steps=128,  # Réduit pour le test
        batch_size=32,
        n_epochs=5,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
    )

    # Initialiser l'agent
    agent.create_agent(env)

    # Entraîner l'agent (avec un nombre réduit d'étapes pour le test)
    logger.info("Début de l'entraînement (test court)...")
    agent.train(total_timesteps=1000, eval_freq=500, n_eval_episodes=3)

    # Évaluer l'agent
    logger.info("Évaluation de l'agent...")
    metrics, trades, infos = agent.evaluate(n_episodes=3)

    return agent, metrics, trades


def plot_training_results(trades, save_path=None):
    """
    Visualise les résultats de l'entraînement.

    Args:
        trades: Liste des transactions effectuées
        save_path: Chemin où sauvegarder le graphique
    """
    if not trades:
        logger.warning("Aucune transaction à visualiser")
        return

    # Convertir en DataFrame
    trades_df = pd.DataFrame(trades)

    # Créer la figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]})

    # 1. Graphique des prix et des transactions
    ax1 = axs[0]

    # Extraire les achats et ventes
    buys = trades_df[trades_df["type"] == "buy"]
    sells = trades_df[trades_df["type"] == "sell"]

    # Tracer les transactions sur le graphique des prix
    if "step" in trades_df.columns and len(buys) > 0:
        ax1.scatter(buys["step"], buys["price"], marker="^", color="green", s=100, label="Achat")
    if "step" in trades_df.columns and len(sells) > 0:
        ax1.scatter(sells["step"], sells["price"], marker="v", color="red", s=100, label="Vente")

    ax1.set_title("Transactions de trading")
    ax1.set_xlabel("Étape")
    ax1.set_ylabel("Prix")
    ax1.legend()
    ax1.grid(True)

    # 2. Graphique du solde
    ax2 = axs[1]

    if "balance" in trades_df.columns:
        ax2.plot(trades_df["step"], trades_df["balance"], label="Solde", color="blue")
        ax2.set_title("Évolution du solde")
        ax2.set_xlabel("Étape")
        ax2.set_ylabel("Solde ($)")
        ax2.grid(True)

    plt.tight_layout()

    # Sauvegarder si un chemin est spécifié
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé dans {save_path}")

    plt.show()


def main():
    """Fonction principale de test."""
    logger.info("Démarrage des tests du module RL")

    # Tester l'environnement de trading
    env = test_trading_environment()

    # Tester l'agent RL
    agent, metrics, trades = test_rl_agent()

    # Visualiser les résultats
    output_dir = os.path.join(current_dir, "output", "test_rl")
    os.makedirs(output_dir, exist_ok=True)
    plot_training_results(trades, save_path=os.path.join(output_dir, "trading_results.png"))

    logger.info("Tests du module RL terminés avec succès")


if __name__ == "__main__":
    main()
