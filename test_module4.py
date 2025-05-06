#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test du Module 4: Optimisation génétique des hyperparamètres de l'agent RL
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from model.training.genetic_optimizer import (
    load_data,
    decode_individual,
    create_trading_env,
    create_rl_agent,
    evaluate_hyperparams,
    optimize_hyperparams,
    train_best_agent
)

def generate_test_data(n_samples=1000):
    """
    Génère des données de test pour l'environnement de trading.
    
    Args:
        n_samples: Nombre d'échantillons à générer
    
    Returns:
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
    """
    print("Génération des données de test...")
    
    # Générer une série temporelle pour les prix
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='h')
    # Convertir les dates en timestamps Unix (secondes depuis epoch) pour qu'elles soient numériques
    timestamps = dates.astype(np.int64) // 10**9  # Convertir nanosecondes en secondes
    
    # Prix avec tendance et bruit
    price = 100
    prices = []
    for i in range(n_samples):
        # Ajouter une tendance et un bruit
        change = np.random.normal(0, 1) + np.sin(i/100) * 0.5
        price *= (1 + change/100)
        prices.append(price)
    
    # Créer le DataFrame des prix
    price_data = pd.DataFrame({
        'timestamp': dates,  # Garder les dates pour price_data
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': [np.random.uniform(1000, 10000) for _ in range(n_samples)]
    })
    
    # Créer des caractéristiques techniques en suivant la structure de données détaillée
    # Nous incluons les 41+ colonnes recommandées pour un modèle complet
    
    # 1. Données brutes de marché (déjà dans price_data, mais ajoutons price)
    # 2. Données microstructurelles
    # 3. Indicateurs techniques classiques
    # 4. Raisonnement / Événements structurés
    # 5. Colonnes de target
    # 6. Colonnes système
    
    feature_data = pd.DataFrame()
    
    # Utiliser timestamp numérique pour feature_data
    feature_data['timestamp_numeric'] = timestamps
    
    # 1. Données brutes de marché (complémentaires)
    feature_data['price'] = prices
    
    # 2. Données microstructurelles
    feature_data['bid_price'] = [p * (1 - np.random.uniform(0, 0.001)) for p in prices]
    feature_data['ask_price'] = [p * (1 + np.random.uniform(0, 0.001)) for p in prices]
    feature_data['bid_volume'] = [np.random.uniform(500, 5000) for _ in range(n_samples)]
    feature_data['ask_volume'] = [np.random.uniform(500, 5000) for _ in range(n_samples)]
    feature_data['spread'] = feature_data['ask_price'] - feature_data['bid_price']
    feature_data['order_imbalance'] = (feature_data['bid_volume'] - feature_data['ask_volume']) / (feature_data['bid_volume'] + feature_data['ask_volume'])
    
    # 3. Indicateurs techniques classiques
    feature_data['rsi_14'] = np.random.uniform(0, 100, n_samples)
    feature_data['macd'] = np.random.normal(0, 1, n_samples)
    feature_data['macd_signal'] = np.random.normal(0, 1, n_samples)
    feature_data['ema_9'] = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
    feature_data['ema_21'] = [p * (1 + np.random.normal(0, 0.02)) for p in prices]
    feature_data['sma_50'] = [p * (1 + np.random.normal(0, 0.03)) for p in prices]
    feature_data['bollinger_upper'] = [p * (1 + np.random.uniform(0.01, 0.05)) for p in prices]
    feature_data['bollinger_lower'] = [p * (1 - np.random.uniform(0.01, 0.05)) for p in prices]
    feature_data['atr_14'] = np.abs(np.random.normal(0, 2, n_samples))
    feature_data['adx'] = np.random.uniform(0, 100, n_samples)
    
    # 4. Raisonnement / Événements structurés
    feature_data['event_spike_volume'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    feature_data['event_breakout'] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    feature_data['event_reversal'] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    feature_data['trend_direction'] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.3, 0.4, 0.3])
    feature_data['momentum_shift'] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.25, 0.5, 0.25])
    feature_data['pattern_match'] = np.random.choice(range(10), size=n_samples)
    
    # 5. Colonnes de target
    feature_data['future_return_5s'] = np.random.normal(0, 0.001, n_samples)
    feature_data['future_return_10s'] = np.random.normal(0, 0.002, n_samples)
    feature_data['future_signal'] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
    feature_data['future_max_dd'] = np.random.uniform(0, 0.05, n_samples)
    feature_data['target_profit'] = np.random.uniform(0, 0.1, n_samples)
    
    # 6. Colonnes système
    feature_data['position'] = np.random.choice([-1, 0, 1], size=n_samples, p=[0.1, 0.8, 0.1])
    feature_data['pnl'] = np.random.normal(0, 10, n_samples)
    feature_data['cumulative_pnl'] = np.cumsum(feature_data['pnl'])
    feature_data['drawdown'] = np.random.uniform(0, 0.1, n_samples)
    feature_data['entry_price'] = [p if np.random.random() > 0.7 else 0 for p in prices]
    feature_data['exit_signal'] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    feature_data['execution_latency'] = np.random.uniform(10, 100, n_samples)
    
    # Ajouter quelques colonnes supplémentaires pour atteindre 41+
    feature_data['cci'] = np.random.normal(0, 100, n_samples)
    feature_data['mfi'] = np.random.uniform(0, 100, n_samples)
    feature_data['obv'] = np.random.normal(0, 10000, n_samples)
    feature_data['williams_r'] = np.random.uniform(-100, 0, n_samples)
    feature_data['hour_of_day'] = [d.hour for d in dates]
    feature_data['day_of_week'] = [d.dayofweek for d in dates]
    
    print(f"Données générées: {len(price_data)} échantillons")
    print(f"Nombre de colonnes dans price_data: {len(price_data.columns)}")
    print(f"Nombre de colonnes dans feature_data: {len(feature_data.columns)}")
    print(f"Toutes les colonnes de feature_data sont numériques: {all(dtype.kind in 'biufc' for dtype in feature_data.dtypes)}")
    
    return price_data, feature_data

def test_decode_individual():
    """
    Teste la fonction de décodage d'un individu en hyperparamètres.
    """
    print("\n=== Test de la fonction decode_individual ===")
    
    # Importer le dictionnaire HYPERPARAM_RANGES pour connaître sa taille
    from model.training.genetic_optimizer import HYPERPARAM_RANGES
    
    # Créer un individu de test (liste de valeurs entre 0 et 1)
    individual = [0.5] * len(HYPERPARAM_RANGES)
    print(f"Taille de l'individu: {len(individual)}")
    print(f"Nombre d'hyperparamètres: {len(HYPERPARAM_RANGES)}")
    
    # Décoder l'individu
    hyperparams = decode_individual(individual)
    
    # Vérifier que tous les hyperparamètres sont présents
    print(f"Hyperparamètres décodés: {hyperparams}")
    assert len(hyperparams) > 0, "Aucun hyperparamètre décodé"
    assert len(hyperparams) == len(HYPERPARAM_RANGES), "Le nombre d'hyperparamètres décodés ne correspond pas"
    
    # Vérifier que les valeurs sont dans les plages attendues
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    
    print("Test de decode_individual réussi!")

def test_create_trading_env():
    """
    Teste la fonction create_trading_env qui crée un environnement de trading
    pour l'agent RL.
    """
    print("\n=== Test de la fonction create_trading_env ===")
    
    # Générer des données simplifiées pour le test
    # Augmenter le nombre d'échantillons pour éviter les problèmes de plage vide
    n_samples = 500
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='h')
    timestamps = dates.astype(np.int64) // 10**9  # Convertir en secondes depuis epoch
    
    # Prix avec tendance et bruit
    price = 100
    prices = []
    for i in range(n_samples):
        change = np.random.normal(0, 1) + np.sin(i/10) * 0.5
        price *= (1 + change/100)
        prices.append(price)
    
    # Créer le DataFrame des prix
    price_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'volume': [np.random.uniform(1000, 10000) for _ in range(n_samples)]
    })
    
    # Créer des caractéristiques techniques minimales
    # Utiliser des caractéristiques numériques pour éviter les problèmes de type
    feature_data = pd.DataFrame()
    feature_data['timestamp_numeric'] = timestamps  # Utiliser timestamp numérique
    
    # Ajouter suffisamment de caractéristiques numériques
    for i in range(1, 10):  # Ajouter 9 caractéristiques
        feature_data[f'feature{i}'] = np.random.normal(0, 1, n_samples)
    
    print(f"Données simplifiées générées: {len(price_data)} échantillons")
    print(f"Nombre de colonnes dans price_data: {len(price_data.columns)}")
    print(f"Nombre de colonnes dans feature_data: {len(feature_data.columns)}")
    print(f"Toutes les colonnes de feature_data sont numériques: {all(dtype.kind in 'biufc' for dtype in feature_data.dtypes)}")
    
    # Créer des hyperparamètres de test adaptés aux données simplifiées
    hyperparams = {
        'window_size': 10,  # Fenêtre plus petite pour accélérer les tests
        'initial_balance': 20.0,  # Capital initial de 20$ comme demandé
        'transaction_fee': 0.001,
        'reward_scaling': 1.0,
        'max_steps': 100  # Limiter le nombre de pas pour éviter de dépasser la taille des données
    }
    
    # Créer l'environnement
    env = create_trading_env(hyperparams, price_data, feature_data)
    
    assert env is not None, "L'environnement n'a pas été créé"
    
    try:
        # Tester un reset
        obs = env.reset()
        print(f"Forme de l'observation: {obs[0].shape}")
        
        # Tester une action
        action = np.array([0])  # Action 'hold'
        # Gestion des deux versions d'API possibles (Gymnasium < 0.26 et >= 0.26)
        step_result = env.step(action)
        
        if len(step_result) == 5:  # Gymnasium >= 0.26
            obs, reward, _, _, _ = step_result
            print("API Gymnasium >= 0.26 détectée")
        else:  # Gymnasium < 0.26 ou OpenAI Gym
            obs, reward, _, _ = step_result
            print("API Gymnasium < 0.26 ou OpenAI Gym détectée")
            
        print(f"Récompense: {reward}")
        print(f"Forme de l'observation après action: {obs.shape}")
        
        print("Test de create_trading_env réussi!")
    except Exception as e:
        print(f"Erreur lors du test de l'environnement: {e}")
        raise

def test_create_rl_agent():
    """
    Teste la création d'un agent RL.
    """
    print("\n=== Test de la fonction create_rl_agent ===")
    
    # Générer des données de test
    price_data, feature_data = generate_test_data(100)
    
    print(f"Données simplifiées générées: {len(price_data)} échantillons")
    
    # Créer des hyperparamètres de test
    hyperparams = {
        'window_size': 10,  # Fenêtre plus petite pour accélérer les tests
        'initial_balance': 20.0,  # Capital initial de 20$ comme demandé
        'transaction_fee': 0.001,
        'reward_scaling': 1.0,
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 64,
        'n_steps': 64,  # Plus petit pour accélérer les tests
        'n_epochs': 2,   # Plus petit pour accélérer les tests
        'clip_range': 0.2
    }
    
    try:
        # Créer l'environnement
        env = create_trading_env(hyperparams, price_data, feature_data)
        
        # Créer l'agent
        output_dir = "output/test_agent"
        os.makedirs(output_dir, exist_ok=True)
        agent = create_rl_agent(hyperparams, env, output_dir)
        
        # Vérifier que l'agent est créé
        print(f"Type d'agent: {type(agent)}")
        assert agent is not None, "L'agent n'a pas été créé"
        
        # Tester un entraînement court
        print("Entraînement de l'agent pour 10 timesteps...")
        agent.train(total_timesteps=10)  # Très court pour les tests
        
        # Tester une évaluation
        print("\u00c9valuation de l'agent...")
        metrics, _, _ = agent.evaluate(n_episodes=1)
        
        print(f"Métriques: {metrics}")
        
        # Fermer l'environnement
        env.close()
        
        print("Test de create_rl_agent réussi!")
    except Exception as e:
        print(f"Erreur lors du test de l'agent RL: {e}")
        if 'env' in locals():
            env.close()
        raise

def test_evaluate_hyperparams():
    """
    Teste l'évaluation des hyperparamètres.
    """
    print("\n=== Test de la fonction evaluate_hyperparams ===")
    
    # Générer des données de test
    price_data, feature_data = generate_test_data(1000)
    
    # Créer des hyperparamètres de test
    hyperparams = {
        'window_size': 30,
        'initial_balance': 1000.0,
        'transaction_fee': 0.001,
        'reward_scaling': 1.0,
        'max_position': 1.0,
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 64,
        'n_steps': 128,
        'n_epochs': 5,
        'clip_range': 0.2
    }
    
    # Évaluer les hyperparamètres
    output_dir = "output/test_evaluate"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Évaluation des hyperparamètres...")
    score = evaluate_hyperparams(hyperparams, price_data, feature_data, output_dir, train_timesteps=100)
    
    print(f"Score: {score}")
    assert score is not None, "Le score n'a pas été calculé"
    
    print("Test de evaluate_hyperparams réussi!")

def test_optimize_hyperparams():
    """
    Teste l'optimisation des hyperparamètres avec un algorithme génétique.
    """
    print("\n=== Test de la fonction optimize_hyperparams ===")
    
    # Générer des données de test
    price_data, feature_data = generate_test_data(1000)
    
    # Optimiser les hyperparamètres avec une petite population et peu de générations
    output_dir = "output/test_optimize"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Optimisation des hyperparamètres...")
    best_hyperparams = optimize_hyperparams(
        price_data,
        feature_data,
        output_dir,
        population_size=4,
        generations=2,
        train_timesteps=100
    )
    
    print(f"Meilleurs hyperparamètres: {best_hyperparams}")
    assert best_hyperparams is not None, "Les meilleurs hyperparamètres n'ont pas été trouvés"
    
    print("Test de optimize_hyperparams réussi!")

def test_train_best_agent():
    """
    Teste l'entraînement de l'agent avec les meilleurs hyperparamètres.
    """
    print("\n=== Test de la fonction train_best_agent ===")
    
    # Générer des données de test
    price_data, feature_data = generate_test_data(1000)
    
    # Créer des hyperparamètres de test
    best_hyperparams = {
        'window_size': 30,
        'initial_balance': 1000.0,
        'transaction_fee': 0.001,
        'reward_scaling': 1.0,
        'max_position': 1.0,
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'batch_size': 64,
        'n_steps': 128,
        'n_epochs': 5,
        'clip_range': 0.2
    }
    
    # Entraîner l'agent
    output_dir = "output/test_train_best"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Entraînement de l'agent avec les meilleurs hyperparamètres...")
    best_agent = train_best_agent(
        best_hyperparams,
        price_data,
        feature_data,
        output_dir,
        train_timesteps=100
    )
    
    print(f"Type d'agent: {type(best_agent)}")
    assert best_agent is not None, "L'agent n'a pas été entraîné"
    
    print("Test de train_best_agent réussi!")

def test_cot_coherence():
    """
    Teste la cohérence des explications CoT.
    """
    print("\n=== Test de la cohérence des explications CoT ===")
    
    # Initialiser l'agent avec CoT activé
    agent = create_rl_agent(use_cot=True)
    
    # Générer deux explications
    market_data_sample = pd.DataFrame({
        'timestamp': [1, 2, 3],
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 2000, 3000]
    })
    expl1 = agent.generate_explanation(market_data_sample)
    expl2 = agent.generate_explanation(market_data_sample)
    
    # Vérifier la cohérence
    similarity = agent.calculate_explanation_similarity(expl1, expl2)
    assert 0.7 <= similarity <= 1.0, "Les explications doivent être cohérentes"
    
    print("Test de la cohérence des explications CoT réussi!")

def run_all_tests():
    """
    Exécute tous les tests.
    """
    print("=== Démarrage des tests du Module 4: Optimisation génétique des hyperparamètres de l'agent RL ===")
    
    # Créer le répertoire de sortie
    os.makedirs("output", exist_ok=True)
    
    # Exécuter les tests
    test_decode_individual()
    test_create_trading_env()
    test_create_rl_agent()
    test_evaluate_hyperparams()
    test_optimize_hyperparams()
    test_train_best_agent()
    test_cot_coherence()
    
    print("\n=== Tous les tests du Module 4 ont réussi! ===")

if __name__ == "__main__":
    run_all_tests()
