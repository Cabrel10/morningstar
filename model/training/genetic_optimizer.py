#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimisation des hyperparamètres du modèle Morningstar avec un algorithme génétique.
Module 4 du framework DECoT-RL-GA : optimisation des hyperparamètres de l'agent RL.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import logging
import random
import time
import torch
from deap import base, creator, tools, algorithms
from model.training.reinforcement_learning import TradingEnvironment

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes pour l'algorithme génétique
POPULATION_SIZE = 10  # Réduit pour les tests initiaux
GENERATIONS = 5       # Réduit pour les tests initiaux
CXPB = 0.7            # Probabilité de croisement
MUTPB = 0.2           # Probabilité de mutation
TOURNAMENT_SIZE = 3

# Plages de valeurs pour les hyperparamètres de l'agent RL
HYPERPARAM_RANGES = {
    # Hyperparamètres de l'agent PPO
    'learning_rate': (0.00001, 0.001),
    'n_steps': (64, 2048),
    'batch_size': (16, 256),
    'n_epochs': (3, 10),
    'gamma': (0.9, 0.9999),
    'gae_lambda': (0.9, 0.99),
    'clip_range': (0.1, 0.3),
    'ent_coef': (0.0, 0.01),
    'vf_coef': (0.5, 1.0),
    'max_grad_norm': (0.5, 1.0),
    
    # Hyperparamètres de l'environnement de trading
    'window_size': (10, 50),
    'reward_scaling': (0.1, 10.0),
    'transaction_fee': (0.0001, 0.002),
    
    # Hyperparamètres du modèle CNN+LSTM
    'cnn_filters': (16, 128),
    'cnn_kernel_size': (2, 5),
    'lstm_units': (32, 256),
    'dropout_rate': (0.1, 0.5),
}

def generate_synthetic_data(num_samples=1000, window_size=30):
    """
    Génère des données synthétiques pour tester l'algorithme génétique.
    
    Args:
        num_samples: Nombre d'échantillons à générer
        window_size: Taille de la fenêtre d'observation
    
    Returns:
        price_data, feature_data: DataFrames des prix et des caractéristiques
    """
    logger.info(f"Génération de données synthétiques avec {num_samples} échantillons")
    
    # Générer des dates
    dates = pd.date_range(start="2023-01-01", periods=num_samples, freq="h")
    
    # Générer des prix avec un mouvement brownien
    np.random.seed(42)
    price = 100.0
    prices = [price]
    for _ in range(1, num_samples):
        price = price * (1 + np.random.normal(0, 0.01))
        prices.append(price)
    
    # Créer le DataFrame des prix
    price_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'volume': np.random.exponential(100, size=num_samples)
    }, index=dates)
    
    # Générer des caractéristiques techniques
    feature_data = pd.DataFrame(index=dates)
    
    # Ajouter des indicateurs techniques simulés
    feature_data['sma_5'] = price_data['close'].rolling(window=5).mean()
    feature_data['sma_10'] = price_data['close'].rolling(window=10).mean()
    feature_data['rsi'] = np.random.uniform(0, 100, size=num_samples)
    feature_data['macd'] = np.random.normal(0, 1, size=num_samples)
    feature_data['bollinger_upper'] = feature_data['sma_10'] + np.random.uniform(1, 2, size=num_samples)
    feature_data['bollinger_lower'] = feature_data['sma_10'] - np.random.uniform(1, 2, size=num_samples)
    
    # Ajouter des variables catégorielles simulées
    feature_data['market_regime'] = np.random.choice([0, 1, 2], size=num_samples)
    feature_data['instrument_type'] = np.random.choice([0, 1], size=num_samples)
    
    # Simuler des embeddings LLM
    for i in range(10):  # 10 dimensions d'embedding LLM
        feature_data[f'llm_{i}'] = np.random.normal(0, 1, size=num_samples)
    
    # Simuler des caractéristiques MCP
    for i in range(5):  # 5 dimensions MCP
        feature_data[f'mcp_{i}'] = np.random.normal(0, 1, size=num_samples)
    
    # Simuler des probabilités HMM
    feature_data['hmm_regime'] = np.random.choice([0, 1, 2], size=num_samples)
    feature_data['hmm_prob_0'] = np.random.uniform(0, 1, size=num_samples)
    feature_data['hmm_prob_1'] = np.random.uniform(0, 1, size=num_samples)
    feature_data['hmm_prob_2'] = np.random.uniform(0, 1, size=num_samples)
    
    # Normaliser les probabilités HMM
    prob_sum = feature_data[['hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']].sum(axis=1)
    feature_data['hmm_prob_0'] = feature_data['hmm_prob_0'] / prob_sum
    feature_data['hmm_prob_1'] = feature_data['hmm_prob_1'] / prob_sum
    feature_data['hmm_prob_2'] = feature_data['hmm_prob_2'] / prob_sum
    
    # Remplir les NaN
    feature_data = feature_data.fillna(method='bfill').fillna(method='ffill')
    
    logger.info(f"Données synthétiques générées: {len(feature_data)} lignes avec {len(feature_data.columns)} caractéristiques")
    return price_data, feature_data

def load_data(data_path=None):
    """
    Charge les données pour l'optimisation génétique.
    Si data_path est None, génère des données synthétiques.
    
    Args:
        data_path: Chemin vers le dataset (optionnel)
    
    Returns:
        price_data, feature_data: DataFrames des prix et des caractéristiques
    """
    if data_path is None:
        logger.info("Aucun chemin de données fourni, génération de données synthétiques")
        return generate_synthetic_data()
    
    logger.info(f"Chargement des données depuis {data_path}")
    try:
        # Vérifier si le fichier est un CSV ou un Parquet
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            logger.warning(f"Format de fichier non reconnu: {data_path}. Génération de données synthétiques à la place.")
            return generate_synthetic_data()
        
        logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
        
        # Séparer les données de prix et les caractéristiques
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        if all(col in df.columns for col in price_cols):
            price_data = df[price_cols]
            feature_data = df.drop(columns=price_cols)
        else:
            logger.warning("Colonnes de prix manquantes dans le dataset. Utilisation de toutes les colonnes comme caractéristiques.")
            # Générer des prix synthétiques basés sur l'index
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.date_range(start="2023-01-01", periods=len(df))
            price_data = pd.DataFrame({
                'open': np.random.normal(100, 1, size=len(df)),
                'high': np.random.normal(101, 1, size=len(df)),
                'low': np.random.normal(99, 1, size=len(df)),
                'close': np.random.normal(100.5, 1, size=len(df)),
                'volume': np.random.exponential(100, size=len(df))
            }, index=dates)
            feature_data = df
        
        return price_data, feature_data
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        logger.info("Génération de données synthétiques à la place")
        return generate_synthetic_data()

def create_trading_env(hyperparams, price_data, feature_data):
    """
    Crée un environnement de trading avec les hyperparamètres spécifiés.
    
    Args:
        hyperparams: Dictionnaire des hyperparamètres
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
    
    Returns:
        Environnement de trading vectorisé
    """
    from stable_baselines3.common.env_util import make_vec_env
    import pandas as pd
    
    # Extraire les hyperparamètres pertinents
    window_size = int(hyperparams.get('window_size', 30))
    initial_balance = float(hyperparams.get('initial_balance', 10000.0))
    transaction_fee = float(hyperparams.get('transaction_fee', 0.001))
    reward_scaling = float(hyperparams.get('reward_scaling', 1.0))
    max_steps = int(hyperparams.get('max_steps', 500))
    
    # Ajuster window_size et max_steps en fonction de la taille des données
    data_length = len(price_data)
    
    # S'assurer que window_size n'est pas trop grand par rapport aux données
    adjusted_window_size = min(window_size, max(1, data_length // 10))
    
    # S'assurer que max_steps permet de laisser une marge pour window_size
    # La formule garantit que window_size + max_steps < data_length
    adjusted_max_steps = min(max_steps, max(10, data_length - adjusted_window_size - 5))
    
    print(f"Taille des données: {data_length}, window_size ajusté: {adjusted_window_size}, max_steps ajusté: {adjusted_max_steps}")
    
    # Définir les poids des récompenses
    reward_weights = {
        'profit': 1.0 * reward_scaling,
        'trade_completion': 0.1 * reward_scaling,
        'trade_duration': -0.05 * reward_scaling,
        'holding_penalty': -0.01 * reward_scaling
    }
    
    # S'assurer que les colonnes requises sont présentes dans les données
    required_price_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in required_price_columns:
        if col not in price_data.columns:
            raise ValueError(f"La colonne {col} est requise dans price_data")
    
    # Vérifier que feature_data a la même longueur que price_data
    if len(price_data) != len(feature_data):
        raise ValueError("price_data et feature_data doivent avoir la même longueur")
    
    # Vérifier que toutes les colonnes de feature_data sont numériques
    non_numeric_cols = feature_data.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"Attention: Les colonnes suivantes de feature_data ne sont pas numériques: {non_numeric_cols}")
        print("Conversion des colonnes non numériques en format numérique...")
        
        # Convertir les colonnes non numériques en format numérique
        for col in non_numeric_cols:
            if col == 'timestamp':
                # Convertir timestamp en secondes depuis epoch
                feature_data['timestamp_numeric'] = pd.to_datetime(feature_data[col]).astype(np.int64) // 10**9
                feature_data = feature_data.drop(columns=[col])
            else:
                # Pour les autres colonnes, essayer de convertir en numérique ou remplacer par 0
                try:
                    feature_data[col] = pd.to_numeric(feature_data[col], errors='coerce').fillna(0)
                except Exception as e:
                    print(f"Impossible de convertir la colonne {col} en numérique: {e}. Elle sera remplacée par des zéros.")
                    feature_data[col] = 0
    
    # Fonction de création d'environnement
    def make_env():
        from model.training.reinforcement_learning import TradingEnvironment
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            window_size=adjusted_window_size,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            reward_weights=reward_weights,
            random_start=True,  # Démarrage aléatoire pour une meilleure généralisation
            max_steps=adjusted_max_steps  # Utiliser la valeur ajustée
        )
        return env
    
    # Créer l'environnement vectorisé
    vec_env = make_vec_env(make_env, n_envs=1)
    
    return vec_env

def create_rl_agent(hyperparams, env, output_dir):
    """
    Crée un agent RL avec les hyperparamètres spécifiés.
    
    Args:
        hyperparams: Dictionnaire des hyperparamètres
        env: Environnement de trading vectorisé
        output_dir: Répertoire de sortie pour les résultats
    
    Returns:
        Agent RL
    """
    from model.training.reinforcement_learning import TradingRLAgent as RLAgent
    
    # Extraire les hyperparamètres de l'agent
    learning_rate = float(hyperparams.get('learning_rate', 0.0003))
    gamma = float(hyperparams.get('gamma', 0.99))
    gae_lambda = float(hyperparams.get('gae_lambda', 0.95))
    batch_size = int(hyperparams.get('batch_size', 64))
    n_steps = int(hyperparams.get('n_steps', 2048))
    n_epochs = int(hyperparams.get('n_epochs', 10))
    clip_range = float(hyperparams.get('clip_range', 0.2))
    
    # Extraire les hyperparamètres du modèle hybride
    cnn_filters = int(hyperparams.get('cnn_filters', 64))
    cnn_kernel_size = int(hyperparams.get('cnn_kernel_size', 3))
    lstm_units = int(hyperparams.get('lstm_units', 128))
    dropout_rate = float(hyperparams.get('dropout_rate', 0.2))
    
    # Configurer les paramètres de la politique
    policy_kwargs = {
        'use_cnn_lstm': True,
        'use_cot': True,
        'net_arch': {
            'cnn_filters': cnn_filters,
            'cnn_kernel_size': cnn_kernel_size,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate
        }
    }
    
    # Créer l'agent RL
    agent = RLAgent(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(output_dir, 'tensorboard')
    )
    
    # Initialiser l'agent avec l'environnement
    agent.create_agent(env)
    
    return agent

def evaluate_hyperparams(hyperparams, price_data, feature_data, output_dir, train_timesteps=10000):
    """
    Évalue les hyperparamètres en entraînant et validant un agent RL.
    
    Args:
        hyperparams: Dictionnaire des hyperparamètres
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
        output_dir: Répertoire de sortie pour les logs et modèles
        train_timesteps: Nombre d'étapes d'entraînement
    
    Returns:
        Score de fitness (plus élevé est meilleur)
    """
    # Créer un sous-répertoire unique pour cette évaluation
    eval_id = int(time.time() * 1000) % 10000  # ID unique basé sur le temps
    eval_dir = os.path.join(output_dir, f"eval_{eval_id}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Créer l'environnement de trading
    env = create_trading_env(hyperparams, price_data, feature_data)
    
    # Créer l'agent RL
    agent = create_rl_agent(hyperparams, env, eval_dir)
    
    # Entraîner l'agent (avec un nombre réduit d'étapes pour l'optimisation)
    logger.info(f"Évaluation des hyperparamètres: {hyperparams}")
    logger.info(f"Entraînement pour {train_timesteps} étapes")
    
    try:
        agent.train(total_timesteps=train_timesteps)
        
        # Évaluer l'agent
        metrics, _, _ = agent.evaluate(n_episodes=5)
        
        # Calculer le score de fitness (plus élevé est meilleur)
        # Nous voulons maximiser la récompense moyenne et le ratio de Sharpe, et minimiser le drawdown
        mean_reward = metrics.get('mean_reward', 0)
        mean_sharpe = metrics.get('mean_sharpe', 0)
        mean_drawdown = metrics.get('mean_drawdown', 1)  # Éviter la division par zéro
        total_trades = metrics.get('total_trades', 0)
        
        # Pénaliser les agents qui ne font pas de transactions
        trade_penalty = 0 if total_trades > 5 else -5
        
        # Formule de fitness combinant plusieurs métriques
        fitness = mean_reward + 2 * mean_sharpe - 3 * mean_drawdown + trade_penalty
        
        logger.info(f"Fitness: {fitness:.4f}, Reward: {mean_reward:.4f}, Sharpe: {mean_sharpe:.4f}, Drawdown: {mean_drawdown:.4f}, Trades: {total_trades}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation des hyperparamètres: {e}")
        fitness = -100  # Valeur de fitness très basse en cas d'erreur
    
    return (fitness,)

def decode_individual(individual):
    """
    Décode un individu en un dictionnaire d'hyperparamètres.
    Du00e9code un individu en un dictionnaire d'hyperparamu00e8tres.
    
    Args:
        individual: Liste de valeurs d'hyperparamu00e8tres
    
    Returns:
        Dictionnaire d'hyperparamu00e8tres
    """
    hyperparams = {}
    idx = 0
    
    for name, (min_val, max_val) in HYPERPARAM_RANGES.items():
        if name == 'batch_size' or name == 'use_batch_norm':
            # Valeurs discru00e8tes
            if name == 'batch_size':
                # Puissance de 2 pour la taille du batch
                hyperparams[name] = int(2 ** (min_val + individual[idx] * (max_val - min_val)))
            else:
                # Valeur binu00e9aire pour use_batch_norm
                hyperparams[name] = int(round(individual[idx]))
        else:
            # Valeurs continues
            hyperparams[name] = min_val + individual[idx] * (max_val - min_val)
        
        idx += 1
    
    return hyperparams

def optimize_hyperparams(price_data, feature_data, output_dir, population_size=POPULATION_SIZE, generations=GENERATIONS, train_timesteps=10000):
    """
    Optimise les hyperparamètres de l'agent RL avec un algorithme génétique.
    
    Args:
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
        output_dir: Répertoire de sortie pour les résultats
        population_size: Taille de la population
        generations: Nombre de générations
        train_timesteps: Nombre d'étapes d'entraînement pour chaque évaluation
    
    Returns:
        Meilleurs hyperparamètres trouvés
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer les types pour l'algorithme génétique
    if 'FitnessMax' not in dir(creator):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if 'Individual' not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialiser la toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(HYPERPARAM_RANGES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Définir les opérateurs génétiques
    toolbox.register("evaluate", lambda ind: evaluate_hyperparams(decode_individual(ind), price_data, feature_data, output_dir, train_timesteps=train_timesteps))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Créer la population initiale
    pop = toolbox.population(n=population_size)
    
    # Statistiques à collecter
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of Fame pour garder les meilleurs individus
    hof = tools.HallOfFame(1)
    
    # Exécuter l'algorithme génétique
    logger.info(f"Démarrage de l'optimisation des hyperparamètres avec {population_size} individus et {generations} générations")
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=generations, stats=stats, halloffame=hof, verbose=True)
    
    # Récupérer les meilleurs hyperparamètres
    best_individual = hof[0]
    best_hyperparams = decode_individual(best_individual)
    best_fitness = best_individual.fitness.values[0]
    
    logger.info(f"Meilleurs hyperparamètres trouvés: {best_hyperparams}")
    logger.info(f"Meilleur fitness: {best_fitness}")
    
    # Sauvegarder les résultats
    results = {
        'best_hyperparams': best_hyperparams,
        'best_fitness': float(best_fitness),
        'generations': generations,
        'population_size': population_size,
        'train_timesteps': train_timesteps,
        'logbook': [{
            'gen': gen,
            'avg': float(avg),
            'min': float(min_val),
            'max': float(max_val)
        } for gen, avg, min_val, max_val in zip(
            range(len(logbook)), 
            logbook.select('avg'), 
            logbook.select('min'), 
            logbook.select('max')
        )]
    }
    
    with open(os.path.join(output_dir, 'genetic_optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_hyperparams

def train_best_agent(best_hyperparams, price_data, feature_data, output_dir, train_timesteps=50000):
    """
    Entraîne l'agent RL avec les meilleurs hyperparamètres trouvés.
    
    Args:
        best_hyperparams: Meilleurs hyperparamètres trouvés
        price_data: DataFrame des prix
        feature_data: DataFrame des caractéristiques
        output_dir: Répertoire de sortie pour les résultats
        train_timesteps: Nombre d'étapes d'entraînement
    
    Returns:
        Agent RL entraîné
    """
    # Créer l'environnement de trading
    env = create_trading_env(best_hyperparams, price_data, feature_data)
    
    # Créer l'agent RL
    best_agent = create_rl_agent(best_hyperparams, env, output_dir)
    
    # Entraîner l'agent avec plus d'étapes
    logger.info(f"Entraînement de l'agent final avec les meilleurs hyperparamètres pour {train_timesteps} étapes")
    best_agent.train(total_timesteps=train_timesteps)
    
    # Évaluer l'agent final
    logger.info("\u00c9valuation de l'agent final")
    metrics, _, _ = best_agent.evaluate(n_episodes=10)
    
    # Afficher les métriques
    logger.info("=== Métriques de l'agent final ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value:.4f}")
    
    # Sauvegarder les métriques
    with open(os.path.join(output_dir, 'final_agent_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.number, float)) else x)
    
    return best_agent

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimise les hyperparamètres de l\'agent RL avec un algorithme génétique.')
    parser.add_argument('--data-path', type=str, help='Chemin vers le dataset (CSV ou Parquet)')
    parser.add_argument('--output-dir', type=str, default='output/genetic_optimizer', help='Répertoire de sortie pour les résultats')
    parser.add_argument('--population-size', type=int, default=POPULATION_SIZE, help='Taille de la population')
    parser.add_argument('--generations', type=int, default=GENERATIONS, help='Nombre de générations')
    parser.add_argument('--train-timesteps', type=int, default=10000, help="Nombre d'étapes d'entraînement pour chaque évaluation")
    parser.add_argument('--final-timesteps', type=int, default=50000, help="Nombre d'étapes d'entraînement pour l'agent final")
    
    args = parser.parse_args()
    
    # Charger les données
    price_data, feature_data = load_data(args.data_path)
    
    # Optimiser les hyperparamètres
    best_hyperparams = optimize_hyperparams(
        price_data, 
        feature_data,
        args.output_dir,
        args.population_size,
        args.generations,
        args.train_timesteps
    )
    
    # Entraîner l'agent final avec les meilleurs hyperparamètres
    best_agent = train_best_agent(
        best_hyperparams,
        price_data,
        feature_data,
        args.output_dir,
        args.final_timesteps
    )
    
    # Sauvegarder l'agent final
    best_agent.save(os.path.join(args.output_dir, 'best_rl_agent'))
    logger.info(f"Agent RL final sauvegardé dans {os.path.join(args.output_dir, 'best_rl_agent')}")
    
    logger.info("Optimisation des hyperparamètres terminée avec succès")

if __name__ == "__main__":
    main()
