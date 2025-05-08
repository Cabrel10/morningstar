# File: model/training/genetic_optimizer.py
# Location: <repo_root>/model/training/genetic_optimizer.py

import os
import sys # Ajout de sys
# Ajout pour que 'from model...' fonctionne, en priorité
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) 
import random
import json
import multiprocessing
from deap import base, creator, tools, algorithms # Importer algorithms
import numpy as np
import pandas as pd
from model.training.reinforcement_learning import create_trading_env_from_data, TradingRLAgent
from config.config import Config

# ----- Configuration et lecture des données -----

def load_price_and_features(data_path):
    """
    Charge les données historiques depuis un Parquet ou CSV.
    Retourne deux DataFrames : price_data (OHLCV) et feature_data.
    """
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True) # Assurer que timestamp est l'index si CSV

    # Séparer price_data et feature_data
    price_cols = ['open', 'high', 'low', 'close', 'volume'] # Timestamp est l'index
    
    # Vérifier que les colonnes existent avant de les sélectionner
    missing_price_cols = [col for col in price_cols if col not in df.columns]
    if missing_price_cols:
        raise ValueError(f"Colonnes de prix manquantes dans le dataset: {missing_price_cols}")
        
    price_data = df[price_cols].copy()
    # Ajouter l'index (timestamp) comme colonne
    price_data['timestamp'] = df.index 
    
    # Exclure les colonnes de prix pour obtenir les features
    # Note: L'index (timestamp) n'est pas dans df.columns, donc pas besoin de l'exclure explicitement ici
    feature_columns = [col for col in df.columns if col not in price_cols]
    feature_data = df[feature_columns].copy()
    
    # S'assurer que feature_data ne contient que des colonnes numériques
    feature_data = feature_data.select_dtypes(include=[np.number])
    
    # Remplacer les NaN potentiels (si select_dtypes ne les gère pas)
    feature_data.fillna(0, inplace=True) 
    
    return price_data, feature_data

# ----- Fitness function pour GA -----

def evaluate_individual(individual, price_data, feature_data, cfg):
    """
    Évalue un individu (ensemble d'hyperparamètres) :
    - Crée env RL
    - Entraîne brièvement l'agent
    - Retourne le Sharpe ratio moyen sur un sous-ensemble de validation
    """
    # Décoder les hyperparamètres
    params = {
        'learning_rate': individual[0],
        'n_steps': int(individual[1]),
        'batch_size': int(individual[2]),
        'n_epochs': int(individual[3]),
        'gamma': individual[4],
        'gae_lambda': individual[5],
        'clip_range': individual[6]
    }
    # Créer env
    env = create_trading_env_from_data(
        price_data=price_data,
        feature_data=feature_data,
        window_size=cfg.get_config('model.window_size', 60)
    )
    # Initialiser agent wrapper
    agent_wrapper = TradingRLAgent(**params, verbose=0) 
    # Créer l'agent PPO interne
    agent = agent_wrapper.create_agent(env) 
    # Entraînement court pour mesurer fitness
    eval_timesteps = cfg.get_config('ga.eval_timesteps', 50000)
    agent_wrapper.train( # Appeler train sur le wrapper
        total_timesteps=eval_timesteps,
        eval_freq=eval_timesteps, # Évaluer à la fin de l'entraînement court
        n_eval_episodes=cfg.get_config('ga.eval_episodes', 3)
    )
    # Évaluer récompense ajustée (Sharpe)
    # Note: agent_wrapper.evaluate retourne (metrics, trades, infos)
    metrics, _, _ = agent_wrapper.evaluate(n_episodes=cfg.get_config('ga.eval_episodes', 3))
    # Utiliser une métrique pertinente comme fitness, par exemple mean_reward ou mean_sharpe
    fitness_value = metrics.get('mean_reward', -np.inf) # Utiliser mean_reward par défaut
    # Ou utiliser Sharpe si disponible et préféré:
    # fitness_value = metrics.get('mean_sharpe', -np.inf) 
    
    # DEAP attend un tuple pour la fitness
    return (fitness_value,) 

# ----- Fonction principale d'optimisation -----

def optimize_hyperparams(
    data_path,
    population_size=20,
    generations=10,
    seed=42,
    output_dir='outputs/ga'
):
    """
    Lance l'optimisation génétique pour trouver les meilleurs hyperparamètres RL.
    Sauve les résultats dans output_dir.
    """
    # Préparer répertoire
    os.makedirs(output_dir, exist_ok=True)
    # Lecture données
    price_data, feature_data = load_price_and_features(data_path)
    # Charger config
    cfg = Config()
    random.seed(seed)

    # Définition de l'espace des individus
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Hyperparamètres à optimiser
    toolbox.register('attr_lr', random.uniform, 1e-5, 1e-3)
    toolbox.register('attr_n_steps', random.randint, 512, 4096)
    toolbox.register('attr_batch', random.randint, 16, 256)
    toolbox.register('attr_epochs', random.randint, 1, 10)
    toolbox.register('attr_gamma', random.uniform, 0.9, 0.9999)
    toolbox.register('attr_gae', random.uniform, 0.8, 0.99)
    toolbox.register('attr_clip', random.uniform, 0.1, 0.3)
    toolbox.register('attr_pos_size', random.uniform, 0.001, 0.1)       # 0.1% à 10% du capital
    toolbox.register('attr_sl_pct',    random.uniform, 0.001, 0.05)     # 0.1% à 5%
    toolbox.register('attr_tp_pct',    random.uniform, 0.001, 0.1)      # 0.1% à 10%
    toolbox.register('attr_order_type', random.randint, 0, 2)          # 0=market,1=limit,2=iceberg
    toolbox.register('attr_slippage',  random.uniform, 0.0, 0.005)     # 0% à 0.5%
    toolbox.register('attr_cot_weight',random.uniform, 0.0, 1.0)       # 0 (pas de CoT) à 1 (full CoT)

    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_lr, toolbox.attr_n_steps, toolbox.attr_batch,
                      toolbox.attr_epochs, toolbox.attr_gamma, toolbox.attr_gae,
                      toolbox.attr_clip), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Register GA operators
    toolbox.register('evaluate', evaluate_individual,
                    price_data=price_data, feature_data=feature_data, cfg=cfg)
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)

    # Exécuter GA
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('max', np.max)

    pop, logbook = algorithms.eaSimple( # Utiliser algorithms.eaSimple
        pop, toolbox,
        cxpb=0.5, mutpb=0.2,
        ngen=generations,
        stats=stats, halloffame=hof,
        verbose=True
    )

    # Sauvegarde des résultats
    best = hof[0]
    best_params = {
        'learning_rate': best[0],
        'n_steps': int(best[1]),
        'batch_size': int(best[2]),
        'n_epochs': int(best[3]),
        'gamma': best[4],
        'gae_lambda': best[5],
        'clip_range': best[6],
        'position_size': best[7],
        'stop_loss_pct': best[8],
        'take_profit_pct': best[9],
        'order_type': int(best[10]),
        'slippage_tolerance': best[11],
        'cot_weight': best[12],
    }
    with open(os.path.join(output_dir, 'best_hyperparams.json'), 'w') as f:
        json.dump(best_params, f, indent=2)

    return best_params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('GA Hyperparam Optimization')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--pop', type=int, default=20)
    parser.add_argument('--gen', type=int, default=10)
    parser.add_argument('--out', type=str, default='outputs/ga')
    args = parser.parse_args()

    best = optimize_hyperparams(
        data_path=args.data_path,
        population_size=args.pop,
        generations=args.gen,
        output_dir=args.out
    )
    print(f"Best hyperparams: {best}")
