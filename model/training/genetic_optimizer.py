#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimisation des hyperparamu00e8tres du modu00e8le Morningstar avec un algorithme gu00e9nu00e9tique.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
import random
import json
from pathlib import Path
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from model.architecture.simplified_model import build_simplified_model, compile_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
POPULATION_SIZE = 20
GENERATIONS = 10
CXPB = 0.7  # Probabilitu00e9 de croisement
MUTPB = 0.2  # Probabilitu00e9 de mutation
TOURNAMENT_SIZE = 3

# Plages de valeurs pour les hyperparamu00e8tres
HYPERPARAM_RANGES = {
    'l2_reg': (0.0001, 0.01),
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (0.0001, 0.01),
    'batch_size': (4, 7),  # 2^4=16 u00e0 2^7=128
    'use_batch_norm': (0, 1),  # 0 = False, 1 = True
}

def load_data(data_path):
    """
    Charge les donnu00e9es normalisu00e9es et les divise en ensembles d'entrau00eenement et de validation.
    
    Args:
        data_path: Chemin vers le dataset normalisu00e9
    
    Returns:
        X_train, X_val, y_train, y_val: Dictionnaires de features et labels
    """
    logger.info(f"Chargement des donnu00e9es depuis {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset chargu00e9 avec {len(df)} lignes et {len(df.columns)} colonnes")
    
    # Su00e9parer les ensembles d'entrau00eenement et de validation
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    test_df = df[df['split'] == 'test'].drop(columns=['split'])
    logger.info(f"Ensemble d'entrau00eenement: {len(train_df)} lignes, Ensemble de test: {len(test_df)} lignes")
    
    # Diviser l'ensemble d'entrau00eenement en entrau00eenement et validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    logger.info(f"Ensemble d'entrau00eenement final: {len(train_df)} lignes, Ensemble de validation: {len(val_df)} lignes")
    
    # Extraire les features et labels
    # Colonnes techniques
    technical_cols = [col for col in train_df.columns if col not in [
        'market_regime', 'level_sl', 'level_tp', 'instrument_type',
        'hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2'
    ] and not col.startswith('llm_') and not col.startswith('mcp_')]
    
    # Colonnes LLM
    llm_cols = [col for col in train_df.columns if col.startswith('llm_')]
    
    # Colonnes MCP
    mcp_cols = [col for col in train_df.columns if col.startswith('mcp_')]
    
    # Colonnes HMM
    hmm_cols = ['hmm_regime', 'hmm_prob_0', 'hmm_prob_1', 'hmm_prob_2']
    
    # Pru00e9parer les dictionnaires de features
    X_train = {
        'technical_input': train_df[technical_cols].values,
        'llm_input': train_df[llm_cols].values,
        'mcp_input': train_df[mcp_cols].values,
        'hmm_input': train_df[hmm_cols].values,
        'instrument_input': train_df[['instrument_type']].values
    }
    
    X_val = {
        'technical_input': val_df[technical_cols].values,
        'llm_input': val_df[llm_cols].values,
        'mcp_input': val_df[mcp_cols].values,
        'hmm_input': val_df[hmm_cols].values,
        'instrument_input': val_df[['instrument_type']].values
    }
    
    # Pru00e9parer les dictionnaires de labels
    y_train = {
        'market_regime': train_df['market_regime'].values,
        'sl_tp': train_df[['level_sl', 'level_tp']].values
    }
    
    y_val = {
        'market_regime': val_df['market_regime'].values,
        'sl_tp': val_df[['level_sl', 'level_tp']].values
    }
    
    return X_train, X_val, y_train, y_val, X_val, y_val

def create_model(hyperparams):
    """
    Cru00e9e un modu00e8le avec les hyperparamu00e8tres spu00e9cifiu00e9s.
    
    Args:
        hyperparams: Dictionnaire d'hyperparamu00e8tres
    
    Returns:
        Modu00e8le Keras compilu00e9
    """
    model = build_simplified_model(
        tech_input_shape=(44,),
        llm_embedding_dim=768,
        mcp_input_dim=128,
        hmm_input_dim=4,
        instrument_vocab_size=10,
        instrument_embedding_dim=8,
        num_market_regime_classes=4,
        num_sl_tp_outputs=2,
        l2_reg=hyperparams['l2_reg'],
        dropout_rate=hyperparams['dropout_rate'],
        use_batch_norm=bool(hyperparams['use_batch_norm'])
    )
    
    model = compile_model(model, learning_rate=hyperparams['learning_rate'])
    
    return model

def evaluate_hyperparams(hyperparams, X_train, X_val, y_train, y_val, epochs=5):
    """
    u00c9value les hyperparamu00e8tres en entrau00eenant un modu00e8le et en le validant.
    
    Args:
        hyperparams: Dictionnaire d'hyperparamu00e8tres
        X_train, X_val, y_train, y_val: Donnu00e9es d'entrau00eenement et de validation
        epochs: Nombre d'u00e9poques d'entrau00eenement
    
    Returns:
        Score de fitness (plus u00e9levu00e9 est meilleur)
    """
    try:
        # Cru00e9er le modu00e8le
        model = create_model(hyperparams)
        
        # Entrau00eener le modu00e8le
        batch_size = int(hyperparams['batch_size'])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # u00c9valuer le modu00e8le
        val_loss = history.history['val_loss'][-1]
        val_market_regime_accuracy = history.history.get('val_market_regime_accuracy', [0])[-1]
        val_sl_tp_rmse = history.history.get('val_sl_tp_rmse', [float('inf')])[-1]
        
        # Calculer le score de fitness (plus u00e9levu00e9 est meilleur)
        # Nous voulons maximiser l'accuracy et minimiser la RMSE
        fitness = val_market_regime_accuracy - 0.1 * val_sl_tp_rmse
        
        # Nettoyer la mu00e9moire
        tf.keras.backend.clear_session()
        
        return (fitness,)
    except Exception as e:
        logger.error(f"Erreur lors de l'u00e9valuation des hyperparamu00e8tres: {e}")
        return (-float('inf'),)

def decode_individual(individual):
    """
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

def optimize_hyperparams(X_train, X_val, y_train, y_val, output_dir, population_size=POPULATION_SIZE, generations=GENERATIONS):
    """
    Optimise les hyperparamu00e8tres avec un algorithme gu00e9nu00e9tique.
    
    Args:
        X_train, X_val, y_train, y_val: Donnu00e9es d'entrau00eenement et de validation
        output_dir: Ru00e9pertoire de sortie pour les ru00e9sultats
        population_size: Taille de la population
        generations: Nombre de gu00e9nu00e9rations
    
    Returns:
        Meilleurs hyperparamu00e8tres trouvu00e9s
    """
    # Cru00e9er les types pour l'algorithme gu00e9nu00e9tique
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialiser la toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(HYPERPARAM_RANGES))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Du00e9finir les opu00e9rateurs gu00e9nu00e9tiques
    toolbox.register("evaluate", lambda ind: evaluate_hyperparams(decode_individual(ind), X_train, X_val, y_train, y_val))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    
    # Cru00e9er la population initiale
    pop = toolbox.population(n=population_size)
    
    # Statistiques u00e0 collecter
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of Fame pour garder les meilleurs individus
    hof = tools.HallOfFame(1)
    
    # Exu00e9cuter l'algorithme gu00e9nu00e9tique
    logger.info(f"Du00e9marrage de l'optimisation des hyperparamu00e8tres avec {population_size} individus et {generations} gu00e9nu00e9rations")
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=generations, stats=stats, halloffame=hof, verbose=True)
    
    # Ru00e9cupu00e9rer les meilleurs hyperparamu00e8tres
    best_individual = hof[0]
    best_hyperparams = decode_individual(best_individual)
    best_fitness = best_individual.fitness.values[0]
    
    logger.info(f"Meilleurs hyperparamu00e8tres trouvu00e9s: {best_hyperparams}")
    logger.info(f"Meilleur fitness: {best_fitness}")
    
    # Sauvegarder les ru00e9sultats
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'best_hyperparams': best_hyperparams,
        'best_fitness': best_fitness,
        'logbook': logbook
    }
    
    with open(os.path.join(output_dir, 'genetic_optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    return best_hyperparams

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimise les hyperparamu00e8tres du modu00e8le Morningstar avec un algorithme gu00e9nu00e9tique.')
    parser.add_argument('--data-path', type=str, required=True, help='Chemin vers le dataset normalisu00e9')
    parser.add_argument('--output-dir', type=str, default='model/improved', help='Ru00e9pertoire de sortie pour les ru00e9sultats')
    parser.add_argument('--population-size', type=int, default=POPULATION_SIZE, help='Taille de la population')
    parser.add_argument('--generations', type=int, default=GENERATIONS, help='Nombre de gu00e9nu00e9rations')
    
    args = parser.parse_args()
    
    # Charger les donnu00e9es
    X_train, X_val, y_train, y_val, X_test, y_test = load_data(args.data_path)
    
    # Optimiser les hyperparamu00e8tres
    best_hyperparams = optimize_hyperparams(
        X_train, X_val, y_train, y_val,
        args.output_dir,
        args.population_size,
        args.generations
    )
    
    # Entrau00eener le modu00e8le final avec les meilleurs hyperparamu00e8tres
    logger.info("Entrau00eenement du modu00e8le final avec les meilleurs hyperparamu00e8tres")
    final_model = create_model(best_hyperparams)
    
    final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=int(best_hyperparams['batch_size']),
        verbose=1
    )
    
    # u00c9valuer le modu00e8le final
    logger.info("u00c9valuation du modu00e8le final sur l'ensemble de test")
    test_results = final_model.evaluate(X_test, y_test, verbose=1)
    
    # Sauvegarder les ru00e9sultats de l'u00e9valuation
    test_metrics = {}
    for i, metric_name in enumerate(final_model.metrics_names):
        test_metrics[metric_name] = float(test_results[i])
        logger.info(f"{metric_name}: {test_results[i]}")
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Sauvegarder le modu00e8le final
    final_model.save(os.path.join(args.output_dir, 'morningstar_improved.h5'))
    logger.info(f"Modu00e8le final sauvegardu00e9 dans {os.path.join(args.output_dir, 'morningstar_improved.h5')}")

if __name__ == "__main__":
    main()
