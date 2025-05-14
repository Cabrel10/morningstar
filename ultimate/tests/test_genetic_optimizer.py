#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module model/training/genetic_optimizer.py
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Importer les fonctions à tester
from model.training.genetic_optimizer import load_price_and_features, evaluate_individual
from config.config import Config  # Pour mocker la config

# --- Fixtures ---


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Crée un répertoire temporaire pour les fichiers de test."""
    return tmp_path_factory.mktemp("ga_test_data")


@pytest.fixture(scope="module")
def sample_ga_parquet_file(test_data_dir):
    """Crée un fichier Parquet valide pour les tests GA."""
    file_path = test_data_dir / "ga_sample_valid.parquet"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "open": [100, 101, 102, 103],
            "high": [102, 103, 104, 105],
            "low": [99, 100, 101, 102],
            "close": [101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300],
            "feature_1": [0.5, 0.6, 0.7, 0.8],
            "feature_2": [1, 0, 1, 0],
            "non_numeric_feature": ["a", "b", "a", "c"],  # Pour tester le filtrage
        }
    ).set_index(
        "timestamp"
    )  # Mettre timestamp comme index
    df.to_parquet(file_path)
    return file_path


@pytest.fixture(scope="module")
def sample_ga_csv_file(test_data_dir):
    """Crée un fichier CSV valide pour les tests GA."""
    file_path = test_data_dir / "ga_sample_valid.csv"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
            "open": [100, 101, 102, 103],
            "high": [102, 103, 104, 105],
            "low": [99, 100, 101, 102],
            "close": [101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300],
            "feature_1": [0.5, 0.6, 0.7, 0.8],
        }
    ).set_index(
        "timestamp"
    )  # Mettre timestamp comme index
    df.to_csv(file_path)
    return file_path


@pytest.fixture
def mock_ga_config(mocker):
    """Mocker la classe Config pour les tests GA."""
    mock = mocker.patch("model.training.genetic_optimizer.Config", autospec=True)
    instance = mock.return_value
    config_values = {
        "model.window_size": 5,  # Petite window pour les tests
        "ga.eval_timesteps": 100,  # Entraînement très court
        "ga.eval_episodes": 1,  # Une seule éval
    }
    instance.get_config.side_effect = lambda key, default=None: config_values.get(key, default)
    return instance


@pytest.fixture
def mock_trading_env(mocker):
    """Mocker l'environnement de trading RL."""
    # Mocker la fonction de création pour retourner un mock
    mock_env = MagicMock()
    mock_env.observation_space = MagicMock()
    mock_env.action_space = MagicMock()
    mocker.patch("model.training.genetic_optimizer.create_trading_env_from_data", return_value=mock_env)
    return mock_env


@pytest.fixture
def mock_trading_agent(mocker):
    """Mocker l'agent de trading RL et ses méthodes."""
    # Mocker la classe TradingRLAgent
    mock_agent_class = mocker.patch("model.training.genetic_optimizer.TradingRLAgent", autospec=True)
    # Mocker l'instance retournée par le constructeur
    mock_agent_instance = mock_agent_class.return_value

    # Mocker les méthodes de l'instance
    mock_agent_instance.create_agent.return_value = MagicMock()  # L'agent PPO interne
    mock_agent_instance.train = MagicMock()  # Méthode train
    # Faire retourner des métriques par evaluate
    mock_agent_instance.evaluate.return_value = ({"mean_reward": 0.75, "mean_sharpe": 1.5}, [], [])

    # Retourner la CLASSE mockée pour pouvoir vérifier son appel (__init__)
    return mock_agent_class


@pytest.fixture
def sample_individual():
    """Crée un individu GA valide (liste d'hyperparamètres)."""
    # lr, n_steps, batch, epochs, gamma, gae, clip
    return [1e-4, 1024, 64, 5, 0.99, 0.95, 0.2]


# --- Tests pour load_price_and_features ---


def test_load_price_features_parquet(sample_ga_parquet_file):
    """Teste le chargement depuis Parquet."""
    price_data, feature_data = load_price_and_features(str(sample_ga_parquet_file))

    assert isinstance(price_data, pd.DataFrame)
    assert isinstance(feature_data, pd.DataFrame)
    assert list(price_data.columns) == ["open", "high", "low", "close", "volume", "timestamp"]
    assert "feature_1" in feature_data.columns
    assert "feature_2" in feature_data.columns
    assert "non_numeric_feature" not in feature_data.columns  # Doit être filtrée
    assert price_data.shape[0] == 4
    assert feature_data.shape[0] == 4
    assert not feature_data.isnull().values.any()  # Vérifier pas de NaN


def test_load_price_features_csv(sample_ga_csv_file):
    """Teste le chargement depuis CSV."""
    price_data, feature_data = load_price_and_features(str(sample_ga_csv_file))

    assert isinstance(price_data, pd.DataFrame)
    assert isinstance(feature_data, pd.DataFrame)
    assert list(price_data.columns) == ["open", "high", "low", "close", "volume", "timestamp"]
    assert "feature_1" in feature_data.columns
    assert price_data.shape[0] == 4
    assert feature_data.shape[0] == 4
    assert isinstance(price_data["timestamp"].iloc[0], pd.Timestamp)  # Vérifier type timestamp


def test_load_price_features_missing_price_col(sample_ga_parquet_file):
    """Teste l'erreur si une colonne de prix manque."""
    df = pd.read_parquet(sample_ga_parquet_file)
    df_missing = df.drop(columns=["close"])  # Enlever 'close'
    temp_path = sample_ga_parquet_file.parent / "missing_col.parquet"
    df_missing.to_parquet(temp_path)

    with pytest.raises(ValueError, match="Colonnes de prix manquantes"):
        load_price_and_features(str(temp_path))


def test_load_price_features_file_not_found(test_data_dir):
    """Teste FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_price_and_features(str(test_data_dir / "non_existent.parquet"))


# --- Tests pour evaluate_individual ---


def test_evaluate_individual_success(
    sample_individual, sample_ga_parquet_file, mock_ga_config, mock_trading_env, mock_trading_agent
):
    """Teste une évaluation réussie."""
    price_data, feature_data = load_price_and_features(str(sample_ga_parquet_file))

    fitness = evaluate_individual(sample_individual, price_data, feature_data, mock_ga_config)

    # Vérifier que l'agent a été appelé avec les bons params décodés
    expected_params = {
        "learning_rate": sample_individual[0],
        "n_steps": int(sample_individual[1]),
        "batch_size": int(sample_individual[2]),
        "n_epochs": int(sample_individual[3]),
        "gamma": sample_individual[4],
        "gae_lambda": sample_individual[5],
        "clip_range": sample_individual[6],
        "verbose": 0,  # Ajouté par la fonction
    }
    # Vérifier que le constructeur de l'instance mockée a été appelé avec les bons params
    # Note: La fixture mock_trading_agent retourne déjà l'instance mockée.
    # L'appel au constructeur est vérifié via mocker.patch lui-même.
    # Nous devons vérifier l'appel sur l'instance retournée par la fixture.
    mock_trading_agent.assert_called_once_with(**expected_params)  # Vérifie l'appel au constructeur (__init__)

    # Vérifier que train et evaluate ont été appelés sur l'INSTANCE retournée par le constructeur
    mock_instance = mock_trading_agent.return_value
    mock_instance.train.assert_called_once()
    mock_instance.evaluate.assert_called_once()

    # Vérifier la valeur de fitness retournée (basée sur mean_reward du mock)
    assert isinstance(fitness, tuple)
    assert len(fitness) == 1
    assert fitness[0] == 0.75  # mean_reward retourné par le mock


def test_evaluate_individual_uses_config(
    sample_individual, sample_ga_parquet_file, mock_ga_config, mock_trading_env, mock_trading_agent
):
    """Vérifie que les paramètres de config sont utilisés."""
    price_data, feature_data = load_price_and_features(str(sample_ga_parquet_file))

    evaluate_individual(sample_individual, price_data, feature_data, mock_ga_config)

    # Vérifier que get_config a été appelé pour les bons paramètres
    mock_ga_config.get_config.assert_any_call("model.window_size", 60)
    mock_ga_config.get_config.assert_any_call("ga.eval_timesteps", 50000)
    mock_ga_config.get_config.assert_any_call("ga.eval_episodes", 3)

    # Vérifier que train et evaluate ont été appelés sur l'INSTANCE avec les bons params
    mock_instance = mock_trading_agent.return_value
    mock_instance.train.assert_called_once_with(
        total_timesteps=100, eval_freq=100, n_eval_episodes=1  # Valeur de mock_ga_config  # Valeur de mock_ga_config
    )
    # Vérifier que evaluate a été appelé avec n_eval_episodes de la config mockée
    mock_instance.evaluate.assert_called_once_with(n_episodes=1)


def test_evaluate_individual_missing_reward(
    sample_individual, sample_ga_parquet_file, mock_ga_config, mock_trading_env, mock_trading_agent
):
    """Teste le cas où 'mean_reward' manque dans les métriques."""
    price_data, feature_data = load_price_and_features(str(sample_ga_parquet_file))

    # Configurer le mock de l'INSTANCE pour ne pas retourner 'mean_reward'
    mock_instance = mock_trading_agent.return_value
    mock_instance.evaluate.return_value = ({"mean_sharpe": 1.5}, [], [])

    fitness = evaluate_individual(sample_individual, price_data, feature_data, mock_ga_config)

    # Doit retourner -inf comme fitness par défaut
    assert fitness == (-np.inf,)


# Ajouter potentiellement des tests pour vérifier la gestion d'erreurs
# lors de la création de l'env ou de l'entraînement de l'agent, si nécessaire.

# --- Tests pour optimize_hyperparams ---

# Importer les dépendances nécessaires pour mocker DEAP
from deap import base, creator, tools, algorithms
import json
import random  # Ajout de l'import random


@patch("model.training.genetic_optimizer.load_price_and_features")
@patch("model.training.genetic_optimizer.Config")
@patch("model.training.genetic_optimizer.algorithms.eaSimple")
@patch("model.training.genetic_optimizer.tools.HallOfFame")
def test_optimize_hyperparams_zero_generations(
    mock_hof,
    mock_ea_simple,
    mock_config_class,
    mock_load_data,
    sample_ga_parquet_file,
    tmp_path,  # tmp_path pour output_dir
):
    """Teste optimize_hyperparams avec generations=0."""
    # Configurer les mocks
    mock_load_data.return_value = (pd.DataFrame({"close": [1, 2, 3]}), pd.DataFrame({"f1": [1, 2, 3]}))
    mock_config_instance = mock_config_class.return_value
    mock_config_instance.get_config.return_value = 10  # Pour ga.eval_timesteps etc.

    # Simuler un HallOfFame qui contient un individu "best"
    # L'individu doit avoir 13 éléments (nombre d'hyperparamètres optimisés)
    best_individual_data = [0.0005, 2048, 128, 5, 0.995, 0.96, 0.25, 0.05, 0.02, 0.05, 1, 0.001, 0.5]

    # Créer un mock pour l'objet FitnessMax si ce n'est pas déjà fait globalement par DEAP
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):  # S'assurer que Individual est créé
        creator.create("Individual", list, fitness=creator.FitnessMax)

    mock_best_individual = creator.Individual(best_individual_data)
    mock_best_individual.fitness.values = (0.8,)  # Fitness simulée

    mock_hof_instance = mock_hof.return_value
    mock_hof_instance.__getitem__.return_value = mock_best_individual  # hof[0] retourne cet individu

    # Simuler le retour de eaSimple (population, logbook)
    # La population retournée peut être la population initiale si generations=0
    initial_pop_mock = [creator.Individual(best_individual_data) for _ in range(5)]  # Pop de 5 individus
    for ind in initial_pop_mock:
        ind.fitness.values = (random.random(),)
    mock_ea_simple.return_value = (initial_pop_mock, "logbook_data")

    output_dir_test = tmp_path / "ga_output_zero_gen"

    # Importer la fonction à tester ici pour s'assurer que les mocks globaux sont actifs
    from model.training.genetic_optimizer import optimize_hyperparams

    best_params = optimize_hyperparams(
        data_path=str(sample_ga_parquet_file),
        population_size=5,
        generations=0,  # Test principal ici
        output_dir=str(output_dir_test),
    )

    mock_ea_simple.assert_called_once()  # Vérifier que l'algo a été appelé
    # Vérifier que le fichier JSON est créé
    assert (output_dir_test / "best_hyperparams.json").exists()

    # Vérifier le contenu du JSON (basé sur mock_best_individual)
    with open(output_dir_test / "best_hyperparams.json", "r") as f:
        saved_params = json.load(f)

    assert saved_params["learning_rate"] == best_individual_data[0]
    assert saved_params["n_steps"] == int(best_individual_data[1])
    # ... ajouter d'autres assertions pour les paramètres si nécessaire


# TODO: Ajouter un test pour vérifier la sauvegarde correcte des hyperparamètres
# en mockant HallOfFame et eaSimple pour retourner un individu spécifique.

# TODO: Ajouter des tests pour les cas limites (population_size=1, etc.)
