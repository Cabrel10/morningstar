#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests unitaires pour le module model/training/reinforcement_learning.py
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from unittest.mock import MagicMock, patch, ANY

# Importer les classes et fonctions à tester
from model.training.reinforcement_learning import (
    TradingEnvironment,
    TradingRLAgent,
    create_trading_env_from_data,
    load_and_prepare_data,
    CustomPPOPolicy,  # Importer si on veut tester son init
)
from config.config import Config  # Pour mocker la config

# --- Fixtures ---


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Crée un répertoire temporaire pour les fichiers de test RL."""
    return tmp_path_factory.mktemp("rl_test_data")


@pytest.fixture(scope="module")
def sample_rl_data_dict():
    """Données de base pour les tests RL."""
    return {
        "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "open": [100, 101, 102, 103, 104],
        "high": [102, 103, 104, 105, 106],
        "low": [99, 100, 101, 102, 103],
        "close": [101, 102, 103, 104, 105],
        "volume": [1000, 1100, 1200, 1300, 1400],
        "feature_1": [0.5, 0.6, 0.7, 0.8, 0.9],
        "feature_2": [1, 0, 1, 0, 1],
    }


@pytest.fixture(scope="module")
def sample_rl_price_data(sample_rl_data_dict):
    """DataFrame de prix pour les tests RL."""
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    return pd.DataFrame({k: sample_rl_data_dict[k] for k in cols}).set_index("timestamp")


@pytest.fixture(scope="module")
def sample_rl_feature_data(sample_rl_data_dict):
    """DataFrame de features pour les tests RL."""
    cols = ["timestamp", "feature_1", "feature_2"]
    return pd.DataFrame({k: sample_rl_data_dict[k] for k in cols}).set_index("timestamp")


@pytest.fixture(scope="module")
def sample_rl_full_parquet(test_data_dir, sample_rl_data_dict):
    """Crée un fichier Parquet complet pour load_and_prepare_data."""
    file_path = test_data_dir / "rl_full_data.parquet"
    df = pd.DataFrame(sample_rl_data_dict).set_index("timestamp")
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def mock_rl_config(mocker):
    """Mocker la classe Config pour les tests RL."""
    mock = mocker.patch("model.training.reinforcement_learning.Config", autospec=True)
    instance = mock.return_value
    instance.get_config.return_value = {}
    return instance


@pytest.fixture
def mock_ppo(mocker):
    """Mocker la classe PPO de Stable Baselines3."""
    mock = mocker.patch("model.training.reinforcement_learning.PPO", autospec=True)
    mock_instance = mock.return_value
    mock_instance.learn = MagicMock()
    mock_instance.predict = MagicMock(return_value=(1, None))
    mock_instance.save = MagicMock()
    mock.load = MagicMock(return_value=mock_instance)
    return mock


@pytest.fixture
def mock_vec_normalize(mocker):
    """Mocker VecNormalize."""
    mock = mocker.patch("model.training.reinforcement_learning.VecNormalize", autospec=True)
    mock.side_effect = lambda env, **kwargs: env
    return mock


@pytest.fixture
def mock_dummy_vec_env(mocker):
    """Mocker DummyVecEnv."""
    mock = mocker.patch("model.training.reinforcement_learning.DummyVecEnv", autospec=True)
    mock.side_effect = lambda env_fns: env_fns[0]()
    return mock


# --- Tests pour TradingEnvironment ---


def test_env_init(sample_rl_price_data, sample_rl_feature_data):
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=3)
    assert env.window_size == 3
    assert env.action_space == spaces.Discrete(3)
    assert isinstance(env.observation_space, spaces.Box)
    expected_shape = (3, sample_rl_feature_data.shape[1] + 3)
    assert env.observation_space.shape == expected_shape


def test_env_init_non_positive_prices(sample_rl_price_data, sample_rl_feature_data, caplog):
    price_data_with_invalid = sample_rl_price_data.copy()
    price_data_with_invalid.loc[price_data_with_invalid.index[1], "close"] = 0
    feature_data_for_invalid = sample_rl_feature_data.copy()
    env = TradingEnvironment(price_data_with_invalid, feature_data_for_invalid, window_size=2)
    assert "Filtrage des données" in caplog.text
    assert "1 lignes supprimées car close <= 0" in caplog.text
    assert len(env.price_data) == len(sample_rl_price_data) - 1
    assert len(env.feature_data) == len(sample_rl_feature_data) - 1
    assert env.max_steps == len(env.price_data) - env.window_size


def test_env_init_with_cot_explanations(sample_rl_price_data, sample_rl_feature_data):
    explanations = ["Explication 1", "Explication 2", "Explication 3", "Explication 4", "Explication 5"]
    env = TradingEnvironment(
        sample_rl_price_data,
        sample_rl_feature_data,
        window_size=2,
        use_cot=True,
        cot_explanations=explanations,
        random_start=False,
    )
    assert env.use_cot is True
    assert env.cot_explanations == explanations
    env.reset()
    env.step(0)
    assert len(env.explanation_history) == 1
    assert env.explanation_history[0] == "Explication 1"
    env.step(0)
    assert len(env.explanation_history) == 2
    assert env.explanation_history[1] == "Explication 2"


def test_env_reset(sample_rl_price_data, sample_rl_feature_data):
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=3)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape == env.observation_space.shape
    assert env.balance == env.initial_balance
    assert env.position == 0
    assert len(env.nav_history) == 1


def test_env_step_buy_sell(sample_rl_price_data, sample_rl_feature_data):
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=2, random_start=False)
    initial_balance = env.initial_balance
    env.step(1)
    assert env.position == 1
    assert env.balance < initial_balance
    assert len(env.trades) == 1
    assert env.trades[0]["type"] == "buy"
    env.step(2)
    assert env.position == 0
    assert len(env.trades) == 2
    assert env.trades[1]["type"] == "sell"
    assert "pnl" in env.trades[1]


def test_env_reward_calculation(sample_rl_price_data, sample_rl_feature_data):
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=2, random_start=False)
    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    env.returns = [0.01, 0.02, -0.01, 0.005, 0.015] * 2
    reward1 = env._calculate_reward(next_price=100, executed_action=0)
    assert isinstance(reward1, float)
    env.returns = [0.01] * 10
    reward2 = env._calculate_reward(next_price=100, executed_action=0)
    assert isinstance(reward2, float)
    env.returns = [0.01, 0.02]
    reward3 = env._calculate_reward(next_price=100, executed_action=0)
    assert isinstance(reward3, float)


def test_env_observation_content(sample_rl_price_data, sample_rl_feature_data):
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=2, random_start=False)
    env.reset()
    env.step(1)
    obs, _, _, _, _ = env.step(0)
    assert obs.shape == (2, sample_rl_feature_data.shape[1] + 3)
    assert obs[-1, -3] == pytest.approx(env.balance / env.initial_balance)
    assert obs[-1, -2] == env.position
    assert obs[-1, -1] == pytest.approx(env.position_price)


# --- Tests pour create_trading_env_from_data ---


def test_create_trading_env(sample_rl_price_data, sample_rl_feature_data, mock_dummy_vec_env, mock_vec_normalize):
    env = create_trading_env_from_data(sample_rl_price_data, sample_rl_feature_data, window_size=3)
    mock_dummy_vec_env.assert_called_once()
    mock_vec_normalize.assert_called_once()
    assert isinstance(env, TradingEnvironment)


# --- Tests pour TradingRLAgent ---


def test_agent_init(mock_rl_config):
    agent_wrapper = TradingRLAgent(model_path="test_model", learning_rate=1e-3, n_steps=512)
    assert agent_wrapper.model_path == "test_model"
    assert agent_wrapper.learning_rate == 1e-3
    assert agent_wrapper.n_steps == 512
    assert agent_wrapper.agent is None


def test_agent_init_with_ppo_params(mock_rl_config):
    ppo_params = {
        "learning_rate": 0.001,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 20,
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "clip_range": 0.1,
        "verbose": 0,
        "tensorboard_log": "/tmp/tb_logs_test",
    }
    agent_wrapper = TradingRLAgent(model_path="test_model_ppo_params", **ppo_params)
    assert agent_wrapper.learning_rate == 0.001
    assert agent_wrapper.n_steps == 1024
    # ... (autres assertions pour ppo_params)


def test_agent_create_agent(mock_rl_config, mock_ppo):
    env_mock = MagicMock(spec=gym.Env)
    env_mock.observation_space = MagicMock(spec=spaces.Box)
    env_mock.observation_space.shape = (5,)
    env_mock.observation_space.dtype = np.float32
    env_mock.action_space = MagicMock(spec=spaces.Discrete)
    env_mock.action_space.n = 3
    agent_wrapper = TradingRLAgent(model_path="test_model")
    agent_wrapper.create_agent(env_mock)
    mock_ppo.assert_called_once_with(
        policy=CustomPPOPolicy,
        env=env_mock,
        learning_rate=ANY,
        n_steps=ANY,
        batch_size=ANY,
        n_epochs=ANY,
        gamma=ANY,
        gae_lambda=ANY,
        clip_range=ANY,
        policy_kwargs=ANY,
        tensorboard_log=ANY,
        verbose=ANY,
    )
    assert agent_wrapper.agent is not None


@patch("os.path.exists", return_value=True)
def test_agent_create_agent_loads_existing(mock_os_exists, mock_rl_config, mock_ppo):
    env_mock = MagicMock(spec=gym.Env)
    env_mock.observation_space = MagicMock(spec=spaces.Box)
    env_mock.observation_space.shape = (5,)
    env_mock.observation_space.dtype = np.float32
    env_mock.action_space = MagicMock(spec=spaces.Discrete)
    env_mock.action_space.n = 3
    model_load_path = "existing_model_path.zip"
    agent_wrapper = TradingRLAgent(model_path=model_load_path)
    loaded_agent_mock = mock_ppo.load.return_value
    agent_wrapper.create_agent(env_mock)
    mock_ppo.load.assert_called_once_with(model_load_path, env=env_mock)
    assert agent_wrapper.agent == loaded_agent_mock


@patch("model.training.reinforcement_learning.load_and_prepare_data")
@patch("model.training.reinforcement_learning.create_trading_env_from_data")
@patch("model.training.reinforcement_learning.TradingRLAgent")
@patch("model.training.reinforcement_learning.os.makedirs")
@patch("builtins.open")
@patch("pandas.DataFrame.to_csv")
def test_train_rl_agent_no_hybrid(
    mock_to_csv,
    mock_open,
    mock_makedirs,
    mock_RLAgent_class,
    mock_create_env,
    mock_load_prepare,
    sample_rl_full_parquet,
    tmp_path,
    mock_rl_config,
):
    mock_load_prepare.return_value = (
        pd.DataFrame({"close": [1, 2, 3]}),
        pd.DataFrame({"f1": [1, 2, 3]}),
        pd.DataFrame({"close": [4, 5]}),
        pd.DataFrame({"f1": [4, 5]}),
    )
    mock_env_instance = MagicMock(spec=TradingEnvironment)
    mock_create_env.return_value = mock_env_instance
    mock_agent_instance = mock_RLAgent_class.return_value
    mock_agent_instance.evaluate.return_value = ({"mean_reward": 0.5}, [], [])
    model_save_path = str(tmp_path / "rl_model_test.zip")
    tb_log_path = str(tmp_path / "tb_logs")
    from model.training.reinforcement_learning import train_rl_agent

    train_rl_agent(
        data_path=str(sample_rl_full_parquet), model_path=model_save_path, hybrid_model_path=None, total_timesteps=100
    )
    expected_call_args = {
        "model_path": model_save_path,
        "hybrid_model_path": None,
        "use_cnn_lstm": True,
        "use_cot": True,
        "tensorboard_log": tb_log_path,
    }
    mock_RLAgent_class.assert_called_once_with(**expected_call_args)
    mock_makedirs.assert_any_call(str(tmp_path), exist_ok=True)
    mock_agent_instance.train.assert_called_once()
    mock_agent_instance.evaluate.assert_called_once()
    mock_open.assert_called_once_with(os.path.join(str(tmp_path), "evaluation_metrics.json"), "w")
    mock_to_csv.assert_called_once_with(os.path.join(str(tmp_path), "evaluation_trades.csv"), index=False)


def test_agent_train(mock_rl_config, mock_ppo):
    env_mock = MagicMock(spec=gym.Env)
    env_mock.observation_space = MagicMock(spec=spaces.Box)
    env_mock.observation_space.shape = (5,)
    env_mock.observation_space.dtype = np.float32
    env_mock.action_space = MagicMock(spec=spaces.Discrete)
    env_mock.action_space.n = 3
    agent_wrapper = TradingRLAgent(model_path="test_model")
    agent_wrapper.create_agent(env_mock)
    agent_wrapper.train(total_timesteps=1000)
    agent_wrapper.agent.learn.assert_called_once_with(total_timesteps=1000, callback=ANY)
    agent_wrapper.agent.save.assert_called_once_with("test_model")


def test_agent_evaluate(mock_rl_config, mock_ppo):
    mock_env = MagicMock(spec=TradingEnvironment)
    mock_env.reset.return_value = (np.array([0.1, 0.2]), {})
    mock_env.step.return_value = (np.array([0.3, 0.4]), 0.5, True, False, {"nav": 1005, "num_trades": 1})
    agent_wrapper = TradingRLAgent(model_path="test_model")
    agent_wrapper.create_agent(mock_env)
    mock_ppo.return_value.predict.return_value = (1, None)
    metrics, trades, infos = agent_wrapper.evaluate(n_episodes=1)
    mock_ppo.return_value.predict.assert_called()
    mock_env.step.assert_called_with(1)
    assert "mean_reward" in metrics


def test_agent_train_zero_timesteps(mock_rl_config, mock_ppo):
    env_mock = MagicMock(spec=gym.Env)
    env_mock.observation_space = MagicMock(spec=spaces.Box)
    env_mock.observation_space.shape = (5,)
    env_mock.observation_space.dtype = np.float32
    env_mock.action_space = MagicMock(spec=spaces.Discrete)
    env_mock.action_space.n = 3
    agent_wrapper = TradingRLAgent(model_path="test_model_zero_ts")
    agent_wrapper.create_agent(env_mock)
    agent_wrapper.train(total_timesteps=0)
    agent_wrapper.agent.learn.assert_called_once_with(total_timesteps=0, callback=ANY)
    agent_wrapper.agent.save.assert_called_once_with("test_model_zero_ts")


def test_agent_evaluate_zero_episodes(mock_rl_config, mock_ppo):
    mock_env = MagicMock(spec=TradingEnvironment)
    agent_wrapper = TradingRLAgent(model_path="test_model_zero_ep")
    agent_wrapper.create_agent(mock_env)
    metrics, trades, infos = agent_wrapper.evaluate(n_episodes=0)
    mock_ppo.return_value.predict.assert_not_called()
    assert metrics["mean_reward"] == 0 or np.isnan(metrics["mean_reward"])
    assert len(trades) == 0
    assert len(infos) == 0


# --- Tests pour load_and_prepare_data ---


def test_load_prepare_data(sample_rl_full_parquet):
    train_p, train_f, test_p, test_f = load_and_prepare_data(str(sample_rl_full_parquet), split_ratio=0.75)
    assert len(train_p) == 3
    assert len(test_p) == 2
    assert len(train_f) == 3
    assert len(test_f) == 2
    assert "feature_1" in train_f.columns
    assert "close" in train_p.columns
    assert not train_f.isnull().values.any()
    assert not test_f.isnull().values.any()


def test_agent_evaluate_metrics_computation(
    mock_rl_config, mock_ppo, sample_rl_price_data, sample_rl_feature_data, tmp_path
):
    """Teste le calcul des métriques dans TradingRLAgent.evaluate."""
    mock_env = MagicMock(spec=TradingEnvironment)
    step_returns = [
        (
            np.array([0.1] * 5),
            10,
            True,
            False,
            {"nav": 1010, "num_trades": 1, "win_rate": 1.0, "sharpe_ratio": 2.0, "drawdown_pct": 1.0, "total_pnl": 10},
        ),
        (
            np.array([0.1] * 5),
            -5,
            True,
            False,
            {"nav": 995, "num_trades": 1, "win_rate": 0.0, "sharpe_ratio": -1.0, "drawdown_pct": 2.0, "total_pnl": -5},
        ),
        (
            np.array([0.1] * 5),
            15,
            True,
            False,
            {"nav": 1015, "num_trades": 1, "win_rate": 1.0, "sharpe_ratio": 2.5, "drawdown_pct": 0.5, "total_pnl": 15},
        ),
    ]
    mock_env.step.side_effect = step_returns
    mock_env.reset.return_value = (
        np.zeros(mock_ppo.observation_space.shape if hasattr(mock_ppo, "observation_space") else (5,)),
        {},
    )
    agent_wrapper = TradingRLAgent(model_path=str(tmp_path / "test_eval_metrics.zip"))
    env_for_agent = MagicMock(spec=gym.Env)
    env_for_agent.observation_space = MagicMock(spec=spaces.Box)
    env_for_agent.observation_space.shape = (5,)
    env_for_agent.observation_space.dtype = np.float32
    env_for_agent.action_space = MagicMock(spec=spaces.Discrete)
    env_for_agent.action_space.n = 3
    agent_wrapper.create_agent(env_for_agent)
    agent_wrapper.env = mock_env
    mock_ppo.return_value.predict.return_value = (1, None)
    metrics, trades, infos = agent_wrapper.evaluate(n_episodes=3)
    assert len(infos) == 3
    assert metrics["mean_reward"] == pytest.approx((10 - 5 + 15) / 3)
    assert metrics["std_reward"] == pytest.approx(np.std([10, -5, 15]))
    assert metrics["min_reward"] == -5
    assert metrics["max_reward"] == 15
    assert metrics["mean_nav"] == pytest.approx(np.mean([1010, 995, 1015]))
    assert metrics["mean_sharpe"] == pytest.approx(np.mean([2.0, -1.0, 2.5]))
    assert metrics["mean_trades"] == pytest.approx(1.0)
    assert metrics["total_trades"] == 3
    assert metrics["mean_win_rate"] == pytest.approx(np.mean([1.0, 0.0, 1.0]))


def test_calculate_reward_edge_cases(sample_rl_price_data, sample_rl_feature_data):
    """Teste _calculate_reward avec des cas limites pour les poids et std_return."""
    env = TradingEnvironment(sample_rl_price_data, sample_rl_feature_data, window_size=2, random_start=False)
    env.reset()
    env.current_step = len(env.price_data) - env.window_size - 1  # Pour s'assurer qu'on a assez de returns pour sharpe

    # Cas 1: std_return = 0 (sharpe_reward devrait être 0)
    env.returns = [0.01] * 20  # Retours constants -> std = 0
    # Modifier le dictionnaire reward_weights directement
    env.reward_weights = {"pnl": 0.0, "sharpe": 1.0, "drawdown": 0.0, "consistency": 0.0, "cot_coherence": 0.0}
    # reward_std_zero = env._calculate_reward(next_price=100, executed_action=0) # Cet appel n'est pas nécessaire pour l'assertion finale
    # Le seul composant non nul devrait être 0 car sharpe est 0
    # (ou une petite valeur due aux arrondis si risk_free_rate n'est pas exactement la moyenne)
    # Pour être sûr, on vérifie que la composante sharpe est nulle.
    # La fonction _calculate_reward ne retourne pas les composantes, seulement la somme.
    # On peut mocker np.std pour qu'il retourne 0 et vérifier la récompense totale.
    with patch("numpy.std", return_value=0):
        reward_std_zero_mocked = env._calculate_reward(next_price=100, executed_action=0)
        assert reward_std_zero_mocked == 0.0  # Si tous les autres poids sont à 0

    # Cas 2: drawdown_weight affecte la récompense
    env.returns = [0.01, 0.01, 0.01] * 7  # Retours stables
    env.reward_weights = {  # Réinitialiser les poids pour ce cas
        "pnl": 0.0,
        "sharpe": 0.0,
        "drawdown": -0.3,  # Valeur de test pour drawdown
        "consistency": 0.0,
        "cot_coherence": 0.0,
    }

    # Simuler un drawdown via nav_history
    env.initial_balance = 10000.0
    env.nav_history = [10000.0, 11000.0, 9000.0]  # Peak = 11000, Current NAV = 9000
    # Drawdown = (11000-9000)/11000 = 2000/11000 approx 0.1818
    # drawdown_penalty = -0.1818 * 100 = -18.18

    # Garder executed_action=0 pour que pnl_reward soit 0
    current_close_price = env.price_data["close"].iloc[env.current_step]

    # Test avec poids de drawdown négatif (comme par défaut)
    # env.reward_weights['drawdown'] est déjà -0.3
    reward_with_drawdown_penalty = env._calculate_reward(next_price=current_close_price, executed_action=0)
    # Expected: (-0.3) * (-18.18) = 5.454

    # Test avec poids de drawdown nul
    env.reward_weights["drawdown"] = 0.0
    reward_without_drawdown_penalty = env._calculate_reward(next_price=current_close_price, executed_action=0)
    # Expected: 0.0 * (-18.18) = 0.0

    # Si la pénalité de drawdown est négative et le poids est négatif, la contribution est positive.
    # Si le poids est 0, la contribution est 0.
    # Donc, reward_with_drawdown_penalty (avec poids négatif) devrait être > reward_without_drawdown_penalty (poids nul)
    # si drawdown_penalty lui-même est négatif (ce qui est le cas).
    actual_drawdown_penalty_val = -((max(env.nav_history) - env.nav_history[-1]) / max(env.nav_history)) * 100

    if actual_drawdown_penalty_val < 0:  # S'il y a une pénalité de drawdown
        assert reward_with_drawdown_penalty > reward_without_drawdown_penalty
    else:  # Pas de drawdown ou drawdown nul
        assert reward_with_drawdown_penalty == reward_without_drawdown_penalty

    # Cas 3: consistency_weight affecte la récompense
    env.returns = [0.01, 0.01, 0.01, 0.01, 0.01]  # Retours très constants
    env.reward_weights = {  # Réinitialiser les poids pour ce cas
        "pnl": 0.0,
        "sharpe": 0.0,
        "drawdown": 0.0,
        "consistency": 0.2,  # Valeur de test pour consistance
        "cot_coherence": 0.0,
    }
    # current_close_price est déjà défini depuis le cas 2

    # consistency_reward est calculé sur les 5 derniers self.returns
    # Si std_recent_returns est petit, consistency_reward est grand.

    reward_with_consistency = env._calculate_reward(next_price=current_close_price, executed_action=0)

    env.reward_weights["consistency"] = 0.0  # Mettre le poids de consistance à 0
    reward_without_consistency = env._calculate_reward(next_price=current_close_price, executed_action=0)

    # consistency_reward est positif ou nul.
    if np.std(env.returns[-5:]) < 1e-7:  # Si la std est très faible (forte consistance)
        assert reward_with_consistency > reward_without_consistency
    else:  # Si la std est plus élevée, la récompense de consistance peut être faible ou nulle
        assert reward_with_consistency >= reward_without_consistency

    # La partie ci-dessous concernant env.consistency_reward_weight et env.num_consistent_days
    # a été supprimée car elle ne teste pas _calculate_reward comme prévu.
    # _calculate_reward utilise self.reward_weights et self.returns.


# --- Tests pour LiveExecutor (ajoutés ici pour l'instant) ---
import unittest
import asyncio
import json
from live.executor import LiveExecutor  # Assurez-vous que l'import est correct


class TestLiveExecutor(unittest.IsolatedAsyncioTestCase):

    @patch("live.executor.load_config")
    @patch("live.executor.MorningstarModel.load_model")
    @patch("live.executor.LiveDataPreprocessor")
    @patch("live.executor.MetricsLogger")
    @patch("live.executor.ccxt.binance")  # Mocking a specific exchange client
    @patch("live.executor.ccxtpro.binance")  # Mocking a specific ccxt.pro client
    async def test_update_live_status_file_includes_pnl(
        self,
        mock_ccxtpro_client,  # ccxtpro.binance
        mock_ccxt_client,  # ccxt.binance
        mock_metrics_logger,
        mock_live_data_preprocessor,
        mock_morningstar_model_load,
        mock_load_config,
    ):
        # Configuration minimale pour instancier LiveExecutor
        mock_config_data = {
            "live_trading": {
                "default_exchange": "binance",
                "symbol": "BTC/USDT",
                "timeframe": "1m",
                "websocket": False,  # Simplifier en désactivant le websocket pour ce test
                "status_file_path": "test_live_status.json",
                "healthcheck": {},
            },
            "exchange_params": {
                "binance": {
                    "apiKey": "test_key",  # Requis par _init_exchange_clients
                    "secret": "test_secret",  # Requis par _init_exchange_clients
                }
            },
            "model": {"save_path": "dummy_model_path"},
            "data_pipeline": {"indicator_window_size": 10},
        }
        mock_load_config.return_value = mock_config_data

        # Mocker les instances de client retournées
        mock_ccxt_instance = MagicMock()
        mock_ccxt_client.return_value = mock_ccxt_instance
        # Pas besoin de mocker ccxtpro si websocket est False

        # Mocker les variables d'environnement pour les clés API
        with patch.dict(os.environ, {"BINANCE_API_KEY": "test_key", "BINANCE_API_SECRET": "test_secret"}):
            executor = LiveExecutor(config_path="dummy_config.yaml", dry_run=True)

        executor.current_pnl = 123.45
        executor.symbol = "BTC/USDT"  # Assurer que le symbole est défini
        executor.position_side = "long"
        executor.entry_price = 30000.0
        executor.current_position_size = 0.1
        executor.last_known_balance = {"USDT": {"free": 1000, "total": 1000}}
        executor.active_sl_order_id = "sl123"
        executor.active_tp_order_id = "tp123"
        executor.trading_active = True
        executor.consecutive_errors = 0

        status_file = Path("test_live_status.json")
        if status_file.exists():
            status_file.unlink()

        executor._update_live_status_file()

        self.assertTrue(status_file.exists())
        with open(status_file, "r") as f:
            status_data = json.load(f)

        self.assertIn("current_pnl", status_data)
        self.assertEqual(status_data["current_pnl"], 123.45)
        self.assertEqual(status_data["symbol"], "BTC/USDT")

        # Nettoyage
        if status_file.exists():
            status_file.unlink()


if __name__ == "__main__":
    # Pour exécuter les tests de cette classe spécifiquement si besoin
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    pass
