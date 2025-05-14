#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'apprentissage par renforcement (Deep RL) pour le projet Morningstar.
Ce module implémente l'algorithme PPO (Proximal Policy Optimization) pour optimiser
les stratégies de trading à partir des signaux générés par le modèle hybride.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import random
import json
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config.config import Config  # Ajout de l'import

# Ajouter le répertoire du projet au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root devrait être le dossier 'Morningstar'
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)  # Insertion en priorité

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_REWARD_WEIGHTS = {
    "pnl": 1.0,  # Profit and Loss
    "sharpe": 0.5,  # Sharpe Ratio
    "drawdown": -0.3,  # Maximum Drawdown (pénalité)
    "consistency": 0.2,  # Constance des gains
    "cot_coherence": 0.2,  # Cohérence des explications Chain-of-Thought
}


class TradingEnvironment(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement, conforme à l'API Gym.
    L'agent interagit avec cet environnement pour apprendre des stratégies de trading optimales.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        feature_data: pd.DataFrame,
        window_size: int = 60,
        max_steps: Optional[int] = None,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        reward_weights: Optional[Dict[str, float]] = None,
        use_cot: bool = True,
        cot_explanations: Optional[List[str]] = None,
        random_start: bool = True,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialise l'environnement de trading.

        Args:
            price_data: DataFrame des données de prix (OHLCV)
            feature_data: DataFrame des caractéristiques techniques, LLM, MCP, HMM, etc.
            window_size: Taille de la fenêtre d'observation
            max_steps: Nombre maximum d'étapes par épisode (None = toute la série)
            initial_balance: Capital initial
            transaction_fee: Frais de transaction (pourcentage)
            reward_weights: Poids des différentes composantes de la récompense
            use_cot: Utiliser les explications Chain-of-Thought
            cot_explanations: Liste des explications Chain-of-Thought précalculées
            random_start: Commencer à un point aléatoire de la série temporelle
            risk_free_rate: Taux sans risque pour le calcul du ratio de Sharpe
        """
        super(TradingEnvironment, self).__init__()

        # Validation des données
        assert len(price_data) == len(
            feature_data
        ), "Les données de prix et de caractéristiques doivent avoir la même longueur"
        assert window_size > 0, "La taille de la fenêtre doit être positive"

        # Filtrer les données où le prix de clôture est non positif
        valid_indices = price_data["close"] > 0
        if not valid_indices.all():
            original_len = len(price_data)
            price_data = price_data[valid_indices].copy()
            feature_data = feature_data[valid_indices].copy()
            logger.warning(f"Filtrage des données: {original_len - len(price_data)} lignes supprimées car close <= 0.")
            # Réindexer pour éviter les trous potentiels si nécessaire (optionnel)
            # price_data.reset_index(drop=True, inplace=True)
            # feature_data.reset_index(drop=True, inplace=True)

        # Stocker les paramètres
        self.price_data = price_data
        self.feature_data = feature_data
        self.window_size = window_size
        # Recalculer max_steps après filtrage
        self.max_steps = max_steps if max_steps is not None else len(self.price_data) - self.window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_weights = reward_weights or DEFAULT_REWARD_WEIGHTS
        self.use_cot = use_cot
        self.cot_explanations = cot_explanations
        self.random_start = random_start
        self.risk_free_rate = risk_free_rate

        # Variables d'état
        self.current_step = None
        self.balance = None
        self.position = None
        self.position_price = None
        self.trades = None
        self.nav_history = None
        self.returns = None

        # Définir l'espace d'action: [0: ne rien faire, 1: acheter, 2: vendre]
        self.action_space = spaces.Discrete(3)

        # Définir l'espace d'observation
        obs_shape = self._get_observation_shape()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Historique des récompenses et des actions pour le calcul des métriques
        self.reward_history = []
        self.action_history = []
        self.explanation_history = []
        self.reset()

    def _get_observation_shape(self) -> Tuple[int]:
        """
        Détermine la forme de l'espace d'observation.

        Returns:
            Tuple des dimensions de l'espace d'observation
        """
        if self.window_size > 0:
            # Nombre de features dans le DataFrame des caractéristiques
            num_features = self.feature_data.shape[1]

            # Forme: [fenêtre temporelle, nombre de features]
            # + [balance, position, prix d'entrée]
            return (self.window_size, num_features + 3)
        else:
            # Cas spécial: window_size = 0 (uniquement l'état actuel)
            return (self.feature_data.shape[1] + 3,)

    def reset(self, *, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed: Graine pour la génération de nombres aléatoires (pour compatibilité gymnasium)
            options: Options supplémentaires (pour compatibilité gymnasium)

        Returns:
            Observation initiale et dictionnaire d'informations (pour compatibilité gymnasium)
        """
        # Initialiser le générateur de nombres aléatoires si une graine est fournie
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Définir le point de départ (aléatoire ou fixe)
        if self.random_start:
            self.current_step = random.randint(0, len(self.price_data) - self.window_size - self.max_steps)
        else:
            self.current_step = 0

        # Réinitialiser l'état
        self.balance = self.initial_balance
        self.position = 0  # 0: pas de position, 1: long
        self.position_price = 0.0
        self.trades = []
        self.nav_history = [self.initial_balance]
        self.returns = []

        # Réinitialiser l'historique
        self.reward_history = []
        self.action_history = []
        self.explanation_history = []

        # Pour compatibilité avec gymnasium >= 0.26.0
        return self._get_observation(), {}

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action: Action à exécuter (0: ne rien faire, 1: acheter, 2: vendre)

        Returns:
            observation: Nouvelle observation
            reward: Récompense
            terminated: Si l'épisode est terminé naturellement
            truncated: Si l'épisode est tronqué (limite de temps, etc.)
            info: Informations supplémentaires
        """
        # Vérifier si l'épisode est terminé
        terminated = self.current_step >= len(self.price_data) - 1
        truncated = self.current_step >= self.window_size + self.max_steps - 1
        done = terminated or truncated

        if done:
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        # Récupérer le prix actuel et suivant
        current_price = self.price_data.iloc[self.current_step]["close"]
        next_price = self.price_data.iloc[self.current_step + 1]["close"]

        # Exécuter l'action
        executed_action = self._execute_action(action, current_price)

        # Mettre à jour le solde en fonction du mouvement de prix
        self._update_portfolio(next_price)

        # Calculer la récompense
        reward = self._calculate_reward(next_price, executed_action)
        self.reward_history.append(reward)
        self.action_history.append(action)

        # Ajouter une explication CoT si disponible
        if self.use_cot and self.cot_explanations is not None:
            if self.current_step < len(self.cot_explanations):
                self.explanation_history.append(self.cot_explanations[self.current_step])
            else:
                self.explanation_history.append("")

        # Passer à l'étape suivante
        self.current_step += 1

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_action(self, action, current_price):
        """
        Exécute une action et met à jour l'état du portefeuille.

        Args:
            action: Action à exécuter (0: ne rien faire, 1: acheter, 2: vendre)
            current_price: Prix actuel

        Returns:
            Action réellement exécutée
        """
        executed_action = 0  # Par défaut: ne rien faire

        # Acheter si l'action est 1 et que nous n'avons pas de position
        if action == 1 and self.position == 0:
            # Vérifier si le prix actuel est valide
            if current_price <= 0:
                logger.warning(
                    f"Tentative d'achat à prix non positif ({current_price}) à l'étape {self.current_step}. Action ignorée."
                )
                executed_action = 0  # Ne rien faire si le prix est invalide
            else:
                cost = current_price * (1 + self.transaction_fee)
                # Vérifier si le capital est suffisant
                if self.balance >= cost:
                    self.position = 1
                    self.position_price = current_price  # Prix est > 0 ici
                self.balance -= cost
                executed_action = 1

                # Enregistrer le trade
                self.trades.append(
                    {
                        "step": self.current_step,
                        "type": "buy",
                        "price": current_price,
                        "fee": current_price * self.transaction_fee,
                        "balance": self.balance,
                    }
                )

        # Vendre si l'action est 2 et que nous avons une position
        elif action == 2 and self.position == 1:
            revenue = current_price * (1 - self.transaction_fee)
            self.balance += revenue
            self.position = 0
            executed_action = 2

            # Calculer le P&L (vérifier que position_price n'est pas zéro)
            pnl = 0.0
            if self.position_price != 0:
                pnl = ((revenue / self.position_price) - 1) * 100  # En pourcentage
            else:
                logger.warning(f"Tentative de calcul de PNL avec position_price=0 à l'étape {self.current_step}")

            # Enregistrer le trade
            self.trades.append(
                {
                    "step": self.current_step,
                    "type": "sell",
                    "price": current_price,
                    "fee": current_price * self.transaction_fee,
                    "balance": self.balance,
                    "pnl": pnl,
                }
            )

            self.position_price = 0

        return executed_action

    def _update_portfolio(self, next_price):
        """
        Met à jour la valeur du portefeuille.

        Args:
            next_price: Prix à la prochaine étape
        """
        # Calculer la valeur nette actuelle (NAV)
        if self.position == 1:
            nav = self.balance + next_price
        else:
            nav = self.balance

        # Mettre à jour l'historique
        self.nav_history.append(nav)

        # Calculer le rendement
        if len(self.nav_history) > 1:
            prev_nav = self.nav_history[-2]
            if prev_nav > 0:
                ret = (nav / prev_nav) - 1
                self.returns.append(ret)
            else:
                self.returns.append(0)

    def _calculate_reward(self, next_price, executed_action):
        """
        Calcule la récompense pour l'étape actuelle.

        Args:
            next_price: Prix à la prochaine étape
            executed_action: Action réellement exécutée

        Returns:
            Récompense totale
        """
        # 1. Récompense basée sur le P&L
        pnl_reward = 0.0
        if self.position == 1 and self.position_price != 0:  # Ajout vérification != 0
            # Rendement non réalisé
            price_change_pct = (next_price / self.position_price) - 1
            pnl_reward = price_change_pct * 100

        # 2. Récompense basée sur le ratio de Sharpe (si assez de données valides)
        sharpe_reward = 0.0
        if len(self.returns) >= 10:  # Assez de retours
            recent_returns = np.array(self.returns[-10:])
            finite_returns = recent_returns[np.isfinite(recent_returns)]  # Filtrer NaN/inf
            if len(finite_returns) >= 2:  # Besoin d'au moins 2 points pour std dev
                mean_return = np.mean(finite_returns)
                std_return = np.std(finite_returns)
            else:
                mean_return = 0.0
                std_return = 0.0  # Ou une autre valeur par défaut
                logger.warning(
                    f"NaN/inf détecté dans les retours récents à l'étape {self.current_step}. Sharpe non calculé."
                )

            # Vérifier si std_return est proche de zéro ou NaN
            if std_return > 1e-8 and not np.isnan(std_return):
                sharpe = (mean_return - self.risk_free_rate) / std_return
                sharpe_reward = np.clip(sharpe, -5, 5)  # Clipper pour éviter valeurs extrêmes
            else:
                sharpe_reward = 0.0  # Pas de récompense si std est trop faible ou NaN

        # 3. Pénalité pour le drawdown
        drawdown_penalty = 0
        if len(self.nav_history) > 1:
            peak = max(self.nav_history)
            drawdown = (peak - self.nav_history[-1]) / peak
            drawdown_penalty = -drawdown * 100  # Négatif pour pénaliser

        # 4. Récompense pour la constance des gains
        consistency_reward = 0.0
        if len(self.returns) >= 5:  # Assez de retours
            recent_returns_consistency = np.array(self.returns[-5:])
            finite_returns_consistency = recent_returns_consistency[
                np.isfinite(recent_returns_consistency)
            ]  # Filtrer NaN/inf
            if len(finite_returns_consistency) >= 2:  # Besoin d'au moins 2 points pour std dev
                std_recent_returns = np.std(finite_returns_consistency)
            else:
                std_recent_returns = np.inf  # Pour que la récompense soit 0
                logger.warning(
                    f"NaN/inf détecté dans les retours récents (consistance) à l'étape {self.current_step}. Constance non calculée."
                )

            # Moins de variance est meilleur.
            if (
                np.isnan(std_recent_returns) or std_recent_returns > 100
            ):  # Si std est NaN ou très grand, récompense nulle
                consistency_reward = 0.0
            elif std_recent_returns < 1e-8:  # Si std est très faible (forte consistance)
                consistency_reward = 10.0  # Récompense maximale (valeur du clip)
            else:  # std est raisonnable et non nul
                consistency_reward = 1 / std_recent_returns
                # Normaliser/clipper pour éviter des valeurs extrêmes
                consistency_reward = np.clip(consistency_reward, 0, 10)

        # 5. Récompense pour la cohérence des explications CoT (simplifié pour éviter dépendance externe ici)
        cot_reward = 0.0
        # if self.use_cot and len(self.explanation_history) >= 2:
        #     # Note: get_embedding et cosine_similarity peuvent être coûteux et nécessitent SentenceTransformer
        #     # Pour la robustesse, on pourrait utiliser une métrique plus simple ou désactiver temporairement
        #     pass # Placeholder - à implémenter si nécessaire et stable

        # S'assurer que toutes les composantes sont des nombres finis
        pnl_reward = np.nan_to_num(pnl_reward, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe_reward = np.nan_to_num(sharpe_reward, nan=0.0, posinf=0.0, neginf=0.0)
        drawdown_penalty = np.nan_to_num(drawdown_penalty, nan=0.0, posinf=0.0, neginf=0.0)
        consistency_reward = np.nan_to_num(consistency_reward, nan=0.0, posinf=0.0, neginf=0.0)
        cot_reward = np.nan_to_num(cot_reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Combiner toutes les récompenses selon les poids
        total_reward = (
            self.reward_weights.get("pnl", 1.0) * pnl_reward
            + self.reward_weights.get("sharpe", 0.5) * sharpe_reward
            + self.reward_weights.get("drawdown", -0.3) * drawdown_penalty
            + self.reward_weights.get("consistency", 0.2) * consistency_reward
            + self.reward_weights.get("cot_coherence", 0.2) * cot_reward
        )

        # S'assurer que la récompense finale est un nombre fini
        total_reward = np.nan_to_num(
            total_reward, nan=0.0, posinf=1e6, neginf=-1e6
        )  # Remplacer inf par une grande valeur

        return float(total_reward)  # Retourner un float standard

    def _get_observation(self):
        """
        Construit l'observation actuelle pour l'agent.

        Returns:
            Observation (tableau numpy)
        """
        # Récupérer les données des features pour la fenêtre d'observation
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1

        # Caractéristiques techniques
        # S'assurer que les données sont numériques (pas de timestamps ou d'objets)
        features = self.feature_data.iloc[start_idx:end_idx].select_dtypes(include=["number"]).values

        # Informations d'état du portefeuille
        portfolio_state = np.array(
            [
                self.balance / self.initial_balance,  # Balance normalisée
                self.position,  # Position actuelle
                self.position_price,  # Prix d'entrée (0 si pas de position)
            ]
        )

        # Si window_size > 0, créer une séquence
        if self.window_size > 0:
            logger.debug(f"Shape of features before padding/stacking: {features.shape}")
            logger.debug(f"Shape of portfolio_state: {portfolio_state.shape}")
            # Padding si nécessaire
            if len(features) < self.window_size:
                padding = np.zeros((self.window_size - len(features), features.shape[1]))
                features = np.vstack([padding, features])

            # Ajouter les infos du portefeuille à chaque pas de temps
            portfolio_info = np.tile(portfolio_state, (self.window_size, 1))
            observation = np.hstack([features, portfolio_info])
        else:
            # Sinon, juste l'état actuel
            observation = np.hstack([features[-1], portfolio_state])

        # Vérifier et convertir les valeurs non numériques en zéros
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation.astype(np.float32)

    def _get_info(self):
        """
        Fournit des informations supplémentaires sur l'état actuel.

        Returns:
            Dictionnaire d'informations
        """
        # Calculer les métriques
        total_pnl = 0
        win_rate = 0

        if len(self.trades) > 0:
            # Calculer le P&L total
            buy_trades = [t for t in self.trades if t["type"] == "buy"]
            sell_trades = [t for t in self.trades if t["type"] == "sell"]

            if len(sell_trades) > 0:
                # P&L total
                total_pnl = sum([t.get("pnl", 0) for t in sell_trades])

                # Win rate
                winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
                win_rate = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0

        # Calculer la valeur nette actuelle (NAV)
        if len(self.nav_history) > 0:
            nav = self.nav_history[-1]
            nav_change = (nav / self.initial_balance - 1) * 100
        else:
            nav = self.initial_balance
            nav_change = 0

        # Calculer le drawdown
        max_nav = max(self.nav_history) if len(self.nav_history) > 0 else self.initial_balance
        drawdown = (max_nav - nav) / max_nav * 100

        # Calculer le ratio de Sharpe
        sharpe = 0
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) + 1e-5
            sharpe = (mean_return - self.risk_free_rate) / std_return

        return {
            "balance": self.balance,
            "position": self.position,
            "nav": nav,
            "nav_change_pct": nav_change,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "drawdown_pct": drawdown,
            "sharpe_ratio": sharpe,
            "num_trades": len(self.trades),
        }

    def render(self):
        """
        Affiche l'état actuel de l'environnement.

        Returns:
            Représentation visuelle (pour compatibilité gymnasium)
        """
        # Afficher les informations courantes
        info = self._get_info()
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"NAV: ${info['nav']:.2f} ({info['nav_change_pct']:.2f}%)")
        print(f"Total P&L: {info['total_pnl']:.2f}%")
        print(f"Win Rate: {info['win_rate']:.2f}")
        print(f"Drawdown: {info['drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
        print(f"Number of Trades: {info['num_trades']}")
        print("=" * 50)

        # Pour compatibilité avec gymnasium, retourner un tableau numpy
        # représentant une image vide (placeholder)
        return np.zeros((300, 600, 3), dtype=np.uint8)


class CustomPPOPolicy(ActorCriticPolicy):
    """
    Politique personnalisée pour PPO qui intègre le modèle CNN+LSTM existant.
    Cette classe permet d'utiliser les caractéristiques extraites par le modèle hybride
    comme entrée pour la politique d'apprentissage par renforcement.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        hybrid_model_path: Optional[str] = None,
        use_cnn_lstm: bool = True,
        use_cot: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialise la politique personnalisée.

        Args:
            observation_space: Espace d'observation Gym
            action_space: Espace d'action Gym
            lr_schedule: Fonction qui renvoie le taux d'apprentissage en fonction du progrès
            hybrid_model_path: Chemin vers le modèle hybride pré-entraîné
            use_cnn_lstm: Utiliser les couches CNN+LSTM pour l'extraction de caractéristiques
            use_cot: Utiliser le module Chain-of-Thought pour l'explication des décisions
        """
        super(CustomPPOPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.hybrid_model_path = hybrid_model_path
        self.use_cnn_lstm = use_cnn_lstm
        self.use_cot = use_cot
        self.hybrid_model = None

        # Charger le modèle hybride si spécifié
        if hybrid_model_path is not None:
            self._load_hybrid_model()

    def _load_hybrid_model(self):
        """
        Charge le modèle hybride pré-entraîné.
        """
        try:
            # Importer les modules nécessaires ici pour éviter les dépendances circulaires
            from model.architecture.morningstar_model import MorningstarModel

            logger.info(f"Chargement du modèle hybride depuis {self.hybrid_model_path}")
            self.hybrid_model = MorningstarModel.load_model(self.hybrid_model_path)
            logger.info("Modèle hybride chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle hybride: {e}")
            self.hybrid_model = None

    def forward(self, obs, deterministic=False):
        """
        Forward pass dans le réseau de politique.

        Args:
            obs: Observations de l'environnement
            deterministic: Si True, renvoie l'action déterministe

        Returns:
            Tuple (actions, valeurs, log_probs)
        """
        # Si un modèle hybride est disponible, utiliser ses caractéristiques extraites
        if self.hybrid_model is not None:
            # TODO: Adapter obs pour le modèle hybride et extraire les caractéristiques
            # Cette partie dépend de l'interface exacte du modèle hybride
            pass

        # Utiliser la méthode forward de la classe parent
        return super().forward(obs, deterministic)


class RewardCallback(BaseCallback):
    """
    Callback pour suivre et visualiser les récompenses pendant l'entraînement.
    """

    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        """
        Appelé à chaque étape de l'entraînement.

        Returns:
            True pour continuer l'entraînement, False pour l'arrêter
        """
        # Récupérer la récompense de l'étape actuelle
        reward = self.locals.get("rewards")[0]
        self.rewards.append(reward)
        self.current_episode_reward += reward

        # Si l'épisode est terminé, enregistrer la récompense totale
        done = self.locals.get("dones")[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

            # Afficher la récompense moyenne des derniers épisodes
            if len(self.episode_rewards) % 10 == 0 and self.verbose > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {len(self.episode_rewards)}, Récompense moyenne: {avg_reward:.2f}")

        return True


class TradingRLAgent:
    """
    Agent d'apprentissage par renforcement pour le trading utilisant PPO.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hybrid_model_path: Optional[str] = None,
        use_cnn_lstm: bool = True,
        use_cot: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Optional[float] = None,  # Rendu optionnel car lu depuis ppo_params si fourni
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialise l'agent d'apprentissage par renforcement.

        Args:
            model_path: Chemin où sauvegarder/charger le modèle RL
            hybrid_model_path: Chemin vers le modèle hybride pré-entraîné
            use_cnn_lstm: Utiliser les couches CNN+LSTM pour l'extraction de caractéristiques
            use_cot: Utiliser le module Chain-of-Thought pour l'explication des décisions
            policy_kwargs: Arguments pour la politique
            learning_rate: Taux d'apprentissage
            n_steps: Nombre d'étapes par mise à jour
            batch_size: Taille du batch
            n_epochs: Nombre d'époques
            gamma: Facteur de réduction
            gae_lambda: Paramètre lambda pour GAE
            clip_range: Paramètre de clip pour PPO
            verbose: Niveau de verbosité
            tensorboard_log: Répertoire pour les logs Tensorboard
        """
        # Ne plus charger Config ici, les paramètres PPO viendront des arguments du constructeur

        self.model_path = model_path
        self.hybrid_model_path = hybrid_model_path
        self.use_cnn_lstm = use_cnn_lstm
        self.use_cot = use_cot

        # Définir les paramètres de la politique
        self.policy_kwargs = policy_kwargs or {}
        if hybrid_model_path is not None:
            self.policy_kwargs.update(
                {"hybrid_model_path": hybrid_model_path, "use_cnn_lstm": use_cnn_lstm, "use_cot": use_cot}
            )

        # Paramètres de PPO - Utiliser directement les arguments passés au constructeur
        # Les valeurs par défaut de la signature seront utilisées si non fournies dans ppo_params
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range if clip_range is not None else 0.2  # Assurer une valeur par défaut si None
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        # L'agent PPO sera initialisé lors de l'appel à create_agent
        self.agent = None
        self.env = None

    def create_agent(self, env):
        """
        Crée l'agent PPO avec l'environnement spécifié.

        Args:
            env: Environnement de trading

        Returns:
            Agent PPO créé
        """
        # Enregistrer l'environnement
        self.env = env

        # Créer l'agent PPO avec la politique personnalisée
        self.agent = PPO(
            policy=CustomPPOPolicy,
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            policy_kwargs=self.policy_kwargs,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
        )

        # Charger le modèle s'il existe
        if self.model_path is not None and os.path.exists(self.model_path):
            try:
                logger.info(f"Chargement du modèle RL depuis {self.model_path}")
                self.agent = PPO.load(self.model_path, env=env)
                logger.info("Modèle RL chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle RL: {e}")
                # Continuer avec le modèle nouvellement créé

        # Charger le modèle PPO s'il existe (utiliser un chemin différent de hybrid_model)
        # if self.model_path is not None and os.path.exists(self.model_path + ".zip"): # SB3 ajoute .zip
        #     try:
        #         logger.info(f"Chargement du modèle RL PPO depuis {self.model_path}.zip")
        #         # Charger les paramètres et potentiellement la politique/optimiseur
        #         self.agent = PPO.load(self.model_path, env=env, policy=CustomPPOPolicy)
        #         logger.info("Modèle RL PPO chargé avec succès")
        #     except Exception as e:
        #         logger.error(f"Erreur lors du chargement du modèle RL PPO: {e}. Création d'un nouvel agent.")
        #         # Continuer avec le modèle nouvellement créé
        # else:
        #     logger.info("Aucun modèle RL PPO pré-entraîné trouvé ou spécifié. Création d'un nouvel agent.")

        # Note: Le chargement de PPO.load avec une politique personnalisée peut être complexe.
        # Pour l'instant, nous créons toujours un nouvel agent PPO et laissons CustomPPOPolicy
        # charger le modèle Keras hybride si hybrid_model_path est fourni.
        # self.model_path sera utilisé pour la sauvegarde à la fin de l'entraînement.

        return self.agent

    def train(self, total_timesteps, eval_freq=10000, n_eval_episodes=10, callback=None):
        """
        Entraîne l'agent sur l'environnement de trading.

        Args:
            total_timesteps: Nombre total d'étapes d'entraînement
            eval_freq: Fréquence d'évaluation (en étapes)
            n_eval_episodes: Nombre d'épisodes pour l'évaluation
            callback: Callback à appeler pendant l'entraînement

        Returns:
            Agent entraîné
        """
        if self.agent is None:
            raise ValueError("L'agent n'a pas été créé. Appelez create_agent d'abord.")

        # Créer un callback d'évaluation
        eval_env = self.env
        eval_callback = EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=os.path.dirname(self.model_path) if self.model_path else None,
            best_model_save_path=os.path.dirname(self.model_path) if self.model_path else None,
            deterministic=True,
        )

        # Créer un callback pour suivre les récompenses
        reward_callback = RewardCallback(verbose=self.verbose)

        # Combiner les callbacks
        callbacks = [eval_callback, reward_callback]
        if callback is not None:
            callbacks.append(callback)

        # Entraîner l'agent
        logger.info(f"Début de l'entraînement pour {total_timesteps} étapes")
        self.agent.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Sauvegarder le modèle final
        if self.model_path is not None:
            logger.info(f"Sauvegarde du modèle RL dans {self.model_path}")
            self.agent.save(self.model_path)

        return self.agent

    def evaluate(self, n_episodes=10, deterministic=True):
        """
        Évalue l'agent sur l'environnement de trading.

        Args:
            n_episodes: Nombre d'épisodes d'évaluation
            deterministic: Si True, utilise la politique déterministe

        Returns:
            Dictionnaire des métriques d'évaluation
        """
        if self.agent is None:
            raise ValueError("L'agent n'a pas été créé. Appelez create_agent d'abord.")

        # Réinitialiser l'environnement
        # Gérer les différents types d'environnements (vectorisés ou non)
        reset_result = self.env.reset()
        # Vérifier si le résultat est un tuple (observation, info) ou juste une observation
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result  # Environnement gymnasium standard
        else:
            obs = reset_result  # Environnement vectorisé

        # Variables pour suivre les performances
        all_rewards = []
        episode_rewards = []
        all_trades = []
        all_infos = []

        # Exécuter l'évaluation
        for episode in range(n_episodes):
            terminated = False
            truncated = False
            done = False
            episode_reward = 0
            episode_info = None

            while not done:
                # Prédire l'action
                action, _ = self.agent.predict(obs, deterministic=deterministic)

                # Exécuter l'action
                step_result = self.env.step(action)

                # Gérer les différents formats de retour (gymnasium vs vectorisé)
                if isinstance(step_result, tuple):
                    if len(step_result) == 5:  # Format gymnasium (obs, reward, terminated, truncated, info)
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:  # Format gym (obs, reward, done, info)
                        obs, reward, done, info = step_result
                    else:
                        raise ValueError(f"Format de retour de step() non reconnu: {len(step_result)} valeurs")
                else:
                    # Pour les environnements vectorisés, step_result est l'observation
                    obs = step_result
                    reward = 0
                    done = False
                    info = {}

                # Enregistrer les résultats
                episode_reward += reward
                episode_info = info

            # Enregistrer les résultats de l'épisode
            episode_rewards.append(episode_reward)
            all_infos.append(episode_info)

            # Accéder aux trades de l'environnement (si c'est TradingEnvironment)
            if hasattr(self.env, "trades"):
                all_trades.extend(self.env.trades)

            # Réinitialiser l'environnement pour le prochain épisode
            reset_result = self.env.reset()
            # Gérer les différents types d'environnements (vectorisés ou non)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result  # Environnement gymnasium standard
            else:
                obs = reset_result  # Environnement vectorisé

            # Convertir les valeurs numpy en float standard pour le formatage
            if episode_info and "nav" in episode_info:
                nav_value = float(episode_info["nav"]) if hasattr(episode_info["nav"], "item") else episode_info["nav"]
                logger.info(
                    f"Episode {episode+1}/{n_episodes}, Reward: {float(episode_reward):.2f}, NAV: {nav_value:.2f}"
                )
            else:
                logger.info(f"Episode {episode+1}/{n_episodes}, Reward: {float(episode_reward):.2f}")

        # Calculer les métriques d'évaluation de base
        if not episode_rewards:  # Si la liste est vide
            metrics = {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }
        else:
            metrics = {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "min_reward": np.min(episode_rewards),
                "max_reward": np.max(episode_rewards),
            }

        # Vérifier que all_infos contient des dictionnaires valides
        valid_infos = [info for info in all_infos if isinstance(info, dict)]

        # Ajouter des métriques supplémentaires si des informations valides sont disponibles
        if valid_infos:
            # Fonction sécurisée pour extraire des valeurs avec une valeur par défaut
            def safe_mean(key, default=0.0):
                values = [
                    info.get(key, default) for info in valid_infos if isinstance(info.get(key), (int, float, np.number))
                ]
                return np.mean(values) if values else default

            metrics.update(
                {
                    "mean_nav": safe_mean("nav"),
                    "mean_nav_change": safe_mean("nav_change_pct"),
                    "mean_drawdown": safe_mean("drawdown_pct"),
                    "mean_sharpe": safe_mean("sharpe_ratio"),
                    "mean_trades": safe_mean("num_trades"),
                }
            )

            # Calculer le taux de victoire moyen (uniquement pour les épisodes avec des transactions)
            trades_infos = [info for info in valid_infos if info.get("num_trades", 0) > 0]
            if trades_infos:
                metrics["mean_win_rate"] = safe_mean("win_rate")

            # Calculer le nombre total de transactions
            metrics["total_trades"] = sum(info.get("num_trades", 0) for info in valid_infos)

        # Afficher les métriques
        logger.info("=== Métriques d'évaluation ===")
        for key, value in metrics.items():
            logger.info(f"{key}: {value:.4f}")

        return metrics, all_trades, all_infos


def create_trading_env_from_data(price_data, feature_data, window_size=60, **kwargs):
    """
    Crée un environnement de trading à partir de DataFrames de prix et de caractéristiques.

    Args:
        price_data: DataFrame des données de prix (OHLCV)
        feature_data: DataFrame des caractéristiques techniques
        window_size: Taille de la fenêtre d'observation
        **kwargs: Arguments supplémentaires pour TradingEnvironment

    Returns:
        Environnement de trading vectorisé
    """
    # Créer l'environnement de trading
    env = TradingEnvironment(price_data=price_data, feature_data=feature_data, window_size=window_size, **kwargs)

    # Envelopper l'environnement dans un DummyVecEnv
    # car SB3 requiert un environnement vectorisé
    env = DummyVecEnv([lambda: env])

    # Normaliser l'environnement pour stabiliser l'apprentissage
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    return env


def load_and_prepare_data(data_path, price_cols=None, feature_cols=None, split_ratio=0.8):
    """
    Charge et prépare les données pour l'apprentissage par renforcement.

    Args:
        data_path: Chemin vers le fichier de données (CSV ou Parquet)
        price_cols: Liste des colonnes de prix à utiliser
        feature_cols: Liste des colonnes de caractéristiques à utiliser
        split_ratio: Ratio de division entre entraînement et test

    Returns:
        Tuples de DataFrames (train_prices, train_features, test_prices, test_features)
    """
    # Détecter le format du fichier
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Définir les colonnes par défaut
    if price_cols is None:
        price_cols = ["open", "high", "low", "close", "volume"]

    if feature_cols is None:
        # Exclure les colonnes de prix et colonnes non numériques
        feature_cols = [
            col for col in df.columns if col not in price_cols and col not in ["timestamp", "date", "symbol", "split"]
        ]

    # Vérifier que toutes les colonnes existent
    for col in price_cols + feature_cols:
        if col not in df.columns:
            logger.warning(f"Colonne {col} non trouvée dans les données. Colonnes disponibles: {df.columns.tolist()}")

    # Sélectionner uniquement les colonnes nécessaires
    df = df[price_cols + feature_cols]

    # Diviser en ensembles d'entraînement et de test
    train_size = int(len(df) * split_ratio)

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # Créer les DataFrames de prix et de caractéristiques
    train_prices = train_df[price_cols].copy()
    train_features = train_df[feature_cols].copy()

    test_prices = test_df[price_cols].copy()
    test_features = test_df[feature_cols].copy()

    # S'assurer que les features sont numériques
    train_features = train_features.select_dtypes(include=[np.number])
    test_features = test_features.select_dtypes(include=[np.number])

    # Remplacer les NaN potentiels
    train_features.fillna(0, inplace=True)
    test_features.fillna(0, inplace=True)

    logger.info(f"Données chargées: {len(df)} échantillons")
    logger.info(f"Entraînement: {len(train_df)} échantillons, Test: {len(test_df)} échantillons")
    logger.info(f"Nombre de caractéristiques: {len(feature_cols)}")

    return train_prices, train_features, test_prices, test_features


def train_rl_agent(
    data_path,
    model_path,
    hybrid_model_path=None,
    use_cnn_lstm=True,
    use_cot=True,
    window_size=60,
    initial_balance=10000.0,
    reward_weights=None,
    total_timesteps=100000,
    evaluation_episodes=10,
    ppo_params=None,
):
    """
    Entraîne un agent RL sur des données de trading.

    Args:
        data_path: Chemin vers le fichier de données
        model_path: Chemin où sauvegarder le modèle RL
        hybrid_model_path: Chemin vers le modèle hybride pré-entraîné
        use_cnn_lstm: Utiliser les couches CNN+LSTM
        use_cot: Utiliser le module Chain-of-Thought
        window_size: Taille de la fenêtre d'observation
        initial_balance: Capital initial
        reward_weights: Poids des différentes composantes de la récompense
        total_timesteps: Nombre total d'étapes d'entraînement
        evaluation_episodes: Nombre d'épisodes pour l'évaluation finale

    Returns:
        Agent entraîné et métriques d'évaluation
    """
    # Charger et préparer les données
    train_prices, train_features, test_prices, test_features = load_and_prepare_data(data_path)

    # Créer l'environnement d'entraînement
    train_env = create_trading_env_from_data(
        price_data=train_prices,
        feature_data=train_features,
        window_size=window_size,
        initial_balance=initial_balance,
        reward_weights=reward_weights,
        use_cot=use_cot,
    )

    # Créer l'agent
    agent = TradingRLAgent(
        model_path=model_path,
        hybrid_model_path=hybrid_model_path,
        use_cnn_lstm=use_cnn_lstm,
        use_cot=use_cot,
        tensorboard_log=os.path.join(os.path.dirname(model_path), "tb_logs"),
        **(ppo_params or {}),  # Passer les hyperparamètres PPO
    )

    # Initialiser l'agent avec l'environnement
    agent.create_agent(train_env)

    # Entraîner l'agent
    agent.train(total_timesteps=total_timesteps, eval_freq=total_timesteps // 10, n_eval_episodes=5)

    # Créer l'environnement de test
    test_env = create_trading_env_from_data(
        price_data=test_prices,
        feature_data=test_features,
        window_size=window_size,
        initial_balance=initial_balance,
        reward_weights=reward_weights,
        use_cot=use_cot,
    )

    # Ré-initialiser l'agent avec l'environnement de test
    agent.env = test_env

    # Évaluer l'agent
    metrics, trades, infos = agent.evaluate(n_episodes=evaluation_episodes)

    # Sauvegarder les résultats
    results_dir = os.path.dirname(model_path)
    os.makedirs(results_dir, exist_ok=True)

    # Convertir les métriques en types JSON sérialisables (float standard)
    serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}

    # Sauvegarder les métriques
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Sauvegarder les transactions
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(results_dir, "evaluation_trades.csv"), index=False)

    logger.info(f"Entraînement et évaluation terminés. Résultats sauvegardés dans {results_dir}")

    return agent, metrics, trades


def get_embedding(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text)


# def create_rl_agent(env, use_cot=False):
#     """
#     Crée un agent RL avec intégration du raisonnement CoT si activé
#     """
#     if use_cot:
#         cot_module = ChainOfThoughtReasoning() # Classe non définie
#         agent = TradingRLAgent(
#             env=env,
#             policy=CustomActorCriticPolicy, # Classe non définie
#             reasoning_module=cot_module,
#             verbose=1
#         )
#     else:
#         agent = TradingRLAgent(
#             env=env,
#             policy=CustomActorCriticPolicy, # Classe non définie
#             verbose=1
#         )
#     return agent

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraîner un agent RL pour le trading")
    parser.add_argument("--data", type=str, required=True, help="Chemin vers le fichier de données")
    parser.add_argument("--model", type=str, required=True, help="Chemin où sauvegarder le modèle RL")
    parser.add_argument("--hybrid-model", type=str, help="Chemin vers le modèle hybride pré-entraîné")
    parser.add_argument("--window-size", type=int, default=60, help="Taille de la fenêtre d'observation")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Capital initial")
    parser.add_argument("--timesteps", type=int, default=100000, help="Nombre total d'étapes d'entraînement")
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes pour l'évaluation finale")
    parser.add_argument("--no-cnn-lstm", action="store_true", help="Désactiver l'utilisation des couches CNN+LSTM")
    parser.add_argument("--no-cot", action="store_true", help="Désactiver l'utilisation du module Chain-of-Thought")
    # Ajouter les arguments pour les hyperparamètres PPO
    parser.add_argument("--n-steps", type=int, default=None, help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO batch_size")
    parser.add_argument("--n-epochs", type=int, default=None, help="PPO n_epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="PPO learning_rate")
    parser.add_argument("--gamma", type=float, default=None, help="PPO gamma")
    parser.add_argument("--gae-lambda", type=float, default=None, help="PPO gae_lambda")
    parser.add_argument("--clip-range", type=float, default=None, help="PPO clip_range")

    args = parser.parse_args()

    # Collecter les hyperparamètres PPO depuis les arguments
    ppo_params = {}
    ppo_arg_names = ["n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda", "clip_range"]
    for name in ppo_arg_names:
        value = getattr(args, name, None)
        if value is not None:
            ppo_params[name] = value

    # Entraîner l'agent
    train_rl_agent(
        data_path=args.data,
        model_path=args.model,
        hybrid_model_path=args.hybrid_model,
        use_cnn_lstm=not args.no_cnn_lstm,
        use_cot=not args.no_cot,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        total_timesteps=args.timesteps,
        evaluation_episodes=args.episodes,
        ppo_params=ppo_params,  # Passer les paramètres PPO collectés
    )
