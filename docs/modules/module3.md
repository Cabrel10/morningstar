# Module 3: Reinforcement Learning (RL) pour l'entraînement de l'agent

## Vue d'ensemble

Le Module 3 du framework DECoT-RL-GA implémente un agent d'apprentissage par renforcement (RL) qui apprend à prendre des décisions de trading optimales à partir de l'expérience. Ce module utilise l'algorithme Proximal Policy Optimization (PPO) pour entraîner un agent capable de maximiser les profits tout en gérant les risques.

## Fonctionnalités

- **Environnement de trading** : Simulation d'un environnement de trading avec des actions d'achat, vente et maintien
- **Fonction de récompense personnalisée** : Récompense basée sur le profit, le ratio de Sharpe et le drawdown
- **Politique d'action** : Politique stochastique qui détermine les actions à prendre
- **Gestion du capital** : Gestion automatique du capital et des positions
- **Évaluation des performances** : Métriques de performance pour évaluer la stratégie de trading

## Implémentation

Le module est implémenté dans les fichiers suivants :
- `model/training/reinforcement_learning.py` : Implémentation de l'agent RL et de l'environnement de trading
- `test_module3.py` : Tests pour valider le fonctionnement du module RL

## Utilisation

```python
from model.training.reinforcement_learning import TradingEnvironment, TradingRLAgent
import pandas as pd

# Charger les données
price_data = pd.read_csv('data/price_data.csv')
feature_data = pd.read_csv('data/feature_data.csv')

# Créer l'environnement de trading
env = TradingEnvironment(
    price_data=price_data,
    feature_data=feature_data,
    window_size=30,
    initial_balance=20.0,  # Capital initial de 20$
    transaction_fee=0.001
)

# Créer l'agent RL
agent = TradingRLAgent(
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="logs/tensorboard"
)

# Initialiser l'agent avec l'environnement
agent.create_agent(env)

# Entraîner l'agent
agent.train(total_timesteps=100000)

# Évaluer l'agent
metrics = agent.evaluate(num_episodes=10)
print(f"Profit moyen: {metrics['mean_reward']}")
```

## Paramètres optimisables

Les hyperparamètres suivants peuvent être optimisés par le Module 4 (GA) :
- `learning_rate` : Taux d'apprentissage pour l'optimiseur
- `n_steps` : Nombre d'étapes par mise à jour
- `batch_size` : Taille du batch pour l'entraînement
- `n_epochs` : Nombre d'époques d'entraînement par mise à jour
- `gamma` : Facteur de réduction pour les récompenses futures
- `gae_lambda` : Paramètre lambda pour l'estimation d'avantage généralisée
- `clip_range` : Paramètre de clipping pour PPO
- `window_size` : Taille de la fenêtre d'observation
- `reward_scaling` : Facteur d'échelle pour les récompenses
- `transaction_fee` : Frais de transaction

## Intégration avec les autres modules

Le Module 3 utilise les caractéristiques extraites par le Module 1 (CNN+LSTM) comme entrée pour l'environnement de trading et peut incorporer les explications générées par le Module 2 (CoT) pour améliorer la transparence des décisions. Les hyperparamètres de l'agent RL sont optimisés par le Module 4 (GA).
