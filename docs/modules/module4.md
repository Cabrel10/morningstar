# Module 4: Genetic Algorithm (GA) pour l'optimisation des hyperparamètres

## Vue d'ensemble

Le Module 4 du framework DECoT-RL-GA implémente un algorithme génétique (GA) pour optimiser automatiquement les hyperparamètres du modèle hybride, y compris les paramètres du CNN+LSTM et de l'agent RL. Ce module permet de trouver la configuration optimale qui maximise les performances de trading.

## Fonctionnalités

- **Optimisation des hyperparamètres** : Recherche automatique des meilleurs hyperparamètres pour tous les modules
- **Évaluation de fitness** : Fonction de fitness basée sur les performances de trading (profit, ratio de Sharpe, drawdown)
- **Opérations génétiques** : Sélection, croisement et mutation pour explorer l'espace des hyperparamètres
- **Parallélisation** : Évaluation parallèle des individus pour accélérer l'optimisation
- **Suivi des performances** : Enregistrement des meilleures configurations et de leur évolution

## Implémentation

Le module est implémenté dans les fichiers suivants :
- `model/training/genetic_optimizer.py` : Implémentation de l'algorithme génétique
- `test_module4.py` : Tests pour valider le fonctionnement du module GA

## Utilisation

```python
from model.training.genetic_optimizer import optimize_hyperparams, train_best_agent
import pandas as pd

# Charger les données
price_data = pd.read_csv('data/price_data.csv')
feature_data = pd.read_csv('data/feature_data.csv')

# Définir les limites des hyperparamètres
hyperparam_ranges = {
    'learning_rate': (0.0001, 0.001),
    'n_steps': (64, 2048),
    'batch_size': (32, 256),
    'n_epochs': (3, 10),
    'gamma': (0.9, 0.999),
    'gae_lambda': (0.9, 0.99),
    'clip_range': (0.1, 0.3),
    'window_size': (10, 50),
    'reward_scaling': (0.1, 10.0),
    'transaction_fee': (0.0001, 0.003),
    'cnn_filters': (32, 128),
    'cnn_kernel_size': (2, 5),
    'lstm_units': (64, 256),
    'dropout_rate': (0.1, 0.5)
}

# Optimiser les hyperparamètres
best_hyperparams, best_fitness = optimize_hyperparams(
    price_data=price_data,
    feature_data=feature_data,
    hyperparam_ranges=hyperparam_ranges,
    population_size=50,
    generations=20,
    output_dir='output/optimization'
)

# Entraîner l'agent avec les meilleurs hyperparamètres
best_agent = train_best_agent(
    best_hyperparams,
    price_data,
    feature_data,
    output_dir='output/best_agent'
)

# Évaluer l'agent final
metrics = best_agent.evaluate(num_episodes=10)
print(f"Profit moyen: {metrics['mean_reward']}")
```

## Paramètres de l'algorithme génétique

- `population_size` : Taille de la population (nombre d'individus)
- `generations` : Nombre de générations
- `mutation_rate` : Taux de mutation
- `crossover_rate` : Taux de croisement
- `tournament_size` : Taille du tournoi pour la sélection

## Hyperparamètres optimisés

Le Module 4 optimise les hyperparamètres suivants pour tous les modules du framework :

### Module 1 (CNN+LSTM)
- `cnn_filters` : Nombre de filtres dans les couches convolutives
- `cnn_kernel_size` : Taille du noyau des filtres convolutifs
- `lstm_units` : Nombre d'unités dans les couches LSTM
- `dropout_rate` : Taux de dropout pour la régularisation

### Module 3 (RL)
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

Le Module 4 optimise les hyperparamètres de tous les autres modules du framework DECoT-RL-GA pour maximiser les performances globales du système de trading. Il permet d'adapter automatiquement le système à différents marchés et conditions de trading.
