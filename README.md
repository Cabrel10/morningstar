# Morningstar: Système de Trading Crypto avec Framework DECoT-RL-GA

Morningstar est un système avancé de trading de crypto-monnaies qui implémente le framework DECoT-RL-GA (Deep Extraction with Chain-of-Thought, Reinforcement Learning, and Genetic Algorithm). Ce framework combine l'extraction de caractéristiques profondes, le raisonnement explicite, l'apprentissage par renforcement et l'optimisation génétique pour créer un système de trading robuste, explicable et adaptable.

## Framework DECoT-RL-GA

Le framework est composé de quatre modules principaux :

1. **CNN+LSTM pour l'extraction de caractéristiques** : Extrait des caractéristiques pertinentes à partir des données de marché en utilisant un réseau de neurones hybride.

2. **Chain-of-Thought (CoT) pour le raisonnement explicite** : Génère des explications claires et logiques pour les décisions de trading, transformant le système d'une "boîte noire" en un système transparent.

3. **Reinforcement Learning (RL) pour l'entraînement de l'agent** : Utilise l'algorithme PPO pour apprendre à prendre des décisions de trading optimales à partir de l'expérience.

4. **Genetic Algorithm (GA) pour l'optimisation des hyperparamètres** : Optimise automatiquement les hyperparamètres de tous les modules pour maximiser les performances de trading.

## Caractéristiques

- **Capacité de raisonnement explicite** : Explique les décisions de trading avec une chaîne de raisonnement logique
- **Optimisation automatique** : Ajuste automatiquement les hyperparamètres pour s'adapter à différents marchés
- **Rentabilité avec capital limité** : Conçu pour être rentable même avec un capital de moins de 20$
- **Analyse multi-facteurs** : Combine données de marché, indicateurs techniques et caractéristiques extraites
- **Gestion des risques intégrée** : Optimise le ratio risque/récompense pour chaque trade
- **Architecture hybride** : Combine CNN, LSTM, RL et GA pour des performances optimales

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Cabrel10/morningstar.git
cd morningstar

# Créer et activer l'environnement conda
conda create -n trading_env python=3.11
conda activate trading_env

# Installer les dépendances
pip install -r trading_env_deps.txt
```

## Configuration

1. Créez un fichier `.env` à la racine du projet avec vos clés API:

```
GEMINI_API_KEY=votre_clé_api_gemini
EXCHANGE_API_KEY=votre_clé_api_exchange
EXCHANGE_API_SECRET=votre_secret_api_exchange
```

2. Pour les tests locaux, vous pouvez activer l'exchange mock:

```
USE_MOCK_EXCHANGE=true
```

## Workflow du framework DECoT-RL-GA

### 1. Collecte et préparation des données

```bash
# Collecter les données de marché
python scripts/collect_market_data.py --output-dir data/raw --start-date 2023-01-01 --end-date 2023-12-31 --symbols BTC/USDT ETH/USDT

# Préparer les données pour l'entraînement
python scripts/prepare_dataset.py --input data/raw/market_data.csv --output data/processed/prepared_data.csv
```

### 2. Optimisation des hyperparamètres avec GA (Module 4)

```bash
# Exécuter l'optimisation génétique
python scripts/optimize_hyperparams.py --data-path data/processed/prepared_data.csv --output-dir output/optimization --population-size 50 --generations 20
```

Cette étape utilise l'algorithme génétique pour trouver les meilleurs hyperparamètres pour tous les modules du framework.

### 3. Entraînement de l'agent RL avec les meilleurs hyperparamètres (Module 3)

```bash
# Entraîner l'agent avec les hyperparamètres optimisés
python scripts/train_agent.py --data-path data/processed/prepared_data.csv --hyperparams-path output/optimization/best_hyperparams.json --output-dir output/trained_agent
```

Cette étape entraîne l'agent RL en utilisant les hyperparamètres optimisés par le Module 4.

### 4. Génération d'explications avec CoT (Module 2)

```bash
# Générer des explications pour les décisions de trading
python scripts/generate_explanations.py --agent-path output/trained_agent/agent.zip --data-path data/test/test_data.csv --output-dir output/explanations
```

Cette étape utilise le mécanisme Chain-of-Thought pour générer des explications pour les décisions de l'agent.

### 5. Backtesting et évaluation

```bash
# Évaluer les performances de l'agent
python scripts/backtest.py --agent-path output/trained_agent/agent.zip --data-path data/test/test_data.csv --initial-balance 20.0 --output-dir output/backtest
```

### 6. Trading en temps réel

```bash
# Démarrer le trading en temps réel
python scripts/live_trading.py --agent-path output/trained_agent/agent.zip --symbols BTC/USDT --initial-balance 20.0
```

## Structure du projet

```
Morningstar/
├── data/                      # Données brutes et traitées
│   ├── raw/                   # Données brutes des exchanges
│   ├── real/                  # Données réelles collectées
│   └── processed/             # Données préparées pour l'entraînement
│       └── normalized/        # Données normalisées
├── docs/                      # Documentation du projet
│   └── modules/               # Documentation des modules DECoT-RL-GA
│       ├── framework_overview.md  # Vue d'ensemble du framework
│       ├── module1.md         # Module 1: CNN+LSTM
│       ├── module2.md         # Module 2: Chain-of-Thought
│       ├── module3.md         # Module 3: Reinforcement Learning
│       └── module4.md         # Module 4: Genetic Algorithm
├── model/                     # Modèles et algorithmes
│   ├── architecture/          # Architecture des modèles
│   ├── reasoning/             # Mécanismes de raisonnement
│   └── training/              # Modules d'entraînement
│       ├── data_loader.py     # Chargement des données
│       ├── reinforcement_learning.py  # Agent RL et environnement
│       └── genetic_optimizer.py  # Optimisation génétique
├── data_collectors/           # Modules de collecte de données
├── data_processors/           # Modules de traitement de données
│   ├── hmm_regime_detector.py # Détection de régime avec HMM
│   └── cryptobert_processor.py # Traitement de texte avec CryptoBERT
├── model/                     # Modèles d'apprentissage
│   ├── architecture/          # Architectures de modèles
│   │   ├── simplified_model.py # Modèle simplifié
│   │   └── reasoning_model.py # Modèle avec capacité de raisonnement
│   ├── reasoning/             # Modules de raisonnement
│   │   └── reasoning_module.py # Module de Chain-of-Thought
│   └── training/              # Scripts d'entraînement
│       ├── genetic_optimizer.py # Optimisation génétique
│       └── reasoning_training.py # Entraînement avec raisonnement
├── notebooks/                 # Notebooks Jupyter/Colab
├── scripts/                   # Scripts utilitaires
│   ├── collect_real_data.py   # Collecte de données réelles
│   ├── prepare_improved_dataset.py # Préparation des données
│   └── generate_enhanced_dataset.py # Génération de dataset amélioré
├── improved_evaluate.py       # Évaluation améliorée
├── improve_model.py           # Orchestration des améliorations
├── trading_bot.py             # Bot de trading
├── requirements.txt           # Dépendances
└── README.md                  # Documentation
```

## Utilisation du modèle avec capacité de raisonnement

Le modèle avec capacité de raisonnement permet d'obtenir non seulement des prédictions, mais aussi des explications détaillées sur les décisions de trading:

```python
import tensorflow as tf
from model.reasoning.reasoning_module import ExplanationDecoder

# Charger le modèle
model = tf.keras.models.load_model('model/reasoning_model/best_model.h5')

# Préparer les données d'entrée
inputs = {
    'technical_input': technical_features,
    'llm_input': llm_embeddings,
    'mcp_input': mcp_features,
    'hmm_input': hmm_features,
    'instrument_input': instrument_ids
}

# Faire des prédictions
predictions = model.predict(inputs)

# Créer un décodeur d'explications
explanation_decoder = ExplanationDecoder(
    feature_names=feature_names,
    market_regime_names=['sideways', 'bullish', 'bearish', 'volatile']
)

# Décoder les explications
market_regime_pred = np.argmax(predictions['market_regime'][0])
market_regime_explanation = explanation_decoder.decode_market_regime_explanation(
    market_regime_pred,
    predictions['market_regime_explanation'][0],
    predictions['attention_scores'][0],
    top_k=3
)

sl_explanation, tp_explanation = explanation_decoder.decode_sl_tp_explanation(
    predictions['sl_tp'][0][0],  # SL
    predictions['sl_tp'][0][1],  # TP
    predictions['sl_explanation'][0],
    predictions['tp_explanation'][0],
    predictions['attention_scores'][0],
    top_k=3
)

print(f"Régime de marché: {explanation_decoder.market_regime_names[market_regime_pred]}")
print(f"Explication: {market_regime_explanation}")
print(f"Stop Loss: {predictions['sl_tp'][0][0]:.4f}")
print(f"Explication SL: {sl_explanation}")
print(f"Take Profit: {predictions['sl_tp'][0][1]:.4f}")
print(f"Explication TP: {tp_explanation}")
```

## Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT.
# morningstar
# morningstar
# morningstar
# morningstar
