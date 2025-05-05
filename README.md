# Morningstar: Système de Trading Crypto avec Capacité de Raisonnement

Morningstar est un système avancé de trading de crypto-monnaies qui combine l'analyse technique, l'analyse de sentiment et l'apprentissage automatique avec une capacité de raisonnement pour expliquer ses décisions de trading.

## Caractéristiques

- **Capacité de raisonnement**: Explique les décisions de trading au lieu de faire des trades systématiques sans justification
- **Analyse multi-sources**: Combine données de marché, sentiment des actualités et analyse technique
- **Détection de régime de marché**: Identifie automatiquement les régimes de marché (haussier, baissier, latéral, volatile)
- **Prédiction de SL/TP optimaux**: Recommande des niveaux de Stop Loss et Take Profit adaptés au contexte
- **Intégration de Gemini API**: Analyse avancée du sentiment des actualités crypto
- **CryptoBERT**: Utilise un modèle spécialisé pour l'analyse des textes liés aux crypto-monnaies
- **Architecture hybride**: Combine réseaux de neurones, HMM et LLM pour des prédictions robustes

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Cabrel10/eva001.git
cd eva001

# Installer les dépendances
pip install -e .
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

## Workflow complet

### 1. Collecte de données réelles

```bash
python scripts/collect_real_data.py --output-dir data/real --start-date 2023-01-01 --end-date 2023-12-31 --symbols BTC/USDT ETH/USDT
```

Options disponibles:
- `--gemini-api-keys`: Liste des clés API Gemini pour l'analyse de sentiment
- `--timeframe`: Intervalle de temps (1d, 1h, etc.)
- `--no-cryptobert`: Désactiver l'utilisation de CryptoBERT
- `--no-hmm`: Désactiver la détection de régime HMM
- `--no-sentiment`: Désactiver l'analyse de sentiment

### 2. Préparation des données

```bash
python scripts/prepare_improved_dataset.py --input data/real/final_dataset.parquet --output data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv
```

### 3. Entraînement du modèle avec capacité de raisonnement

#### Option 1: Entraînement local

```bash
python model/training/reasoning_training.py --data-path data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv --output-dir model/reasoning_model
```

#### Option 2: Entraînement sur Google Colab

1. Ouvrez le notebook `notebooks/morningstar_reasoning_model_training.ipynb` dans Google Colab
2. Suivez les instructions dans le notebook pour télécharger les données et entraîner le modèle

### 4. Évaluation du modèle

```bash
python improved_evaluate.py --model-path model/reasoning_model/best_model.h5 --data-path data/processed/normalized/multi_crypto_dataset_prepared_normalized.csv
```

### 5. Trading en temps réel

```bash
python trading_bot.py --model-path model/reasoning_model/best_model.h5 --symbols BTC/USDT ETH/USDT
```

## Structure du projet

```
Morningstar/
├── data/                      # Données brutes et traitées
│   ├── raw/                   # Données brutes des exchanges
│   ├── real/                  # Données réelles collectées
│   └── processed/             # Données préparées pour l'entraînement
│       └── normalized/        # Données normalisées
├── data_collectors/           # Modules de collecte de données
│   ├── market_data_collector.py  # Collecte des données de marché
│   ├── news_collector.py      # Collecte des actualités crypto
│   └── sentiment_analyzer.py  # Analyse de sentiment avec Gemini
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
