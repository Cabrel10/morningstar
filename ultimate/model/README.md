# Modèle Monolithique Morningstar

## Vue d'ensemble

Le modèle monolithique Morningstar est une architecture unifiée qui combine toutes les fonctionnalités précédemment dispersées dans différents modèles en une seule architecture cohérente. Cette approche simplifie le développement, l'entraînement et le déploiement tout en permettant une flexibilité maximale.

## Architecture

Le modèle monolithique ingère:
- **Données techniques**: OHLCV et indicateurs techniques
- **Embeddings LLM**: Représentations vectorielles de texte/nouvelles
- **Features MCP**: Market Context Processor pour la détection de régimes
- **Identifiants d'instruments**: Pour la spécialisation par instrument
- **Chain-of-Thought** (optionnel): Pour le raisonnement explicite

L'architecture se compose de:
1. **Backbone partagé**: Dense → LSTM → Transformer
2. **Trois têtes spécialisées**:
   - Signal: Classification pour {Sell, Neutral, Buy}
   - SL: Régression pour niveau de stop-loss
   - TP: Régression pour niveau de take-profit

## Structure des fichiers

```
ultimate/model/
├── architecture/
│   ├── __init__.py
│   └── monolith_model.py     # Définition du modèle monolithique
├── training/
│   ├── __init__.py
│   └── train_monolith.py     # Script d'entraînement
├── utils/
│   ├── __init__.py
│   └── data_loader.py        # Utilitaires de chargement de données
├── __init__.py
└── run_backtest.py           # Script de backtesting
```

## Installation et configuration

1. Activez l'environnement conda:
   ```bash
   conda activate trading_env
   ```

2. Exécutez le script d'initialisation:
   ```bash
   python ultimate/init_trading_env.py
   ```

## Utilisation

### Importation et initialisation

```python
from ultimate.model.architecture.monolith_model import MonolithModel

# Configuration par défaut
model = MonolithModel()

# Configuration personnalisée
config = {
    "tech_input_shape": (42,),
    "backbone_config": {
        "lstm_units": 128,
        "transformer_blocks": 3
    },
    "active_outputs": ["signal", "sl", "tp"]
}
model = MonolithModel(config=config)
```

### Entraînement

```bash
python -m ultimate.model.training.train_monolith \
    --data path/to/data.parquet \
    --output-dir ./model_output \
    --epochs 100 \
    --batch-size 32
```

### Backtest

```bash
python -m ultimate.model.run_backtest \
    --model path/to/model.keras \
    --data path/to/backtest_data.parquet \
    --output-dir ./backtest_results
```

## Avantages du modèle monolithique

1. **Simplicité**: Une seule architecture à comprendre et maintenir
2. **Partage de features**: Les représentations apprises sont partagées entre les différentes tâches
3. **Entraînement efficient**: Un seul modèle à entraîner au lieu de plusieurs
4. **Flexibilité**: Facile à configurer pour différents cas d'usage
5. **Maintenance**: Simplifie la mise à jour et l'amélioration du modèle

## Personnalisation

Le modèle monolithique est hautement configurable via le dictionnaire de configuration. Vous pouvez ajuster:

- Dimensions d'entrée pour chaque type de données
- Architecture du backbone (taille, nombre de couches, etc.)
- Configurations des têtes de sortie
- Activation/désactivation de composants spécifiques

## Intégration avec l'existant

Le modèle monolithique est conçu pour être un remplaçant direct des architectures précédentes tout en étant rétrocompatible. Les interfaces pour l'entraînement, la prédiction et le backtest sont maintenues pour assurer une transition en douceur. 