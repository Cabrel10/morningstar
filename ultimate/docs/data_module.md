# Module de Données et Dataset

## Vue d'ensemble

Le module de données est responsable de la génération et de la gestion des datasets pour l'entraînement et l'inférence. Il comprend plusieurs composants clés :

### 1. Générateur de Dataset ETH (`generate_eth_dataset.py`)

Ce script génère un dataset complet pour ETH avec :

- **Données OHLCV** : Téléchargées depuis Binance
- **38 Indicateurs Techniques** :
  - 5 indicateurs de base (OHLCV)
  - 14 indicateurs classiques (SMA, EMA, RSI, MACD, etc.)
  - 19 indicateurs additionnels (ADX, CCI, VWAP, etc.)
  - 5 indicateurs finaux (STOCHRSI, CMO, PPO, etc.)
- **Détection de Régimes** : HMM à 4 états
- **Embeddings LLM** : Via CryptoBERT (ElKulako/cryptobert) et contexte Gemini
- **Features MCP** : Métriques de complexité du marché
- **Métriques de Trading** : Volatilité, stop-loss, take-profit

### 2. Gestionnaire API (`api_manager.py`)

Interface unifiée pour :
- Connexion aux exchanges (ccxt)
- Téléchargement de données OHLCV
- Gestion des rate limits et erreurs
- Formatage des données

### 3. Préparation des Données (`data_preparation.py`)

Module pour :
- Nettoyage des données
- Gestion des valeurs manquantes
- Normalisation
- Validation des features

## Workflow de Génération de Dataset

1. **Configuration**
   ```bash
   python generate_eth_dataset.py \
     --start-date YYYY-MM-DD \
     --end-date YYYY-MM-DD \
     --interval 1h \
     --output-path path/to/output.parquet
   ```

2. **Pipeline de Traitement**
   - Téléchargement OHLCV
   - Calcul des indicateurs techniques
   - Détection des régimes de marché
   - Génération des embeddings LLM
   - Calcul des features MCP
   - Ajout des métriques de trading
   - Validation et sauvegarde

3. **Validation**
   - Vérification du nombre d'indicateurs (38)
   - Test de variance non-nulle
   - Gestion des NaN
   - Validation des noms de colonnes

## Bonnes Pratiques

1. **Gestion des Erreurs**
   - Retry automatique pour les erreurs réseau
   - Logging détaillé
   - Validation des données à chaque étape

2. **Performance**
   - Optimisation des calculs d'indicateurs
   - Gestion efficace de la mémoire
   - Parallélisation quand possible

3. **Maintenance**
   - Tests unitaires pour chaque composant
   - Documentation des fonctions et paramètres
   - Versioning des datasets

## Utilisation dans le Workflow de Trading

1. **Entraînement**
   ```python
   from scripts.generate_eth_dataset import ETHDatasetGenerator
   
   generator = ETHDatasetGenerator()
   train_data = generator.generate(
       start_date="2023-01-01",
       end_date="2023-12-31"
   )
   ```

2. **Inférence Live**
   ```python
   from utils.api_manager import APIManager
   
   api = APIManager(config)
   market_data = api.get_market_data()
   ```

## Maintenance et Mise à Jour

1. **Mise à Jour des Features**
   - Ajouter de nouveaux indicateurs dans `feature_engineering.py`
   - Mettre à jour la validation dans `validate_features()`
   - Régénérer les datasets d'entraînement

2. **Mise à Jour des Sources de Données**
   - Ajouter de nouveaux exchanges dans `api_manager.py`
   - Mettre à jour les paramètres de rate limiting
   - Tester la compatibilité des formats

3. **Optimisation**
   - Profiling régulier des performances
   - Optimisation des calculs gourmands
   - Mise à jour des dépendances
