# Workflow de Données et Dataset

## Vue d'ensemble

Le workflow de données est un processus complet qui transforme les données brutes du marché en datasets structurés prêts pour l'entraînement et l'inférence. Ce document détaille chaque étape du workflow et les outils utilisés.

## Étapes du Workflow

### 1. Acquisition des Données

#### Sources de Données
- **Binance API** : Données OHLCV (Open, High, Low, Close, Volume)
- **Gemini API** : Contexte de marché et informations textuelles
- **Autres sources** : Données macroéconomiques, sentiment social

#### Processus d'Acquisition
```bash
# Téléchargement de données OHLCV via le script principal
python scripts/generate_eth_dataset.py \
  --start-date 2023-01-01 \
  --end-date 2024-01-01 \
  --interval 1h \
  --output-path data/eth_dataset.parquet
```

### 2. Prétraitement et Nettoyage

- Gestion des valeurs manquantes
- Détection et traitement des anomalies
- Normalisation des données temporelles
- Validation de la qualité des données

### 3. Feature Engineering

#### Indicateurs Techniques (38 au total)
- **Indicateurs de base** (5) : OHLCV
- **Indicateurs classiques** (14) : SMA, EMA, RSI, MACD, etc.
- **Indicateurs additionnels** (19) : ADX, CCI, VWAP, etc.
- **Indicateurs finaux** (5) : STOCHRSI, CMO, PPO, etc.

#### Validation des Features
- Vérification du nombre exact d'indicateurs (38)
- Test de variance non-nulle pour chaque feature
- Gestion des NaN et valeurs infinies
- Normalisation des distributions

### 4. Détection de Régimes de Marché

- **Modèle** : Hidden Markov Model (HMM) à 4 états
- **Entrée** : Données de prix et volume
- **Sortie** : 
  - État du régime (1-4)
  - Probabilités de chaque régime

### 5. Génération d'Embeddings

#### Modèle CryptoBERT
- **Source** : ElKulako/cryptobert
- **Caractéristiques** : 
  - Modèle BERT spécialisé pour le domaine crypto
  - Dimension d'embedding : 768
  - Adapté au contexte financier et crypto

#### Processus d'Embedding
1. Récupération du contexte de marché via Gemini API
2. Tokenisation du texte avec le tokenizer CryptoBERT
3. Génération des embeddings via le modèle CryptoBERT
4. Intégration des embeddings au dataset

### 6. Métriques de Trading

- **Volatilité** : Calculée sur fenêtre glissante de 24h
- **ATR** : Average True Range pour mesurer la volatilité
- **Stop-Loss** : Niveaux dynamiques basés sur l'ATR
- **Take-Profit** : Niveaux dynamiques basés sur l'ATR
- **Signaux** : Générés à partir des régimes et de la volatilité

### 7. Sauvegarde et Versioning

- Format de sauvegarde : Parquet (compression et performance)
- Métadonnées : Date de génération, paramètres, statistiques
- Versioning : Nomenclature claire pour les différentes versions

## Outils et Composants

### Scripts Principaux
- `generate_eth_dataset.py` : Script principal de génération de dataset
- `data_preparation.py` : Fonctions de prétraitement et nettoyage

### Classes Clés
- `APIManager` : Gestion des connexions API et récupération de données
- `CryptoBERTEmbedder` : Génération d'embeddings avec CryptoBERT
- `MarketRegimeDetector` : Détection des régimes de marché avec HMM

## Bonnes Pratiques

1. **Reproductibilité**
   - Utilisation de seeds pour les composants aléatoires
   - Documentation des paramètres et configurations
   - Versioning des datasets générés

2. **Performance**
   - Optimisation des calculs d'indicateurs techniques
   - Mise en cache des embeddings LLM
   - Traitement par lots pour les opérations coûteuses

3. **Maintenance**
   - Tests unitaires pour chaque composant
   - Validation automatique des datasets générés
   - Monitoring des performances et de la qualité

## Exemple d'Utilisation

```python
# Import des composants nécessaires
from utils.data_preparation import CryptoBERTEmbedder
from utils.api_manager import APIManager
from scripts.generate_eth_dataset import calculate_trading_metrics

# Initialisation des composants
api = APIManager({'exchange': 'binance'})
embedder = CryptoBERTEmbedder(model_name="ElKulako/cryptobert")

# Récupération des données OHLCV
df = api.fetch_ohlcv_data('binance', 'ETH/USDT', '1h', '2023-01-01', '2023-01-31')

# Calcul des métriques de trading
df = calculate_trading_metrics(df)

# Génération d'embeddings pour un texte
context = "Ethereum price surges as market sentiment improves"
embedding = embedder.embed_text(context)
```

## Évolutions Futures

1. **Intégration de nouvelles sources de données**
   - Données on-chain (transactions, activité du réseau)
   - Sentiment social (Twitter, Reddit)
   - Données macroéconomiques

2. **Amélioration des embeddings**
   - Fine-tuning de CryptoBERT sur des données spécifiques
   - Exploration de modèles multimodaux

3. **Optimisation du pipeline**
   - Parallélisation des calculs
   - Streaming de données en temps réel
   - Réduction de la dimensionnalité pour les embeddings
