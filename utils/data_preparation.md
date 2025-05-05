---

## 📘 `utils/data_preparation.md` – Spécification complète

### 🔧 Objectif Général

#### 🚀 Téléchargement des Données et Création de Dataset
Ajout d'une section expliquant comment utiliser les fonctions pour télécharger les données brutes et générer des datasets enrichis. Exemple :
```python
from utils.api_manager import fetch_ohlcv_data, save_data
df = fetch_ohlcv_data('binance', 'BTC/USDT', '1h', '2024-01-01', '2024-12-31')
save_data(df, 'data/raw/btc_usdt_1h.csv')
```

Ce module est responsable de transformer les données brutes multi-sources en jeux de données standardisés, enrichis, labelisés et utilisables directement par les modèles. Il agit en interface entre la couche de collecte (`data/raw/`) et les modules de modélisation (`model/`).

---

## 🧩 Composition du module

### 1. `load_raw_data(file_path: str)`

**But :** Charger un fichier de données brutes (CSV ou JSON) depuis le chemin spécifié. Gère la lecture, la validation des colonnes minimales et la normalisation du timestamp en UTC.

- **Entrées :**
  - `file_path` (str) : Chemin complet vers le fichier brut (ex: `'data/raw/BTC_USDT_1h.csv'`).
- **Sortie :**
  - `Optional[pandas.DataFrame]` : DataFrame contenant les données brutes avec un index `Timestamp` UTC, ou `None` si une erreur survient (fichier non trouvé, format invalide, colonnes manquantes, erreur de conversion de timestamp).
- **Validation :**
  - Vérifie l'existence du fichier.
  - Supporte les formats `.csv` et `.json` (line-delimited).
  - Vérifie la présence des colonnes minimales requises : `timestamp`, `open`, `high`, `low`, `close`, `volume`.
  - Convertit la colonne `timestamp` en `datetime` UTC et la définit comme index du DataFrame. Gère les erreurs de conversion et les fuseaux horaires potentiels.
- **Logs / Prints :**
  - Indique le succès ou l'échec du chargement.
  - Précise les colonnes manquantes si applicable.
  - Signale les erreurs de conversion de timestamp.

---

### 2. `clean_data(df: pd.DataFrame)`

**But :** Nettoyer le DataFrame brut en gérant les valeurs manquantes (NaN) et les outliers potentiels.

- **Entrées :**
  - `df` (pd.DataFrame) : DataFrame brut issu de `load_raw_data`, avec un index `Timestamp` UTC.
- **Sortie :**
  - `pd.DataFrame` : DataFrame nettoyé, prêt pour l'étape de feature engineering.
- **Comportement :**
  - **Gestion des NaN :**
    - Applique `ffill()` (forward fill) sur les colonnes OHLCV pour propager la dernière valeur valide.
    - Si des NaN persistent au début, applique `bfill()` (backward fill) ou remplace par 0 pour le volume.
    - Supprime les lignes où `close` reste NaN après imputation.
  - **Gestion des outliers (Exemple simple basé sur IQR des rendements) :**
    - Calcule les rendements (`pct_change`).
    - Identifie les valeurs en dehors de l'intervalle interquartile (IQR * 1.5).
    - Logue le nombre d'outliers détectés (l'action de correction/suppression est commentée dans le code actuel et peut être adaptée).
- **Logs / Prints :**
  - Affiche les dimensions avant et après nettoyage.
  - Indique le nombre de NaNs remplis par colonne (ffill, bfill, 0).
  - Signale le nombre de lignes supprimées à cause de NaNs persistants.
  - Indique le nombre d'outliers détectés.

---

### 3. `save_processed_data(df: pd.DataFrame, output_path: str, format: str = 'parquet')`

**But :** Sauvegarder le DataFrame traité dans un fichier au format spécifié (Parquet par défaut).

- **Entrées :**
  - `df` (pd.DataFrame) : Le DataFrame (généralement nettoyé ou enrichi) à sauvegarder.
  - `output_path` (str) : Chemin complet du fichier de sortie (ex: `'data/processed/BTC_USDT_1h_clean.parquet'`).
  - `format` (str, optionnel) : Format de sortie, 'parquet' ou 'csv'. Défaut : 'parquet'.
- **Comportement :**
  - Crée le répertoire de sortie s'il n'existe pas.
  - Sauvegarde le DataFrame en utilisant `to_parquet()` ou `to_csv()`.
- **Logs / Prints :**
  - Confirme la création du répertoire si nécessaire.
  - Indique le succès ou l'échec de la sauvegarde et le chemin du fichier.
  - Signale si le format demandé n'est pas supporté.

---

### 4. `apply_technical_indicators(df)`

**But :** Calculer les indicateurs techniques de base. (Fonctionnalité à implémenter par Développeur B)

- **Entrée :** Données propres
- **Sortie :** `DataFrame` avec colonnes supplémentaires :
  - RSI, MACD, SMA/EMA (5, 20, 50), Bollinger, ATR, ADX, StochRSI, Ichimoku
- **Librairies :**
  - `ta-lib` (ou `pandas-ta` si ta-lib n’est pas dispo)
- **Paramètres configurables :**
  - Fenêtres glissantes (ex. `rsi_window=14`, `sma_window=20`)

---

### 5. `integrate_sentiment_data(market_df, sentiment_df)`

**But :** Fusionner les signaux de sentiment (Reddit, Twitter, news). (Fonctionnalité à implémenter par Développeur B/C)

- **Entrées :**
  - Données de marché (`market_df`)
  - Données de sentiment (`sentiment_df`)
- **Sortie :**
  - `DataFrame` enrichi : `sentiment_score`, `buzz_index`, `toxicity_score`
- **Méthodes de fusion :**
  - Join temporel arrondi à la minute/heures/jour
  - Moyenne glissante pondérée par fréquence de mention

---

### 6. `generate_labels(df, horizon='15min', type='classification')`

**But :** Générer les cibles d’apprentissage supervisé. (Fonctionnalité à implémenter par Développeur C)

- **Entrée :**
  - Données enrichies
  - Horizon (15min, 1h, 1d)
  - Type : `classification`, `regression`, `market_regime`
- **Sortie :** DataFrame avec colonnes de labels :
  - `price_change_label`, `volatility_cluster`, `tp_sl_targets`
- **Méthodes proposées :**
  - Classification : changement de prix seuil (e.g., ±1%)
  - Régimes : KMeans ou HMM sur volatilité/volume
  - TP/SL : +2%/-1% à horizon T → cible multi-tête

---

### 7. `build_dataset(symbols, freq='15min')`

**But :** Fonction orchestratrice globale. (Fonctionnalité à implémenter après les étapes individuelles)

- **Étapes exécutées :**
  1. Charger et nettoyer
  2. Appliquer indicateurs techniques
  3. Fusionner sentiment/news
  4. Générer labels
- **Retourne :**
  - `features_df`, `labels_df`, `full_df`
- **Logging structuré :**
  - Nb. de points par symbole
  - Répartition des classes
  - Temps de traitement par étape

---

## 🔄 Comportement général attendu

- Compatible multi-actifs (BTC, ETH, altcoins).
- Multi-échelles temporelles.
- Conçu pour une mise à jour incrémentale quotidienne.
- Modularité maximale : chaque étape testable seule.

---

## 📌 Dépendances (Module Complet)

```txt
# Pour les fonctions implémentées (Dev A):
pandas
numpy
python-dotenv # Bien que non utilisé directement dans les fonctions, listé pour le projet

# Pour les étapes futures (Dev B & C):
ta-lib (ou pandas-ta)
scikit-learn
# nltk / textblob / transformers (si scoring NLP local)
# Potentiellement d'autres selon les sources de données (ex: ccxt, newsapi-python)
```

---

## 📡 Extensions prévues

- Intégration d’on-chain analytics (avec Coinmetrics / Glassnode).
- Ajout d’un LLM scoring assistant pour taguer les événements financiers.
- Pipeline d’échantillonnage intelligent : sélection de fenêtres de marché pertinentes (drawdown, rallye, panique).
