# 📘 `utils/feature_engineering.md` – Documentation du Module de Feature Engineering

## 🎯 Objectif Général

Ce module est responsable du calcul des indicateurs techniques et de l'intégration des features contextuelles (y compris celles issues d'un LLM, via un placeholder pour le moment) aux données de marché nettoyées. Il prend en entrée un DataFrame `pandas` contenant au minimum les colonnes OHLCV et retourne un DataFrame enrichi avec les nouvelles features.

---

## ⚙️ Fonctions Principales

### 1. Fonctions de Calcul d'Indicateurs Techniques

Ces fonctions utilisent principalement la librairie `pandas-ta` pour calculer divers indicateurs. Elles prennent un DataFrame en entrée et retournent une `pandas.Series` ou un `pandas.DataFrame` contenant le ou les indicateurs calculés.

#### `compute_sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series`
- **But :** Calcule la Moyenne Mobile Simple (SMA).
- **Entrées :**
    - `df`: DataFrame contenant les données.
    - `period`: Période de calcul (fenêtre glissante).
    - `column`: Nom de la colonne sur laquelle calculer la SMA (par défaut 'close').
- **Sortie :** `pd.Series` avec les valeurs de la SMA.

#### `compute_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series`
- **But :** Calcule la Moyenne Mobile Exponentielle (EMA).
- **Entrées :**
    - `df`: DataFrame contenant les données.
    - `period`: Période de calcul.
    - `column`: Nom de la colonne ('close' par défaut).
- **Sortie :** `pd.Series` avec les valeurs de l'EMA.

#### `compute_rsi(df: pd.DataFrame, period: int = DEFAULT_RSI_PERIOD, column: str = 'close') -> pd.Series`
- **But :** Calcule l'Indice de Force Relative (RSI).
- **Entrées :**
    - `df`: DataFrame contenant les données.
    - `period`: Période de calcul (14 par défaut).
    - `column`: Nom de la colonne ('close' par défaut).
- **Sortie :** `pd.Series` avec les valeurs du RSI.

#### `compute_macd(df: pd.DataFrame, fast: int = ..., slow: int = ..., signal: int = ..., column: str = 'close') -> pd.DataFrame`
- **But :** Calcule la Convergence/Divergence de Moyenne Mobile (MACD).
- **Entrées :**
    - `df`: DataFrame contenant les données.
    - `fast`, `slow`, `signal`: Périodes pour les EMA rapides, lentes et la ligne de signal (valeurs par défaut définies dans le script).
    - `column`: Nom de la colonne ('close' par défaut).
- **Sortie :** `pd.DataFrame` avec les colonnes 'MACD', 'MACDs' (signal), 'MACDh' (histogramme).

#### `compute_bollinger_bands(df: pd.DataFrame, period: int = ..., std_dev: float = ..., column: str = 'close') -> pd.DataFrame`
- **But :** Calcule les Bandes de Bollinger.
- **Entrées :**
    - `df`: DataFrame contenant les données.
    - `period`: Période de calcul (20 par défaut).
    - `std_dev`: Nombre d'écarts-types (2.0 par défaut).
    - `column`: Nom de la colonne ('close' par défaut).
- **Sortie :** `pd.DataFrame` avec les colonnes 'BBU' (supérieure), 'BBM' (médiane), 'BBL' (inférieure).

#### `compute_atr(df: pd.DataFrame, period: int = ..., high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series`
- **But :** Calcule l'Average True Range (ATR).
- **Entrées :**
    - `df`: DataFrame contenant les données (doit avoir 'high', 'low', 'close').
    - `period`: Période de calcul (14 par défaut).
    - `high_col`, `low_col`, `close_col`: Noms des colonnes High, Low, Close.
- **Sortie :** `pd.Series` avec les valeurs de l'ATR.

#### `compute_stochastics(df: pd.DataFrame, k: int = ..., d: int = ..., smooth_k: int = ..., high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame`
- **But :** Calcule l'Oscillateur Stochastique.
- **Entrées :**
    - `df`: DataFrame contenant les données ('high', 'low', 'close').
    - `k`, `d`, `smooth_k`: Paramètres de l'oscillateur (valeurs par défaut définies).
    - `high_col`, `low_col`, `close_col`: Noms des colonnes.
- **Sortie :** `pd.DataFrame` avec les colonnes 'STOCHk' (%K) et 'STOCHd' (%D).

---

### 2. Intégration du Contexte LLM (Placeholder)

#### `integrate_llm_context(df: pd.DataFrame) -> pd.DataFrame`
- **But :** [Placeholder] Simuler l'intégration de données contextuelles générées par un LLM.
- **Entrées :** `df`: DataFrame à enrichir.
- **Sortie :** `df`: DataFrame enrichi avec des colonnes placeholders :
    - `llm_context_summary`: (str) "Placeholder LLM Summary"
    - `llm_embedding`: (str) "[0.1, -0.2, 0.3]" (représentation simplifiée)
- **Comportement Actuel :** Ajoute des colonnes avec des valeurs fixes et affiche un avertissement.
- **Comportement Futur :** Devra appeler une API LLM externe, gérer l'authentification, le coût, la latence et la synchronisation temporelle précise des données contextuelles avec les données de marché.

---

### 3. Pipeline Principal

#### `apply_feature_pipeline(df: pd.DataFrame, include_llm: bool = False) -> pd.DataFrame`
- **But :** Orchestrer l'application de toutes les étapes de feature engineering.
- **Entrées :**
    - `df`: DataFrame nettoyé (contenant au moins OHLCV).
    - `include_llm`: Booléen pour activer ou non l'intégration (placeholder) du contexte LLM.
- **Sortie :** `pd.DataFrame` enrichi avec tous les indicateurs calculés et potentiellement les données LLM (placeholder). Les lignes contenant des NaN introduites par les calculs de fenêtres glissantes sont supprimées.
- **Étapes :**
    1. Vérifie la présence des colonnes requises (OHLCV).
    2. Appelle les fonctions `compute_*` pour chaque indicateur technique et ajoute les résultats au DataFrame.
    3. Si `include_llm` est `True`, appelle `integrate_llm_context` (placeholder).
    4. Supprime les lignes avec des valeurs NaN.
    5. Affiche des logs sur la progression et le nombre de lignes supprimées.
- **TODO :** Ajouter une étape de vérification de conformité avec le schéma final attendu (38 colonnes).

---

## 🔧 Dépendances

- `pandas`
- `numpy`
- `pandas-ta` (alternative à `ta-lib`)
- `typing`

Pour l'intégration LLM future :
- Une librairie cliente pour l'API LLM choisie (ex: `openai`, `google-generativeai`, `transformers`).

---

## 🧪 Tests

Les fonctions de ce module devraient être testées unitairement dans `tests/test_utils.py` (ou un fichier dédié `tests/test_feature_engineering.py`) pour vérifier :
- Le type et la forme des sorties.
- L'exactitude des calculs sur des données connues (si possible).
- La gestion correcte des NaNs.
- Le bon fonctionnement du pipeline `apply_feature_pipeline`.
