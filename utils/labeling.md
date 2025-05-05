# 📘 Documentation du Module `utils.labeling`

## Objectif

Ce module fournit un ensemble de fonctions pour générer différentes cibles (labels) nécessaires à l'entraînement supervisé de modèles de trading algorithmique. Il vise à transformer les données de marché enrichies en signaux exploitables pour la prédiction.

---

## 🧩 Fonctions Principales

### 1. `generate_binary_labels(df, price_col='close', horizon=15, threshold=0.01)`

**But :** Générer des signaux de trading binaires simples (Achat/Vente).

- **Logique :**
    1. Calcule le changement de prix relatif entre le prix actuel (`price_col`) et le prix `horizon` périodes dans le futur.
    2. Si le changement dépasse `+threshold`, le label est `1` (Achat).
    3. Si le changement est inférieur à `-threshold`, le label est `-1` (Vente).
    4. Sinon, le label est `0` (Neutre/Hold).
- **Paramètres :**
    - `df` (pd.DataFrame): Données d'entrée indexées par timestamp.
    - `price_col` (str): Nom de la colonne de prix à utiliser.
    - `horizon` (int): Nombre de périodes futures pour calculer le changement.
    - `threshold` (float): Seuil de changement de prix relatif (ex: 0.01 pour 1%).
- **Sortie :** `pd.Series` avec les labels (1, -1, 0). Les `horizon` dernières valeurs sont `NaN` car le futur est inconnu.
- **Choix des seuils :** Le `threshold` doit être choisi en fonction de la volatilité typique de l'actif et de l'horizon. Un seuil trop bas génère beaucoup de signaux (potentiellement bruyants), un seuil trop haut génère peu de signaux. À ajuster via backtesting ou analyse statistique.

---

### 2. `generate_multiclass_labels(df, price_col='close', horizon=15, thresholds=(0.005, 0.015))`

**But :** Générer des signaux de trading multi-classes pour capturer l'intensité du mouvement.

- **Logique :** Utilise deux seuils (`lower_threshold`, `upper_threshold`) pour définir 5 classes :
    - `STRONG_BUY (2)`: Changement > `upper_threshold`
    - `BUY (1)`: `lower_threshold` < Changement <= `upper_threshold`
    - `HOLD (0)`: `-lower_threshold` <= Changement <= `lower_threshold`
    - `SELL (-1)`: `-upper_threshold` <= Changement < `-lower_threshold`
    - `STRONG_SELL (-2)`: Changement < `-upper_threshold`
- **Paramètres :**
    - `df` (pd.DataFrame): Données d'entrée.
    - `price_col` (str): Colonne de prix.
    - `horizon` (int): Horizon de prédiction.
    - `thresholds` (Tuple[float, float]): Tuple `(lower_threshold, upper_threshold)`.
- **Sortie :** `pd.Series` avec les labels (-2, -1, 0, 1, 2). Les `horizon` dernières valeurs sont `NaN`.
- **Choix des seuils :** Les seuils doivent être positifs et `lower < upper`. Ils définissent la granularité des signaux. Des seuils plus serrés captureront des mouvements plus petits. À déterminer par analyse de la distribution des rendements futurs.

---

### 3. `generate_market_regimes(df, features=['volatility_atr', 'volume_change'], n_regimes=3, **kmeans_kwargs)`

**But :** Identifier des régimes de marché distincts (ex: tendance haussière volatile, range calme, tendance baissière) en utilisant le clustering.

- **Logique :**
    1. Sélectionne les `features` pertinentes (préalablement calculées, ex: volatilité, momentum, volume).
    2. Supprime les lignes avec des valeurs manquantes pour ces features.
    3. Applique l'algorithme K-Means pour regrouper les points de données en `n_regimes` clusters.
    4. Chaque cluster représente un régime de marché.
- **Paramètres :**
    - `df` (pd.DataFrame): Données d'entrée contenant les `features`.
    - `features` (list): Liste des noms de colonnes à utiliser pour le clustering.
    - `n_regimes` (int): Nombre de clusters (régimes) à identifier.
    - `**kmeans_kwargs`: Arguments additionnels pour `sklearn.cluster.KMeans`.
- **Sortie :** `pd.Series` contenant le numéro du cluster (régime) pour chaque point de données. Les points avec des features manquantes auront `NaN`.
- **Choix des features et `n_regimes` :** Crucial et dépend du contexte. L'analyse exploratoire (ex: méthode du coude, score de silhouette) peut aider à déterminer le nombre optimal de régimes. Les features doivent capturer différentes dynamiques du marché. La mise à l'échelle (scaling) des features est recommandée avant clustering.

---

### 4. `calculate_tp_sl_targets(df, price_col='close', atr_col='atr', tp_multiplier=2.0, sl_multiplier=1.5)`

**But :** Calculer des niveaux dynamiques de Take Profit (TP) et Stop Loss (SL) basés sur la volatilité (ATR).

- **Logique :**
    1. Utilise l'Average True Range (ATR) comme mesure de la volatilité locale.
    2. Calcule les niveaux potentiels de TP et SL en ajoutant/soustrayant un multiple (`tp_multiplier`, `sl_multiplier`) de l'ATR au prix actuel (`price_col`).
    3. Fournit des niveaux pour les positions Long (TP au-dessus, SL en dessous) et Short (TP en dessous, SL au-dessus).
- **Paramètres :**
    - `df` (pd.DataFrame): Données d'entrée avec `price_col` et `atr_col`.
    - `price_col` (str): Colonne de prix de référence.
    - `atr_col` (str): Colonne contenant l'ATR pré-calculé.
    - `tp_multiplier` (float): Multiplicateur ATR pour le TP.
    - `sl_multiplier` (float): Multiplicateur ATR pour le SL.
- **Sortie :** `pd.DataFrame` avec 4 colonnes : `potential_tp_long`, `potential_sl_long`, `potential_tp_short`, `potential_sl_short`.
- **Choix des multiplicateurs :** Dépendent de l'aversion au risque et de la stratégie. Des multiplicateurs plus élevés signifient des objectifs plus larges et des stops plus éloignés (potentiellement moins de trades, mais des gains/pertes plus importants par trade). À optimiser par backtesting.

---

### 5. `build_labels(df, price_col='close', atr_col=None, ..., regime_features=None, ...)`

**But :** Fonction orchestratrice qui appelle les autres fonctions de génération de labels et les combine en un seul DataFrame.

- **Logique :**
    1. Initialise un DataFrame vide `labels_df` avec le même index que `df`.
    2. Appelle `generate_binary_labels` et stocke le résultat.
    3. Appelle `generate_multiclass_labels` et stocke le résultat.
    4. Appelle `generate_market_regimes` (si `regime_features` est fourni et valide) et stocke le résultat.
    5. Appelle `calculate_tp_sl_targets` (si `atr_col` est fourni et valide) et stocke le résultat.
    6. Gère les erreurs potentielles (ex: colonnes manquantes) en affichant des avertissements et en retournant des `NaN` pour les labels non calculés.
- **Paramètres :** Combine les paramètres des fonctions individuelles.
- **Sortie :** `pd.DataFrame` contenant toutes les colonnes de labels générées.

---

## 🔄 Comportement Général et Bonnes Pratiques

- **Alignement Temporel :** Toutes les fonctions retournent des Series/DataFrames alignés sur l'index temporel du DataFrame d'entrée.
- **Gestion des NaN :** Les labels ne peuvent pas être calculés pour les périodes les plus récentes (dépendant de `horizon`) ou lorsque les données d'entrée sont manquantes (ex: ATR manquant pour SL/TP, features manquantes pour régimes). Ces périodes auront des `NaN`.
- **Modularité :** Chaque fonction de génération de label peut être testée et utilisée indépendamment.
- **Configuration :** Les paramètres clés (horizons, seuils, multiplicateurs, features) sont configurables pour permettre l'expérimentation et l'optimisation.

---

## 📡 Extensions Prévues

- **Labels basés sur le Drawdown/Run-up :** Générer des labels indiquant si un certain niveau de drawdown ou de run-up est atteint avant un seuil de profit/perte.
- **Labels de Confiance :** Ajouter un score de confiance aux signaux binaires/multi-classes, basé par exemple sur la volatilité ou le volume au moment du signal.
- **Intégration de Méthodes Alternatives :**
    - Régimes de marché via Hidden Markov Models (HMM).
    - Seuils de labels dynamiques basés sur la volatilité historique.
- **Labels pour l'Apprentissage par Renforcement :** Définir des états et récompenses pour un agent RL.
