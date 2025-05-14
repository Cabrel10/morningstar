#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le chargement et le prétraitement des données pour le modèle monolithique.

Ce module fournit des fonctions pour:
1. Charger les données brutes depuis différentes sources
2. Appliquer les transformations et feature engineering
3. Préparer les données pour l'entraînement et l'inférence
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any, Union
from pathlib import Path

# Configuration du logger
logger = logging.getLogger("data_loader")


def load_data(
    data_path: str,
    parse_dates: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Charge les données depuis diverses sources (parquet, csv, etc.).
    
    Args:
        data_path: Chemin vers le fichier de données
        parse_dates: Si True, parse les dates dans l'index
        **kwargs: Arguments additionnels passés à la fonction de lecture
        
    Returns:
        DataFrame avec les données chargées
    """
    logger.info(f"Chargement des données depuis {data_path}")
    
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path, **kwargs)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, **kwargs)
        if parse_dates and 'index_col' in kwargs:
            df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    logger.info(f"Données chargées: {df.shape[0]} échantillons, {df.shape[1]} colonnes")
    return df


def apply_feature_pipeline(
    df: pd.DataFrame,
    calculate_technical: bool = True,
    calculate_mcp: bool = True,
    calculate_embeddings: bool = False,
    timeframe: str = "1d",
    indicators_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Applique la pipeline de feature engineering aux données brutes.
    
    Args:
        df: DataFrame des données OHLCV
        calculate_technical: Si True, calcule les indicateurs techniques
        calculate_mcp: Si True, calcule les features MCP
        calculate_embeddings: Si True, calcule/charge les embeddings
        timeframe: Timeframe des données ('1d', '1h', etc.)
        indicators_config: Configuration des indicateurs à calculer
        
    Returns:
        DataFrame enrichi avec les features calculées
    """
    logger.info("Application de la pipeline de feature engineering")
    
    # Vérifier la présence des colonnes OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Colonnes OHLCV manquantes: {missing_cols}")
        # Essayer de récupérer les noms de colonnes avec casse différente
        for col in missing_cols.copy():
            col_upper = col.upper()
            if col_upper in df.columns:
                df[col] = df[col_upper]
                missing_cols.remove(col)
    
    if missing_cols:
        raise ValueError(f"Colonnes essentielles manquantes: {missing_cols}")
    
    # Assurer que les colonnes OHLCV sont en minuscules
    for col in required_cols:
        if col.upper() in df.columns and col not in df.columns:
            df[col] = df[col.upper()]
    
    # Assurer que l'index est datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            logger.warning("Impossible de définir un index datetime, création d'un index générique")
            df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    
    # 1. Calcul des indicateurs techniques
    if calculate_technical:
        df = add_technical_indicators(df, indicators_config)
    
    # 2. Calcul des features MCP
    if calculate_mcp:
        df = add_mcp_features(df, timeframe)
    
    # 3. Calcul/Chargement des embeddings
    if calculate_embeddings:
        df = add_embeddings(df)
    
    # Nettoyage: supprimer les lignes avec des valeurs NaN
    nan_count_before = df.isna().sum().sum()
    if nan_count_before > 0:
        logger.warning(f"Valeurs NaN détectées: {nan_count_before}")
        # Remplacer les NaN par des valeurs par défaut
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
    
    logger.info(f"Pipeline terminée: {df.shape[0]} échantillons, {df.shape[1]} colonnes")
    return df


def add_technical_indicators(
    df: pd.DataFrame,
    indicators_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Ajoute des indicateurs techniques au DataFrame.
    
    Args:
        df: DataFrame avec données OHLCV
        indicators_config: Configuration des indicateurs à calculer
        
    Returns:
        DataFrame avec indicateurs techniques ajoutés
    """
    try:
        import talib
    except ImportError:
        logger.warning("TA-Lib non installé. Utilisation d'implémentations alternatives.")
        talib = None
    
    logger.info("Calcul des indicateurs techniques")
    
    # Configuration par défaut si non fournie
    if indicators_config is None:
        indicators_config = {
            "moving_averages": [7, 14, 21, 50, 200],
            "oscillators": ["RSI", "MACD", "Stochastic"],
            "volatility": ["ATR", "Bollinger"],
            "volume": ["OBV", "ADI"]
        }
    
    # Extraire les séries OHLCV
    open_price = df['open'].values
    high_price = df['high'].values
    low_price = df['low'].values
    close_price = df['close'].values
    volume = df['volume'].values
    
    # 1. Calculer les moyennes mobiles
    for period in indicators_config.get("moving_averages", []):
        if talib:
            df[f'tech_sma_{period}'] = talib.SMA(close_price, timeperiod=period)
            df[f'tech_ema_{period}'] = talib.EMA(close_price, timeperiod=period)
        else:
            # Implémentation alternative avec pandas
            df[f'tech_sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'tech_ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # 2. Calculer les oscillateurs
    if "RSI" in indicators_config.get("oscillators", []):
        if talib:
            df['tech_rsi_14'] = talib.RSI(close_price, timeperiod=14)
        else:
            # Implémentation simplifiée
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['tech_rsi_14'] = 100 - (100 / (1 + rs))
    
    if "MACD" in indicators_config.get("oscillators", []):
        if talib:
            df['tech_macd'], df['tech_macd_signal'], df['tech_macd_hist'] = talib.MACD(
                close_price, fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            # Implémentation simplifiée
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['tech_macd'] = ema12 - ema26
            df['tech_macd_signal'] = df['tech_macd'].ewm(span=9, adjust=False).mean()
            df['tech_macd_hist'] = df['tech_macd'] - df['tech_macd_signal']
    
    if "Stochastic" in indicators_config.get("oscillators", []):
        if talib:
            df['tech_slowk'], df['tech_slowd'] = talib.STOCH(
                high_price, low_price, close_price, fastk_period=14, slowk_period=3, slowd_period=3)
        else:
            # Implémentation simplifiée
            high_14 = df['high'].rolling(window=14).max()
            low_14 = df['low'].rolling(window=14).min()
            k = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['tech_slowk'] = k.rolling(window=3).mean()
            df['tech_slowd'] = df['tech_slowk'].rolling(window=3).mean()
    
    # 3. Calculer les indicateurs de volatilité
    if "ATR" in indicators_config.get("volatility", []):
        if talib:
            df['tech_atr'] = talib.ATR(high_price, low_price, close_price, timeperiod=14)
        else:
            # Implémentation simplifiée
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['tech_atr'] = tr.rolling(window=14).mean()
    
    if "Bollinger" in indicators_config.get("volatility", []):
        if talib:
            df['tech_bb_upper'], df['tech_bb_middle'], df['tech_bb_lower'] = talib.BBANDS(
                close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        else:
            # Implémentation simplifiée
            sma20 = df['close'].rolling(window=20).mean()
            stddev = df['close'].rolling(window=20).std()
            df['tech_bb_upper'] = sma20 + 2 * stddev
            df['tech_bb_middle'] = sma20
            df['tech_bb_lower'] = sma20 - 2 * stddev
    
    # 4. Calculer les indicateurs de volume
    if "OBV" in indicators_config.get("volume", []):
        if talib:
            df['tech_obv'] = talib.OBV(close_price, volume)
        else:
            # Implémentation simplifiée
            df['tech_obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # 5. Calculer les ratios et autres indicateurs personnalisés
    # Ratio entre close et SMA200
    if 200 in indicators_config.get("moving_averages", []):
        df['tech_sma200_ratio'] = df['close'] / df[f'tech_sma_{200}']
    
    # Croisements de moyennes mobiles
    if 50 in indicators_config.get("moving_averages", []) and 200 in indicators_config.get("moving_averages", []):
        df['tech_golden_cross'] = (df[f'tech_sma_{50}'] > df[f'tech_sma_{200}']).astype(int)
    
    # Ratio Haut/Bas de la fenêtre
    df['tech_high_low_ratio_14'] = df['high'].rolling(window=14).max() / df['low'].rolling(window=14).min()
    
    # Retours sur différentes périodes
    for period in [1, 3, 5, 10, 21]:
        df[f'tech_return_{period}d'] = df['close'].pct_change(periods=period)
    
    # Ajouter préfixe 'tech_' aux colonnes sans préfixe
    for col in df.columns:
        if not col.startswith('tech_') and col not in ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'instrument']:
            if 'tech_' + col not in df.columns:
                df['tech_' + col] = df[col]
    
    logger.info(f"Indicateurs techniques ajoutés: {df.shape[1] - 5} indicateurs")
    return df


def add_mcp_features(
    df: pd.DataFrame,
    timeframe: str = "1d"
) -> pd.DataFrame:
    """
    Ajoute les features Market Context Processor (MCP) au DataFrame.
    
    Args:
        df: DataFrame avec données OHLCV
        timeframe: Timeframe des données ('1d', '1h', etc.)
        
    Returns:
        DataFrame avec features MCP ajoutées
    """
    logger.info("Calcul des features MCP")
    
    # Fonctions d'extraction temporelle
    if isinstance(df.index, pd.DatetimeIndex):
        # Composantes de date
        df['mcp_hour'] = df.index.hour / 24
        df['mcp_day'] = df.index.day / 31
        df['mcp_month'] = df.index.month / 12
        df['mcp_year'] = (df.index.year - df.index.year.min()) / max(1, df.index.year.max() - df.index.year.min())
        df['mcp_dayofweek'] = df.index.dayofweek / 6  # 0-6
        
        # Composantes cycliques
        df['mcp_day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
        df['mcp_day_cos'] = np.cos(2 * np.pi * df.index.day / 31)
        df['mcp_month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['mcp_month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        df['mcp_dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['mcp_dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Détection de tendance à différentes échelles
    for window in [7, 14, 30, 60]:
        # Calculer la pente de la régression linéaire sur les prix de clôture
        if len(df) >= window:
            x = np.arange(window)
            for i in range(window, len(df) + 1):
                y = df['close'].iloc[i-window:i].values
                if len(y) == window:  # Éviter les séquences incomplètes
                    slope, _ = np.polyfit(x, y, 1)
                    df.loc[df.index[i-1], f'mcp_trend_{window}'] = slope / df['close'].iloc[i-1]
    
    # Volatilité normalisée sur différentes périodes
    for window in [7, 14, 30]:
        df[f'mcp_volatility_{window}'] = df['close'].rolling(window=window).std() / df['close']
    
    # Volumes relatifs
    df['mcp_volume_sma10_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Détection de régime de marché (HMM proxy simplifié)
    # 1: Forte tendance haussière, 2: Tendance haussière modérée, 
    # 3: Consolidation/Range, 4: Tendance baissière modérée, 5: Forte tendance baissière
    
    # Calculer les retours
    returns = df['close'].pct_change(5)
    volatility = returns.rolling(window=20).std()
    
    # Définir les régimes de marché simplifiés
    regime = np.zeros(len(df))
    regime[(returns > 0.02) & (volatility < 0.02)] = 1  # Forte tendance haussière
    regime[(returns > 0.005) & (returns <= 0.02)] = 2   # Tendance haussière modérée
    regime[(returns >= -0.005) & (returns <= 0.005)] = 3  # Consolidation
    regime[(returns < -0.005) & (returns >= -0.02)] = 4  # Tendance baissière modérée
    regime[(returns < -0.02) & (volatility < 0.02)] = 5  # Forte tendance baissière
    
    df['mcp_market_regime'] = regime
    
    # One-hot encoding du régime de marché
    for i in range(1, 6):
        df[f'mcp_regime_{i}'] = (df['mcp_market_regime'] == i).astype(float)
    
    logger.info(f"Features MCP ajoutées: {sum(1 for col in df.columns if col.startswith('mcp_'))} features")
    return df


def add_embeddings(
    df: pd.DataFrame,
    embeddings_source: Optional[str] = None,
    model_name: str = "cryptobert"
) -> pd.DataFrame:
    """
    Ajoute ou calcule les embeddings à partir de texte ou d'une source externe.
    
    Args:
        df: DataFrame des données
        embeddings_source: Chemin vers une source d'embeddings externe (optionnel)
        model_name: Nom du modèle à utiliser ("cryptobert", "finbert", etc.)
        
    Returns:
        DataFrame avec embeddings ajoutés
    """
    logger.info("Ajout des embeddings")
    
    # Si une source d'embeddings est fournie, la charger
    if embeddings_source and os.path.exists(embeddings_source):
        embeddings_df = pd.read_parquet(embeddings_source)
        
        # Fusionner avec le DataFrame principal
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("L'index n'est pas une DatetimeIndex, fusion par position")
            # Supposer que les DataFrames sont alignés par position
            for col in embeddings_df.columns:
                if col not in df.columns:
                    df[col] = embeddings_df[col].values[:len(df)]
        else:
            # Fusionner par index datetime
            common_index = df.index.intersection(embeddings_df.index)
            if len(common_index) > 0:
                for col in embeddings_df.columns:
                    if col not in df.columns:
                        df.loc[common_index, col] = embeddings_df.loc[common_index, col]
            else:
                logger.warning("Pas d'index commun entre les données et les embeddings")
    else:
        # Simuler des embeddings aléatoires si aucune source n'est fournie
        logger.warning("Aucune source d'embeddings fournie. Génération d'embeddings aléatoires.")
        np.random.seed(42)
        embedding_dim = 768  # Dimension standard pour BERT
        random_embeddings = np.random.randn(len(df), embedding_dim) * 0.1
        
        for i in range(embedding_dim):
            df[f'embedding_{i}'] = random_embeddings[:, i]
    
    logger.info(f"Embeddings ajoutés: {sum(1 for col in df.columns if col.startswith('embedding_'))} dimensions")
    return df


def prepare_features_for_model(
    df: pd.DataFrame,
    tech_feature_cols: Optional[List[str]] = None,
    embedding_cols: Optional[List[str]] = None,
    mcp_cols: Optional[List[str]] = None,
    instrument_col: str = "instrument",
    instrument_map: Optional[Dict[str, int]] = None,
    sequence_length: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Prépare les features pour l'entrée du modèle monolithique.
    
    Args:
        df: DataFrame avec les features
        tech_feature_cols: Liste des colonnes de features techniques (si None, détection automatique)
        embedding_cols: Liste des colonnes d'embeddings (si None, détection automatique)
        mcp_cols: Liste des colonnes MCP (si None, détection automatique)
        instrument_col: Nom de la colonne d'instrument 
        instrument_map: Mapping des instruments vers entiers (si None, création automatique)
        sequence_length: Longueur de séquence pour les données séquentielles (optionnel)
        
    Returns:
        Dictionnaire des entrées formatées pour le modèle
    """
    logger.info("Préparation des features pour le modèle")
    
    # Détecter automatiquement les colonnes si non spécifiées
    if tech_feature_cols is None:
        tech_feature_cols = [col for col in df.columns if col.startswith('tech_') or col in ['open', 'high', 'low', 'close', 'volume']]
    
    if embedding_cols is None:
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
    
    if mcp_cols is None:
        mcp_cols = [col for col in df.columns if col.startswith('mcp_')]
    
    # Encoder la colonne d'instrument si présente
    if instrument_col in df.columns:
        if instrument_map is None:
            instruments = df[instrument_col].unique()
            instrument_map = {inst: i for i, inst in enumerate(instruments)}
        
        df['instrument_encoded'] = df[instrument_col].map(instrument_map).fillna(0).astype(int)
    else:
        # Créer une colonne fictive si absente
        df['instrument_encoded'] = 0
    
    # Extraire les features
    X_tech = df[tech_feature_cols].values
    
    # Gérer le cas où embedding_cols ou mcp_cols est vide
    if embedding_cols:
        X_emb = df[embedding_cols].values 
    else:
        # Créer un vecteur d'embedding factice de dimension 1
        X_emb = np.zeros((len(df), 1))
    
    if mcp_cols:
        X_mcp = df[mcp_cols].values
    else:
        # Créer un vecteur MCP factice de dimension 1
        X_mcp = np.zeros((len(df), 1))
    
    X_inst = df['instrument_encoded'].values.reshape(-1, 1)
    
    # Créer des séquences si nécessaire
    if sequence_length is not None:
        # Fonction pour créer des séquences
        def create_sequences(data, seq_length):
            n = len(data)
            seq_data = []
            for i in range(n - seq_length + 1):
                seq_data.append(data[i:i+seq_length])
            return np.array(seq_data)
        
        # Créer des séquences pour les features techniques
        X_tech_seq = create_sequences(X_tech, sequence_length)
        
        # Ajuster les autres entrées pour correspondre aux séquences
        # (utiliser seulement le dernier point pour chaque séquence)
        X_emb_seq = X_emb[sequence_length-1:]
        X_mcp_seq = X_mcp[sequence_length-1:]
        X_inst_seq = X_inst[sequence_length-1:]
        
        # Préparer les données pour le modèle
        inputs = {
            "technical_input": X_tech_seq,
            "embeddings_input": X_emb_seq,
            "mcp_input": X_mcp_seq,
            "instrument_input": X_inst_seq
        }
        
        logger.info(f"Features séquentielles préparées: {len(X_tech_seq)} séquences de longueur {sequence_length}")
    else:
        # Cas non-séquentiel
        inputs = {
            "technical_input": X_tech,
            "embeddings_input": X_emb,
            "mcp_input": X_mcp,
            "instrument_input": X_inst
        }
        
        logger.info(f"Features préparées: {len(X_tech)} échantillons")
    
    return inputs


def load_and_split_data(
    data_path: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    target_cols: Optional[List[str]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les données et les divise en jeux d'entraînement, validation et test.
    
    Args:
        data_path: Chemin vers le fichier de données
        test_size: Proportion de données pour le test
        validation_size: Proportion de données pour la validation
        target_cols: Liste des colonnes cibles (si None, détection automatique)
        **kwargs: Arguments supplémentaires pour load_data
        
    Returns:
        Tuple de DataFrames (train, validation, test)
    """
    # Charger les données
    df = load_data(data_path, **kwargs)
    
    # Détecter les colonnes cibles si non spécifiées
    if target_cols is None:
        target_cols = [col for col in df.columns if col.startswith('target_')]
    
    # Diviser les données de manière temporelle (préserver l'ordre)
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * validation_size)
    
    train_df = df.iloc[:-n_test-n_val].copy()
    val_df = df.iloc[-n_test-n_val:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    
    logger.info(f"Données divisées: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df 