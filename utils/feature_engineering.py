from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from .market_regime import MarketRegimeDetector # Import HMM detector

# Configuration (peut être externalisée dans config.yaml)
DEFAULT_RSI_PERIOD = 14
DEFAULT_SMA_SHORT = 20
DEFAULT_SMA_LONG = 50
DEFAULT_EMA_SHORT = 12
DEFAULT_EMA_LONG = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2
DEFAULT_ATR_PERIOD = 14
DEFAULT_STOCH_K = 14
DEFAULT_STOCH_D = 3
DEFAULT_STOCH_SMOOTH_K = 3


def compute_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Calcule la Moyenne Mobile Simple (SMA)."""
    return ta.sma(df[column], length=period)


def compute_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Calcule la Moyenne Mobile Exponentielle (EMA)."""
    return ta.ema(df[column], length=period)


def compute_rsi(df: pd.DataFrame, period: int = DEFAULT_RSI_PERIOD, column: str = "close") -> pd.Series:
    """Calcule l'Indice de Force Relative (RSI)."""
    return ta.rsi(df[column], length=period)


def compute_macd(
    df: pd.DataFrame,
    fast: int = DEFAULT_EMA_SHORT,
    slow: int = DEFAULT_EMA_LONG,
    signal: int = DEFAULT_MACD_SIGNAL,
    column: str = "close",
) -> pd.DataFrame:
    """
    Calcule la Convergence/Divergence de Moyenne Mobile (MACD).
    Retourne un DataFrame avec MACD, histogramme (MACDh) et signal (MACDs).
    """
    macd_df = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
    # Renommer les colonnes pour la clarté
    macd_df.columns = ["MACD", "MACDh", "MACDs"]
    return macd_df[["MACD", "MACDs", "MACDh"]]  # Réorganiser pour correspondre à l'ordre commun


def compute_bollinger_bands(
    df: pd.DataFrame, period: int = DEFAULT_BBANDS_PERIOD, std_dev: float = DEFAULT_BBANDS_STDDEV, column: str = "close"
) -> pd.DataFrame:
    """
    Calcule les Bandes de Bollinger.
    Retourne un DataFrame avec les bandes supérieure (BBU), médiane (BBM) et inférieure (BBL).
    """
    bbands_df = ta.bbands(df[column], length=period, std=std_dev)
    # Renommer les colonnes pour la clarté et l'ordre standard
    bbands_df.columns = ["BBL", "BBM", "BBU", "BBB", "BBP"]  # Lower, Middle, Upper, Bandwidth, Percent
    return bbands_df[["BBU", "BBM", "BBL"]]  # Garder seulement Upper, Middle, Lower


def compute_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """Calcule l'Average True Range (ATR)."""
    return ta.atr(df[high_col], df[low_col], df[close_col], length=period)


def compute_stochastics(
    df: pd.DataFrame,
    k: int = DEFAULT_STOCH_K,
    d: int = DEFAULT_STOCH_D,
    smooth_k: int = DEFAULT_STOCH_SMOOTH_K,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Calcule l'Oscillateur Stochastique.
    Retourne un DataFrame avec %K (STOCHk) et %D (STOCHd).
    """
    stoch_df = ta.stoch(df[high_col], df[low_col], df[close_col], k=k, d=d, smooth_k=smooth_k)
    # Renommer les colonnes pour la clarté
    stoch_df.columns = ["STOCHk", "STOCHd"]
    return stoch_df


def integrate_llm_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Placeholder] Intègre les données contextuelles générées par un LLM.
    Cette fonction simulera l'ajout de colonnes pour le résumé et l'embedding.
    Dans une implémentation réelle, elle appellerait une API LLM.
    """
    # Simule l'ajout de colonnes - à remplacer par un appel API réel
    df["llm_context_summary"] = "Placeholder LLM Summary"
    # Simule un embedding (par exemple, un vecteur de petite taille ou un JSON)
    # Pour un vrai embedding, ce serait une liste/array numpy ou stocké différemment
    df["llm_embedding"] = "[0.1, -0.2, 0.3]"  # Exemple simplifié en string

    # TODO: Ajouter la logique d'appel à l'API LLM (OpenAI, HuggingFace, etc.)
    # TODO: Gérer la synchronisation temporelle précise entre les données de marché et le contexte LLM.
    print("WARNING: integrate_llm_context is a placeholder and does not call a real LLM API.")
    return df


def apply_feature_pipeline(df: pd.DataFrame, include_llm: bool = False) -> pd.DataFrame:
    """
    Applique le pipeline complet de feature engineering au DataFrame.
    Calcule les 38 indicateurs techniques requis et intègre (optionnellement) le contexte LLM.
    """
    print("Applying feature engineering pipeline...")

    # Vérifications initiales
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame missing required columns: {required_cols}")

    # --- Indicateurs de base (5) ---
    print("Calculating base features...")
    base_features = ["open", "high", "low", "close", "volume"]

    # --- Indicateurs classiques (14) ---
    print("Calculating classic technical indicators...")
    # Moyennes mobiles
    df["SMA_short"] = compute_sma(df, period=DEFAULT_SMA_SHORT)
    df["SMA_long"] = compute_sma(df, period=DEFAULT_SMA_LONG)
    df["EMA_short"] = compute_ema(df, period=DEFAULT_EMA_SHORT)
    df["EMA_long"] = compute_ema(df, period=DEFAULT_EMA_LONG)

    # Momentum
    df["RSI"] = compute_rsi(df)
    macd_df = compute_macd(df)
    df = pd.concat([df, macd_df], axis=1)

    # Volatilité
    bbands_df = compute_bollinger_bands(df)
    df = pd.concat([df, bbands_df], axis=1)
    df["ATR"] = compute_atr(df)

    # Stochastique
    stoch_df = compute_stochastics(df)
    df = pd.concat([df, stoch_df], axis=1)

    # --- Nouveaux indicateurs (19) ---
    print("Calculating additional indicators...")
    # Momentum additionnels
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    df["CCI"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    df["Momentum"] = ta.mom(df["close"], length=10)
    df["ROC"] = ta.roc(df["close"], length=12)
    df["Williams_%R"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    # TRIX retourne 2 colonnes, on ne garde que la première (TRIX_15_9)
    trix_df = ta.trix(df["close"], length=15)
    df["TRIX"] = trix_df["TRIX_15_9"]
    df["Ultimate_Osc"] = ta.uo(df["high"], df["low"], df["close"])
    df["DPO"] = ta.dpo(df["close"], length=20)

    # Volume
    df["OBV"] = ta.obv(df["close"], df["volume"])
    df["VWMA"] = ta.vwma(df["close"], df["volume"], length=20)
    df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
    df["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    # Indicateurs avancés
    # Parabolic SAR retourne 4 colonnes, on prend PSARl_0.02_0.2
    psar_df = ta.psar(df["high"], df["low"], df["close"])
    df["Parabolic_SAR"] = psar_df["PSARl_0.02_0.2"]

    # Ichimoku Cloud - gère le tuple retourné
    ichimoku_result = ta.ichimoku(df["high"], df["low"], df["close"])
    if isinstance(ichimoku_result, tuple):
        # Version récente de pandas_ta retourne un tuple
        ichimoku_df = ichimoku_result[0]  # Premier élément du tuple est le DataFrame
        df["Ichimoku_Tenkan"] = ichimoku_df["ITS_9"]
        df["Ichimoku_Kijun"] = ichimoku_df["IKS_26"]
        df["Ichimoku_SenkouA"] = ichimoku_df["ISA_9"]
        df["Ichimoku_SenkouB"] = ichimoku_df["ISB_26"]
        df["Ichimoku_Chikou"] = ichimoku_df["ICS_26"]
    else:
        # Ancienne version retourne directement le DataFrame
        df["Ichimoku_Tenkan"] = ichimoku_result["ITS_9"]
        df["Ichimoku_Kijun"] = ichimoku_result["IKS_26"]
        df["Ichimoku_SenkouA"] = ichimoku_result["ISA_9"]
        df["Ichimoku_SenkouB"] = ichimoku_result["ISB_26"]
        df["Ichimoku_Chikou"] = ichimoku_result["ICS_26"]

    # Moyennes mobiles adaptatives (KAMA retourne une seule colonne)
    df["KAMA"] = ta.kama(df["close"], length=10, fast=2, slow=30)

    # --- Indicateurs supplémentaires pour atteindre 38 (5) ---
    print("Calculating 5 additional indicators for 38 total...")
    # VWAP (nécessite high, low, close, volume)
    # S'assurer que l'index est trié dans l'ordre chronologique
    df = df.sort_index(ascending=True)
    df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    # Stochastic RSI (nécessite close, utilise RSI déjà calculé si possible)
    # Recalculer RSI si nécessaire pour stochrsi ou s'assurer qu'il est présent
    if "RSI" not in df.columns:
        df["RSI"] = compute_rsi(df)  # Assurer que RSI est calculé avant
    stoch_rsi_df = ta.stochrsi(
        df["RSI"], length=14, rsi_length=14, k=3, d=3
    )  # Utilise les params par défaut de stochrsi
    df["STOCHRSIk"] = stoch_rsi_df["STOCHRSIk_14_14_3_3"]  # Garder seulement %K

    # Chande Momentum Oscillator
    df["CMO"] = ta.cmo(df["close"], length=14)

    # Percentage Price Oscillator (garder PPO principal)
    ppo_df = ta.ppo(df["close"], fast=12, slow=26, signal=9)
    df["PPO"] = ppo_df["PPO_12_26_9"]

    # Fisher Transform (garder la transformation principale)
    fisher_df = ta.fisher(df["high"], df["low"], length=9)
    # Utiliser le nom correct retourné par la librairie (FISHERT majuscule)
    if "FISHERT_9_1" in fisher_df.columns:
        df["FISHERt"] = fisher_df["FISHERT_9_1"] # Correction: Utiliser FISHERT_9_1
    else:
        # Si même la colonne majuscule n'est pas là, lever une erreur
        raise KeyError(f"Expected column 'FISHERT_9_1' not found in ta.fisher output. Columns: {fisher_df.columns}")

    # --- Validation des 38 indicateurs techniques ---
    print("Validating 38 technical indicators...")
    # Exclure les colonnes de base OHLCV, les colonnes non techniques et instrument_type
    base_cols = ["open", "high", "low", "close", "volume"]
    non_tech_cols = [
        "trading_signal",
        "volatility",
        "market_regime",
        "level_sl",
        "level_tp",
        "instrument_type", # Exclure instrument_type du compte des features techniqu
        "position_size" # Exclure position_size du comptees
    ]  # Colonnes potentielles futures ou non-features

    # Traiter la colonne 'symbol' comme une feature catégorielle spéciale
    has_symbol = 'symbol' in df.columns
    expected_tech_count = 38  # Nombre standard d'indicateurs techniques
    
    tech_cols = [
        col for col in df.columns if col not in base_cols and col not in non_tech_cols 
        and not col.startswith("llm_") and not col.startswith("mcp_") and not col.startswith("hmm_")
        and col != 'symbol'  # Exclure symbol des colonnes techniques standard
    ]

    num_tech_cols = len(tech_cols)

    # 1. Vérifier le nombre exact de 38 indicateurs techniques (sans compter 'symbol')
    if num_tech_cols != expected_tech_count:
        print(f"Colonnes techniques trouvées ({num_tech_cols}): {tech_cols}")
        raise ValueError(f"ERREUR: Nombre incorrect d'indicateurs techniques. Attendu: {expected_tech_count}, Trouvé: {num_tech_cols}.")

    # 2. Vérifier l'absence de NaN après le calcul initial (avant fillna)
    # Note: Certains indicateurs (ex: KAMA, Ichimoku) peuvent générer des NaN au début.
    # Le fillna plus bas les gérera, mais on peut vérifier ici s'il y a des NaN inattendus.
    # Cette vérification est commentée car le fillna est la stratégie adoptée.
    # if df[tech_cols].isnull().any().any():
    #     print("Attention: Des NaN sont présents dans les indicateurs techniques avant fillna.")
    #     # raise ValueError("ERREUR: Présence de NaN dans les features techniques.")

    # 3. Vérifier l'absence de variance nulle (colonnes constantes)
    # Exclure la colonne 'symbol' de cette vérification si elle est présente dans un groupe
    cols_to_check = tech_cols
    constant_cols = []
    for col in cols_to_check:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        raise ValueError(f"ERREUR: Colonnes techniques constantes détectées: {constant_cols}")

    print("Technical indicator validation complete.")
    # --- Intégration des Régimes de Marché (HMM) ---
    print("Calculating market regimes using HMM...")
    try:
        # Assurer que l'index est de type DatetimeIndex pour le HMM
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Initialiser et entraîner le détecteur HMM
        hmm_detector = MarketRegimeDetector(n_regimes=3) # 3 régimes par défaut
        hmm_detector.fit(df.copy())

        # Prédire les régimes
        regimes = hmm_detector.predict(df.copy())

        # Ajouter les régimes au DataFrame principal
        df["hmm_regime"] = regimes

        # Calculer les probabilités de chaque régime
        features = hmm_detector.prepare_features(df.copy())
        scaled_features = hmm_detector.scaler.transform(features)
        regime_probs = hmm_detector.model.predict_proba(scaled_features)
        
        # Ajouter les probabilités des régimes
        for i in range(hmm_detector.n_regimes):
            df[f"hmm_prob_{i}"] = regime_probs[:, i]

        # Remplir les NaN potentiels introduits par le HMM
        hmm_cols = ["hmm_regime"] + [f"hmm_prob_{i}" for i in range(hmm_detector.n_regimes)]
        df[hmm_cols] = df[hmm_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)

        print(f"Added HMM features: {hmm_cols}")

    except Exception as e:
        print(f"WARNING: Failed to calculate HMM features: {e}")
        # Ajouter des colonnes placeholder en cas d'échec
        df["hmm_regime"] = 0
        for i in range(3): # Supposant 3 régimes par défaut
            df[f"hmm_prob_{i}"] = 1.0 / 3.0


    # --- Intégration du Contexte LLM (Placeholder) ---
    if include_llm:
        print("Integrating LLM context (placeholder)...")
        df = integrate_llm_context(df)

    # --- Nettoyage final ---
    # Remplacer les NaN restants par 0 pour les indicateurs techniques calculés
    # Utiliser la liste `tech_cols` déjà définie (38 indicateurs)
    df[tech_cols] = df[tech_cols].fillna(0)

    # Supprimer uniquement les lignes où les colonnes de base OHLCV sont NaN
    base_cols = ["open", "high", "low", "close", "volume"]  # Redéfini pour clarté ici
    initial_rows = len(df)
    df.dropna(subset=base_cols, inplace=True)
    print(f"Removed {initial_rows - len(df)} rows with missing base values")

    print("Feature engineering pipeline completed.")
    # TODO: Vérifier la conformité avec le schéma final de 38 colonnes (ajouter/supprimer si besoin)
    # Pour l'instant, on retourne le df avec les colonnes ajoutées.
    return df


# Exemple d'utilisation (peut être mis dans un script de test ou notebook)
if __name__ == "__main__":
    # Créer un DataFrame d'exemple
    data = {
        "timestamp": pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:15:00",
                "2023-01-01 00:30:00",
                "2023-01-01 00:45:00",
                "2023-01-01 01:00:00",
            ]
            * 10
        ),  # Assez de données pour les calculs
        "open": np.random.rand(50) * 100 + 1000,
        "high": np.random.rand(50) * 5 + 1005,
        "low": 1000 - np.random.rand(50) * 5,
        "close": np.random.rand(50) * 10 + 1000,
        "volume": np.random.rand(50) * 1000 + 100,
    }
    sample_df = pd.DataFrame(data)
    sample_df["high"] = sample_df[["open", "close"]].max(axis=1) + np.random.rand(50) * 2
    sample_df["low"] = sample_df[["open", "close"]].min(axis=1) - np.random.rand(50) * 2
    sample_df.set_index("timestamp", inplace=True)

    print("Original DataFrame:")
    print(sample_df.head())

    # Appliquer le pipeline
    features_df = apply_feature_pipeline(sample_df.copy(), include_llm=True)

    print("\nDataFrame with Features:")
    print(features_df.head())
    print("\nColumns added:")
    print(features_df.columns)
