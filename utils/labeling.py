from typing import Tuple

import numpy as np
import pandas as pd


def build_labels(df: pd.DataFrame, instrument_type: str, horizon: int = 5, vol_window: int = 20) -> pd.DataFrame:
    """
    Generate real trading labels based on price data.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
        instrument_type: The symbol of the instrument (e.g., 'pepe', 'btc').
        horizon: Lookahead period for signal generation (default 5 periods)
        vol_window: Window size for volatility calculation (default 20 periods)

    Returns:
        DataFrame with added label columns:
        - trading_signal: 1 (buy), -1 (sell), 0 (hold). For PEPE, only 1 (buy) or 0 (hold).
        - volatility: Rolling standard deviation
        - market_regime: 'bullish', 'bearish', 'sideways'
        - level_sl: Stop-loss level based on ATR and volatility (PEPE specific logic applied).
        - level_tp: Take-profit level based on ATR and volatility (PEPE specific logic applied).
    """
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        raise ValueError("Missing required price columns")

    # 1. Calculate returns and volatility
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(vol_window).std()

    # 2. Generate trading signals
    future_returns = df["close"].pct_change(horizon).shift(-horizon)
    # Default signal logic: Buy if >1%, else Hold
    df["trading_signal"] = np.where(future_returns > 0.01, 1, 0)
    if instrument_type != 'pepe':
        # Apply sell signal only for non-PEPE instruments
        df["trading_signal"] = np.where(future_returns < -0.01, -1, df["trading_signal"]) # Sell if <-1%
    # For PEPE, the signal remains 0 or 1 as initialized above.
    # Note: Consider adding logging here if needed, ensuring logger is accessible.

    # 3. Determine market regimes
    rolling_mean = df["close"].rolling(vol_window).mean()
    rolling_std = df["close"].rolling(vol_window).std()
    df["market_regime"] = np.where(
        df["close"] > rolling_mean + rolling_std,
        "bullish",
        np.where(df["close"] < rolling_mean - rolling_std, "bearish", "sideways"),
    )

    # 4. Calculate SL/TP levels based on volatility/ATR
    # Calculate ATR (using simple mean of High-Low for now, could use ta library)
    df['atr'] = (df["high"] - df["low"]).rolling(vol_window).mean()

    if instrument_type == 'pepe':
        # PEPE specific SL/TP logic
        # Define high volatility threshold (e.g., 75th percentile of ATR)
        # Ensure enough data points before calculating quantile, handle potential NaNs
        if not df['atr'].dropna().empty:
             atr_threshold_high = df['atr'].quantile(0.75)
             is_high_volatility = df['atr'] > atr_threshold_high
        else:
             # Handle case with insufficient data or all NaNs for ATR
             is_high_volatility = pd.Series([False] * len(df), index=df.index) # Default to not high volatility
             atr_threshold_high = np.nan # Indicate threshold couldn't be calculated

        df["level_sl"] = df["close"] - np.where(is_high_volatility, 1.5 * df['atr'], 1.0 * df['atr'])
        df["level_tp"] = df["close"] + np.where(is_high_volatility, 3.0 * df['atr'], 2.0 * df['atr'])
    else:
        # Default SL/TP logic for other instruments
        df["level_sl"] = df["close"] - 2 * df['atr'] # Example: Default 2*ATR SL
        df["level_tp"] = df["close"] + 3 * df['atr'] # Example: Default 3*ATR TP

    # Clean up intermediate columns
    df.drop(columns=["returns"], inplace=True, errors="ignore")

    return df
