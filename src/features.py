"""
Feature Engineering Module - USES existing data.py functions

This module provides feature engineering capabilities for stock price data.
It imports and uses existing data.py functions for data loading and configuration.

Author: Stock Price Indicator Project
Dependencies: src.data (existing module)
"""

from data import (
    calculate_returns,
    clean_data,
    load_raw_data,
)
from src.utils import load_config, setup_logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore")

# Setup logging from existing data module
logger = setup_logging()

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================


def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicators from OHLCV data.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV columns

    Returns:
        pd.DataFrame: DataFrame with added technical indicators

    Technical Indicators Added:
        - Simple Moving Averages (SMA_5, SMA_10, SMA_20, SMA_50, SMA_200)
        - Exponential Moving Averages (EMA_12, EMA_26)
        - Relative Strength Index (RSI_14)
        - MACD (MACD, MACD_Signal, MACD_Histogram)
        - Bollinger Bands (BB_Upper, BB_Middle, BB_Lower, BB_Width)
        - Average True Range (ATR_14)
        - Stochastic Oscillator (Stoch_K, Stoch_D)
        - Williams %R (Williams_R)
        - Volume indicators (Volume_SMA, Volume_Ratio)
    """
    result_df = df.copy()

    # Simple Moving Averages
    for period in [5, 10, 20, 50, 200]:
        result_df[f"SMA_{period}"] = result_df["Close"].rolling(
            window=period).mean()

    # Exponential Moving Averages
    result_df["EMA_12"] = result_df["Close"].ewm(span=12).mean()
    result_df["EMA_26"] = result_df["Close"].ewm(span=26).mean()

    # RSI (Relative Strength Index)
    result_df["RSI_14"] = _calculate_rsi(result_df["Close"], window=14)

    # MACD
    macd_data = _calculate_macd(result_df["Close"])
    result_df["MACD"] = macd_data["MACD"]
    result_df["MACD_Signal"] = macd_data["Signal"]
    result_df["MACD_Histogram"] = macd_data["Histogram"]

    # Bollinger Bands
    bb_data = _calculate_bollinger_bands(result_df["Close"])
    result_df["BB_Upper"] = bb_data["Upper"]
    result_df["BB_Middle"] = bb_data["Middle"]
    result_df["BB_Lower"] = bb_data["Lower"]
    result_df["BB_Width"] = bb_data["Width"]

    # Average True Range
    result_df["ATR_14"] = _calculate_atr(result_df, window=14)

    # Stochastic Oscillator
    stoch_data = _calculate_stochastic(result_df)
    result_df["Stoch_K"] = stoch_data["%K"]
    result_df["Stoch_D"] = stoch_data["%D"]

    # Williams %R
    result_df["Williams_R"] = _calculate_williams_r(result_df)

    # Volume indicators
    result_df["Volume_SMA_20"] = result_df["Volume"].rolling(window=20).mean()
    result_df["Volume_Ratio"] = result_df["Volume"] / \
        result_df["Volume_SMA_20"]

    logger.info("Technical indicators created.")
    return result_df


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return {"MACD": macd, "Signal": macd_signal, "Histogram": macd_histogram}


def _calculate_bollinger_bands(
    prices: pd.Series, window: int = 20, std_dev: int = 2
) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = sma + (rolling_std * std_dev)
    lower_band = sma - (rolling_std * std_dev)
    width = upper_band - lower_band
    return {"Upper": upper_band, "Middle": sma, "Lower": lower_band, "Width": width}


def _calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr


def _calculate_stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = df["Low"].rolling(window=k_window).min()
    highest_high = df["High"].rolling(window=k_window).max()
    k_percent = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_window).mean()
    return {"%K": k_percent, "%D": d_percent}


def _calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = df["High"].rolling(window=window).max()
    lowest_low = df["Low"].rolling(window=window).min()
    williams_r = -100 * \
        (highest_high - df["Close"]) / (highest_high - lowest_low)
    return williams_r


# =============================================================================
# LAG FEATURES
# =============================================================================


def create_lag_features(
    df: pd.DataFrame, lags: List[int], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lag features for specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        lags (List[int]): List of lag periods to create
        columns (List[str], optional): Columns to create lags for. If None, uses ['Close', 'Volume']

    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    result_df = df.copy()
    if columns is None:
        columns = ["Close", "Volume"]

    for col in columns:
        if col in result_df.columns:
            for lag in lags:
                result_df[f"{col}_lag_{lag}"] = result_df[col].shift(lag)

    logger.info(f"Lag features created for lags: {lags}")
    return result_df


# =============================================================================
# ROLLING WINDOW FEATURES
# =============================================================================


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[int],
    columns: Optional[List[str]] = None,
    statistics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Args:
        df (pd.DataFrame): Input DataFrame
        windows (List[int]): List of window sizes
        columns (List[str], optional): Columns to calculate statistics for
        statistics (List[str], optional): Statistics to calculate ['mean', 'std', 'min', 'max']

    Returns:
        pd.DataFrame: DataFrame with added rolling features
    """
    result_df = df.copy()
    if columns is None:
        columns = ["Close", "Volume"]
    if statistics is None:
        statistics = ["mean", "std", "min", "max"]

    for col in columns:
        if col in result_df.columns:
            for window in windows:
                for stat in statistics:
                    name = f"{col}_rolling_{window}_{stat}"
                    if stat == "mean":
                        result_df[name] = result_df[col].rolling(window).mean()
                    elif stat == "std":
                        result_df[name] = result_df[col].rolling(window).std()
                    elif stat == "min":
                        result_df[name] = result_df[col].rolling(window).min()
                    elif stat == "max":
                        result_df[name] = result_df[col].rolling(window).max()

    logger.info(f"Rolling features created for windows: {windows}")
    return result_df


# =============================================================================
# RETURN FEATURES
# =============================================================================


def create_return_features(
    df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]
) -> pd.DataFrame:
    """
    Create return-based features using existing calculate_returns function.

    Args:
        df (pd.DataFrame): Input DataFrame with Close prices
        periods (List[int]): List of periods for return calculation

    Returns:
        pd.DataFrame: DataFrame with added return features
    """
    result_df = df.copy()
    for period in periods:
        if period == 1:
            result_df["Returns_1d"] = calculate_returns(
                result_df["Close"], method="simple")
            result_df["Log_Returns_1d"] = calculate_returns(
                result_df["Close"], method="log")
        else:
            result_df[f"Returns_{period}d"] = result_df["Close"].pct_change(
                periods=period)
            result_df[f"Log_Returns_{period}d"] = np.log(
                result_df["Close"] / result_df["Close"].shift(period)
            )

    # Volatility features
    result_df["Volatility_10d"] = result_df["Returns_1d"].rolling(10).std()
    result_df["Volatility_30d"] = result_df["Returns_1d"].rolling(30).std()

    logger.info("Return features created.")
    return result_df


# =============================================================================
# PRICE FEATURES
# =============================================================================


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-based features.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data

    Returns:
        pd.DataFrame: DataFrame with added price features
    """
    result_df = df.copy()
    result_df["High_Low_Ratio"] = result_df["High"] / result_df["Low"]
    result_df["Close_Open_Ratio"] = result_df["Close"] / result_df["Open"]
    result_df["Daily_Range"] = result_df["High"] - result_df["Low"]
    result_df["Daily_Range_Pct"] = (
        result_df["High"] - result_df["Low"]) / result_df["Close"]
    result_df["Gap"] = result_df["Open"] - result_df["Close"].shift(1)
    result_df["Gap_Pct"] = result_df["Gap"] / result_df["Close"].shift(1)
    result_df["Close_Position"] = (result_df["Close"] - result_df["Low"]) / (
        result_df["High"] - result_df["Low"]
    )
    logger.info("Price features created.")
    return result_df


# =============================================================================
# VOLUME FEATURES
# =============================================================================


def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create volume-based features.

    Args:
        df (pd.DataFrame): Input DataFrame with Volume data

    Returns:
        pd.DataFrame: DataFrame with added volume features
    """
    result_df = df.copy()
    result_df["Volume_MA_5"] = result_df["Volume"].rolling(5).mean()
    result_df["Volume_MA_20"] = result_df["Volume"].rolling(20).mean()
    result_df["Volume_Ratio_5"] = result_df["Volume"] / \
        result_df["Volume_MA_5"]
    result_df["Volume_Ratio_20"] = result_df["Volume"] / \
        result_df["Volume_MA_20"]
    result_df["Price_Volume"] = result_df["Close"] * result_df["Volume"]
    result_df["Volume_Price_Trend"] = result_df["Price_Volume"].rolling(
        10).mean()
    logger.info("Volume features created.")
    return result_df


# =============================================================================
# TARGET VARIABLES
# =============================================================================


def create_targets(df: pd.DataFrame, horizons: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    """
    Create target variables for forecasting.

    Args:
        df (pd.DataFrame): Input DataFrame with Close prices
        horizons (List[int]): List of forecast horizons (days)

    Returns:
        pd.DataFrame: DataFrame with added target columns
    """
    result_df = df.copy()
    for h in horizons:
        result_df[f"Target_{h}d"] = result_df["Close"].shift(-h)
    logger.info(f"Target variables created for horizons: {horizons}")
    return result_df


# =============================================================================
# FEATURE SCALING & NORMALIZATION
# =============================================================================


def scale_features(
    df: pd.DataFrame,
    method: str = "standard",
    exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Scale or normalize feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        method (str): "standard" or "minmax"
        exclude (List[str], optional): Columns to exclude (e.g., ['Date','Ticker'])

    Returns:
        Tuple[pd.DataFrame, scaler]: Scaled DataFrame and fitted scaler
    """
    result_df = df.copy()
    if exclude is None:
        exclude = []

    numeric_cols = result_df.select_dtypes(
        include=[np.number]).columns.difference(exclude)

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaling method must be 'standard' or 'minmax'")

    result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
    logger.info(f"Features scaled using {method} method.")
    return result_df, scaler


# =============================================================================
# FEATURE PIPELINE
# =============================================================================


def create_all_features(
    df: pd.DataFrame,
    include_technical: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
    lag_periods: List[int] = [1, 2, 3, 5],
    rolling_windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Create all features in one function call.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data
        include_technical (bool): Whether to include technical indicators
        include_lags (bool): Whether to include lag features
        include_rolling (bool): Whether to include rolling features
        lag_periods (List[int]): Lag periods to create
        rolling_windows (List[int]): Rolling window sizes

    Returns:
        pd.DataFrame: DataFrame with all requested features
    """
    result_df = df.copy()
    if include_technical:
        result_df = create_technical_indicators(result_df)
    result_df = create_price_features(result_df)
    result_df = create_volume_features(result_df)
    result_df = create_return_features(result_df)
    if include_lags:
        result_df = create_lag_features(result_df, lag_periods)
    if include_rolling:
        result_df = create_rolling_features(result_df, rolling_windows)

    logger.info("All features created.")
    return result_df


def apply_configured_features(
    df: pd.DataFrame, config_path: str = "config/config.yaml"
) -> pd.DataFrame:
    """
    Apply feature engineering using parameters from config.yaml.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV
        config_path (str): Path to config.yaml

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    config = load_config(config_path)
    feature_cfg = config.get("features", {})

    lag_periods = feature_cfg.get("lags", [1, 2, 3, 5, 10])
    rolling_windows = feature_cfg.get("rolling_windows", [5, 10, 20])
    horizons = feature_cfg.get("targets", [1, 7, 14, 28])
    scaling = feature_cfg.get("scaling", "standard")

    df = create_all_features(
        df,
        include_technical=True,
        include_lags=True,
        include_rolling=True,
        lag_periods=lag_periods,
        rolling_windows=rolling_windows,
    )
    df = create_targets(df, horizons=horizons)
    df, _ = scale_features(df, method=scaling, exclude=["Ticker"])
    df = df.ffill().dropna()
    logger.info("Configured feature engineering pipeline applied.")
    return df


def process_stock_features(
    ticker: str, data_path: str, config_path: str = "config/config.yaml", **kwargs
) -> pd.DataFrame:
    """
    Process features for a single stock using existing load_raw_data function.

    Args:
        ticker (str): Stock ticker symbol
        data_path (str): Path to data file
        config_path (str): Path to config.yaml
        **kwargs: Extra args for feature generation

    Returns:
        pd.DataFrame: DataFrame with all features and targets
    """
    try:
        stock_data = load_raw_data(data_path)
        stock_data = clean_data(stock_data)
        features_df = apply_configured_features(
            stock_data, config_path=config_path)
        features_df["Ticker"] = ticker
        logger.info(
            f"Features processed for {ticker}, shape={features_df.shape}")
        return features_df
    except Exception as e:
        logger.error(f"Feature processing failed for {ticker}: {e}")
        raise
