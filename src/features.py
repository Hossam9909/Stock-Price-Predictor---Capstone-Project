"""
Stock Price Feature Engineering Module

This module provides functions to create technical indicators, lag features,
rolling window statistics, and target variables for stock price prediction.
All functions are designed to avoid lookahead bias for time series modeling.

Author: Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import warnings


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate
    required_columns : List[str]
        List of required column names

    Raises
    ------
    ValueError
        If required columns are missing from DataFrame
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def calculate_sma(df: pd.DataFrame, column: str = 'Adj Close',
                  windows: List[int] = [5, 20, 50]) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA) for specified windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate SMA for
    windows : List[int], default [5, 20, 50]
        List of window sizes for SMA calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA columns added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    for window in windows:
        col_name = f'SMA_{window}'
        df[col_name] = df[column].rolling(window=window, min_periods=1).mean()

    return df


def calculate_ema(df: pd.DataFrame, column: str = 'Adj Close',
                  windows: List[int] = [12, 26]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Average (EMA) for specified windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate EMA for
    windows : List[int], default [12, 26]
        List of window sizes for EMA calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with EMA columns added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    for window in windows:
        col_name = f'EMA_{window}'
        df[col_name] = df[column].ewm(span=window, adjust=False).mean()

    return df


def calculate_rsi(df: pd.DataFrame, column: str = 'Adj Close',
                  window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate RSI for
    window : int, default 14
        Window size for RSI calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with RSI column added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    # Calculate price changes
    delta = df[column].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()

    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    df[f'RSI_{window}'] = rsi
    return df


def calculate_macd(df: pd.DataFrame, column: str = 'Adj Close',
                   fast_period: int = 12, slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate MACD for
    fast_period : int, default 12
        Fast EMA period
    slow_period : int, default 26
        Slow EMA period
    signal_period : int, default 9
        Signal line EMA period

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD, Signal, and Histogram columns added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    # Calculate fast and slow EMAs
    ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram

    return df


def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'Adj Close',
                              window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate Bollinger Bands for
    window : int, default 20
        Window size for moving average and standard deviation
    num_std : float, default 2.0
        Number of standard deviations for bands

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Bands columns added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    # Calculate moving average and standard deviation
    ma = df[column].rolling(window=window, min_periods=1).mean()
    std = df[column].rolling(window=window, min_periods=1).std()

    # Calculate bands
    df[f'BB_Upper_{window}'] = ma + (num_std * std)
    df[f'BB_Middle_{window}'] = ma
    df[f'BB_Lower_{window}'] = ma - (num_std * std)
    df[f'BB_Width_{window}'] = df[f'BB_Upper_{window}'] - \
        df[f'BB_Lower_{window}']
    df[f'BB_Percent_{window}'] = (
        df[column] - df[f'BB_Lower_{window}']) / df[f'BB_Width_{window}']

    return df


def calculate_atr(df: pd.DataFrame, high_col: str = 'High',
                  low_col: str = 'Low', close_col: str = 'Adj Close',
                  window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    high_col : str, default 'High'
        Column name for high prices
    low_col : str, default 'Low'
        Column name for low prices
    close_col : str, default 'Adj Close'
        Column name for close prices
    window : int, default 14
        Window size for ATR calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR column added
    """
    validate_dataframe(df, [high_col, low_col, close_col])
    df = df.copy()

    # Calculate True Range components
    tr1 = df[high_col] - df[low_col]
    tr2 = abs(df[high_col] - df[close_col].shift(1))
    tr3 = abs(df[low_col] - df[close_col].shift(1))

    # True Range is the maximum of the three components
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR as moving average of True Range
    df[f'ATR_{window}'] = true_range.rolling(
        window=window, min_periods=1).mean()

    return df


def create_lag_features(df: pd.DataFrame, columns: List[str],
                        lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Create lag features for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    columns : List[str]
        List of column names to create lags for
    lags : List[int], default [1, 2, 3, 5, 10]
        List of lag periods

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features added
    """
    validate_dataframe(df, columns)
    df = df.copy()

    for column in columns:
        for lag in lags:
            col_name = f'{column}_lag_{lag}'
            df[col_name] = df[column].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame, column: str = 'Adj Close',
                            windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate rolling features for
    windows : List[int], default [5, 10, 20]
        List of window sizes

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    for window in windows:
        # Basic rolling statistics
        df[f'Rolling_Mean_{window}'] = df[column].rolling(
            window=window, min_periods=1).mean()
        df[f'Rolling_Std_{window}'] = df[column].rolling(
            window=window, min_periods=1).std()
        df[f'Rolling_Min_{window}'] = df[column].rolling(
            window=window, min_periods=1).min()
        df[f'Rolling_Max_{window}'] = df[column].rolling(
            window=window, min_periods=1).max()

        # Higher-order moments
        df[f'Rolling_Skew_{window}'] = df[column].rolling(
            window=window, min_periods=1).skew()
        df[f'Rolling_Kurt_{window}'] = df[column].rolling(
            window=window, min_periods=1).kurt()

        # Range and volatility measures
        df[f'Rolling_Range_{window}'] = (df[f'Rolling_Max_{window}'] -
                                         df[f'Rolling_Min_{window}'])
        df[f'Rolling_CV_{window}'] = (df[f'Rolling_Std_{window}'] /
                                      df[f'Rolling_Mean_{window}'])

    return df


def create_return_features(df: pd.DataFrame, column: str = 'Adj Close',
                           periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Create return features for different periods.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to calculate returns for
    periods : List[int], default [1, 5, 10, 20]
        List of periods for return calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with return features added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    for period in periods:
        # Simple returns
        df[f'Return_{period}d'] = df[column].pct_change(periods=period)

        # Log returns
        df[f'LogReturn_{period}d'] = np.log(
            df[column] / df[column].shift(period))

    return df


def create_target_variables(df: pd.DataFrame, column: str = 'Adj Close',
                            horizons: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    """
    Create target variables for future price prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data
    column : str, default 'Adj Close'
        Column name to create targets for
    horizons : List[int], default [1, 7, 14, 28]
        List of future horizons in days

    Returns
    -------
    pd.DataFrame
        DataFrame with target variables added
    """
    validate_dataframe(df, [column])
    df = df.copy()

    for horizon in horizons:
        # Future prices
        df[f'Target_Price_{horizon}d'] = df[column].shift(-horizon)

        # Future returns
        df[f'Target_Return_{horizon}d'] = (
            df[f'Target_Price_{horizon}d'] / df[column]) - 1

        # Future log returns
        df[f'Target_LogReturn_{horizon}d'] = np.log(
            df[f'Target_Price_{horizon}d'] / df[column])

        # Price change direction (binary)
        df[f'Target_Direction_{horizon}d'] = (
            df[f'Target_Price_{horizon}d'] > df[column]).astype(int)

        # Price change magnitude categories
        df[f'Target_Change_{horizon}d'] = df[f'Target_Return_{horizon}d']

    return df


def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all technical indicators in one function.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data (must have OHLCV columns)

    Returns
    -------
    pd.DataFrame
        DataFrame with all technical indicators added
    """
    required_columns = ['High', 'Low', 'Adj Close', 'Volume']
    validate_dataframe(df, required_columns)

    df = df.copy()

    # Technical indicators
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)

    return df


def create_all_features(df: pd.DataFrame,
                        target_horizons: List[int] = [1, 7, 14, 28],
                        include_volume_features: bool = True) -> pd.DataFrame:
    """
    Create all features including technical indicators, lags, rolling stats, and targets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock data (must have OHLCV columns)
    target_horizons : List[int], default [1, 7, 14, 28]
        List of target prediction horizons
    include_volume_features : bool, default True
        Whether to include volume-based features

    Returns
    -------
    pd.DataFrame
        DataFrame with all features and targets added

    Notes
    -----
    This function creates a comprehensive feature set for stock price prediction.
    The resulting DataFrame will have NaN values for the first few rows (due to lags)
    and last few rows (due to forward-looking targets).
    """
    required_columns = ['High', 'Low', 'Adj Close', 'Volume']
    validate_dataframe(df, required_columns)

    df = df.copy()

    print("Creating technical indicators...")
    df = create_technical_indicators(df)

    print("Creating return features...")
    df = create_return_features(df)

    print("Creating lag features...")
    price_columns = ['Adj Close', 'High', 'Low']
    if include_volume_features:
        price_columns.append('Volume')
    df = create_lag_features(df, price_columns)

    print("Creating rolling features...")
    df = create_rolling_features(df, 'Adj Close')
    if include_volume_features:
        df = create_rolling_features(df, 'Volume')

    print("Creating target variables...")
    df = create_target_variables(df, horizons=target_horizons)

    # Additional derived features
    print("Creating additional features...")

    # Price relative to moving averages
    df['Price_to_SMA20'] = df['Adj Close'] / df['SMA_20']
    df['Price_to_SMA50'] = df['Adj Close'] / df['SMA_50']

    # Volume relative to moving average
    if include_volume_features:
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

    # Volatility measures
    df['Price_Volatility_20'] = df['Adj Close'].pct_change().rolling(20).std()
    df['High_Low_Ratio'] = df['High'] / df['Low']

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame,
                        exclude_targets: bool = True,
                        exclude_original: bool = True) -> List[str]:
    """
    Get list of feature columns, excluding targets and original OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features
    exclude_targets : bool, default True
        Whether to exclude target columns
    exclude_original : bool, default True
        Whether to exclude original OHLCV columns

    Returns
    -------
    List[str]
        List of feature column names
    """
    original_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    target_cols = [col for col in df.columns if col.startswith('Target_')]

    feature_cols = list(df.columns)

    if exclude_targets:
        feature_cols = [col for col in feature_cols if col not in target_cols]

    if exclude_original:
        feature_cols = [
            col for col in feature_cols if col not in original_cols]

    return feature_cols


def prepare_modeling_data(df: pd.DataFrame,
                          target_horizon: int = 1,
                          dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling by selecting features and target for specific horizon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all features and targets
    target_horizon : int, default 1
        Target prediction horizon in days
    dropna : bool, default True
        Whether to drop rows with NaN values

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features DataFrame and target Series

    Raises
    ------
    ValueError
        If target column for specified horizon doesn't exist
    """
    target_col = f'Target_Price_{target_horizon}d'

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found. "
                         f"Available targets: {[col for col in df.columns if col.startswith('Target_')]}")

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if dropna:
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

    return X, y


if __name__ == "__main__":
    # Example usage
    import yfinance as yf

    print("Downloading sample data...")
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")

    print(f"Original data shape: {df.shape}")

    # Create all features
    df_features = create_all_features(df)

    print(f"Data with features shape: {df_features.shape}")
    print(f"Feature columns: {len(get_feature_columns(df_features))}")

    # Prepare data for 7-day prediction
    X, y = prepare_modeling_data(df_features, target_horizon=7)
    print(f"Modeling data shape: X={X.shape}, y={y.shape}")
