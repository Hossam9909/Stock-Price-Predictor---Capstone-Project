"""
Stock Data Download and Processing Module

This module provides comprehensive functionality for downloading, processing,
and validating stock data from Yahoo Finance. It includes data cleaning,
outlier detection, missing value handling, and various utility functions
for financial data analysis.

Usage:
    python src/data.py --tickers AAPL GOOGL --start 2020-01-01 --end 2023-12-31
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from src.utils import load_config, setup_logging


def get_default_tickers(config: Dict[str, Any]) -> List[str]:
    """Get default ticker list from configuration."""
    return config.get('data', {}).get('tickers', ['AAPL', 'GOOGL', 'MSFT'])


def get_default_date_range(config: Dict[str, Any]) -> tuple:
    """Get default date range from configuration."""
    try:
        date_config = config.get('data', {}).get('date_range', {})
        start_date = date_config.get('start_date', '2020-01-01')
        end_date = date_config.get('end_date', None)

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        return start_date, end_date
    except:
        return '2020-01-01', datetime.now().strftime('%Y-%m-%d')


def validate_date_format(date_str: Optional[str]) -> bool:
    """
    Validate date string format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if date_str is None or date_str == "":
        return False

    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except (ValueError, TypeError):
        return False


def download_ticker(ticker: str, start: str, end: str, out_dir: str = 'data/raw') -> Optional[str]:
    """
    Download stock data for a single ticker and save to CSV.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        out_dir: Output directory path

    Returns:
        Path to saved CSV file if successful, None otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Validate inputs
        if not validate_date_format(start) or not validate_date_format(end):
            logger.error(
                f"Invalid date format for {ticker}. Use YYYY-MM-DD format.")
            return None

        logger.info(f"Downloading data for {ticker} from {start} to {end}")

        # Create output directory
        os.makedirs(out_dir, exist_ok=True)

        # Download data with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None

        # Debug: Print available columns
        logger.info(f"Available columns for {ticker}: {list(df.columns)}")

        # Handle different column structures
        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            # If MultiIndex, flatten it and take the first level
            df.columns = df.columns.droplevel(
                1) if df.columns.nlevels > 1 else df.columns

        # Define the columns we want to keep, in order of preference
        desired_columns = ['Open', 'High', 'Low',
                           'Close', 'Adj Close', 'Volume']
        available_columns = []

        # Check which columns are actually available
        for col in desired_columns:
            if col in df.columns:
                available_columns.append(col)
            else:
                logger.warning(f"Column '{col}' not found for {ticker}")

        # If we don't have Adj Close, use Close
        if 'Adj Close' not in available_columns and 'Close' in available_columns:
            logger.warning(
                f"Using 'Close' instead of 'Adj Close' for {ticker}")
            df['Adj Close'] = df['Close']
            available_columns.append('Adj Close')

        # Select only available columns
        if available_columns:
            df = df[available_columns].copy()
        else:
            logger.error(f"No recognizable price columns found for {ticker}")
            return None

        # Remove any rows with all NaN values
        df = df.dropna(how='all')

        if df.empty:
            logger.warning(f"No valid data after cleaning for ticker {ticker}")
            return None

        # Save to CSV
        out_path = os.path.join(out_dir, f'{ticker}.csv')
        df.to_csv(out_path)

        logger.info(
            f"Successfully saved {out_path} (rows={len(df)}, columns={list(df.columns)})")
        return out_path

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        # Add more detailed error information
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


def download_multiple_tickers(tickers: List[str], start: str, end: str,
                              out_dir: str = 'data/raw') -> List[Optional[str]]:
    """
    Download stock data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        out_dir: Output directory path

    Returns:
        List of file paths (None for failed downloads)
    """
    logger = logging.getLogger(__name__)
    results = []

    logger.info(f"Starting download for {len(tickers)} tickers")

    for ticker in tickers:
        result = download_ticker(ticker.upper(), start, end, out_dir)
        results.append(result)

    successful_count = sum(1 for r in results if r is not None)
    logger.info(
        f"Successfully downloaded {successful_count}/{len(tickers)} tickers")

    return results


def save_raw_data(df: pd.DataFrame, ticker: str, out_dir: str = 'data/raw') -> str:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        ticker: Ticker symbol for filename
        out_dir: Output directory

    Returns:
        Path to saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f'{ticker}.csv')

    # Ensure consistent index naming
    df_to_save = df.copy()
    if df_to_save.index.name is None:
        df_to_save.index.name = 'Date'

    df_to_save.to_csv(filepath)
    return filepath


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with date index

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Ensure consistent index naming
    df.index.name = "Date"

    # Restore frequency
    if len(df.index) >= 3:
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df.index.freq = inferred_freq
        except Exception:
            df.index.freq = None
    elif len(df.index) == 2:
        delta = df.index[1] - df.index[0]
        if delta.days == 1:
            df.index.freq = "D"  # force daily freq for consecutive 2-day case
        else:
            df.index.freq = None
    else:
        df.index.freq = None

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by handling various data quality issues.

    Args:
        df: Raw stock data DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger = logging.getLogger(__name__)
    df_clean = df.copy()

    # Replace infinite values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

    # Handle zero or negative volumes
    if 'Volume' in df_clean.columns:
        df_clean.loc[df_clean['Volume'] <= 0, 'Volume'] = np.nan

    # Handle negative prices (should not happen in normal data)
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    for col in price_columns:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] <= 0, col] = np.nan

    # Forward fill missing values for price data
    df_clean = df_clean.ffill()

    # Drop any remaining rows with all NaN values
    df_clean = df_clean.dropna(how='all')

    # Final validation - drop rows where High < Low (data integrity issue)
    if 'High' in df_clean.columns and 'Low' in df_clean.columns:
        invalid_rows = df_clean['High'] < df_clean['Low']
        if invalid_rows.any():
            logger.warning(
                f"Removing {invalid_rows.sum()} rows where High < Low")
            df_clean = df_clean[~invalid_rows]

    return df_clean


def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a pandas Series.

    Args:
        series: Data series to analyze
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return z_scores > threshold

    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


def handle_missing_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing data using various imputation methods.

    Args:
        df: DataFrame with missing data
        method: Imputation method ('forward_fill', 'backward_fill', 'interpolate', 'mean', 'drop')

    Returns:
        DataFrame with missing data handled
    """
    df_imputed = df.copy()

    if method == 'forward_fill':
        df_imputed = df_imputed.ffill()  # Updated syntax
    elif method == 'backward_fill':
        df_imputed = df_imputed.bfill()  # Updated syntax
    elif method == 'interpolate':
        df_imputed = df_imputed.interpolate(method='time')
    elif method == 'mean':
        df_imputed = df_imputed.fillna(df_imputed.mean())
    elif method == 'drop':
        df_imputed = df_imputed.dropna()
    else:
        raise ValueError("Invalid imputation method")

    return df_imputed


def validate_data_quality(df: pd.DataFrame, detailed: bool = False) -> Union[bool, Dict[str, Any]]:
    """
    Validate data quality with comprehensive checks.

    Args:
        df: DataFrame to validate
        detailed: Whether to return detailed validation results

    Returns:
        Boolean indicating overall quality, or detailed results dict
    """
    results = {
        'overall': True,
        'issues': [],
        'high_low_check': True,
        'volume_check': True,
        'missing_data_check': True,
        'price_consistency_check': True
    }

    # Check High >= Low
    if 'High' in df.columns and 'Low' in df.columns:
        high_low_violations = (df['High'] < df['Low']).sum()
        if high_low_violations > 0:
            results['high_low_check'] = False
            results['issues'].append(
                f"{high_low_violations} rows where High < Low")

    # Check for negative volumes
    if 'Volume' in df.columns:
        negative_volumes = (df['Volume'] < 0).sum()
        if negative_volumes > 0:
            results['volume_check'] = False
            results['issues'].append(
                f"{negative_volumes} rows with negative volume")

    # Check for excessive missing data
    missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_percentage > 10:  # More than 10% missing
        results['missing_data_check'] = False
        results['issues'].append(
            f"High missing data percentage: {missing_percentage:.1f}%")

    # Check price consistency (Open should be reasonable relative to Close)
    if 'Open' in df.columns and 'Close' in df.columns:
        price_ratio = df['Open'] / df['Close']
        extreme_ratios = ((price_ratio > 2) | (price_ratio < 0.5)).sum()
        if extreme_ratios > 0:
            results['price_consistency_check'] = False
            results['issues'].append(
                f"{extreme_ratios} rows with extreme Open/Close ratios")

    # Overall assessment
    results['overall'] = all([
        results['high_low_check'],
        results['volume_check'],
        results['missing_data_check'],
        results['price_consistency_check']
    ])

    if detailed:
        return results
    else:
        return results['overall']


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get trading days (business days) between start and end dates.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DatetimeIndex of trading days
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    return pd.bdate_range(start=start, end=end)


def align_timestamps(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Align timestamps across multiple DataFrames.

    Args:
        dataframes: List of DataFrames to align

    Returns:
        List of aligned DataFrames with common index
    """
    if not dataframes:
        return []

    # Find common date range
    start_dates = [df.index.min() for df in dataframes]
    end_dates = [df.index.max() for df in dataframes]

    common_start = max(start_dates)
    common_end = min(end_dates)

    # Create a unified date range using the intersection of all indices
    all_dates = set(dataframes[0].index)
    for df in dataframes[1:]:
        all_dates = all_dates.intersection(set(df.index))

    # Sort the common dates
    common_dates = sorted(all_dates)

    # Align all DataFrames to common dates
    aligned_dfs = []
    for df in dataframes:
        aligned_df = df.loc[common_dates].copy()
        aligned_dfs.append(aligned_df)

    return aligned_dfs


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate financial returns from price series.

    Args:
        prices: Price series (e.g., Adj Close)
        method: Return calculation method ('simple' or 'log')

    Returns:
        Series of returns (first value will be NaN)
    """
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("Method must be 'simple' or 'log'")

    return returns


def main():
    """Main function to handle CLI arguments and execute downloads."""
    logger = setup_logging()

    # Load configuration
    config = load_config()
    default_tickers = get_default_tickers(config)
    default_start, default_end = get_default_date_range(config)

    parser = argparse.ArgumentParser(
        description="Download and process stock data from Yahoo Finance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=default_tickers,
        help=f'Stock ticker symbols (default: {default_tickers})'
    )
    parser.add_argument(
        '--start',
        default=default_start,
        help=f'Start date in YYYY-MM-DD format (default: {default_start})'
    )
    parser.add_argument(
        '--end',
        default=default_end,
        help=f'End date in YYYY-MM-DD format (default: {default_end})'
    )
    parser.add_argument(
        '--out',
        default='data/raw',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run data quality validation after download'
    )

    args = parser.parse_args()

    # Reload config if different path specified
    if args.config != 'config/config.yaml':
        config = load_config(args.config)

    try:
        results = download_multiple_tickers(
            args.tickers,
            args.start,
            args.end,
            args.out
        )

        successful_files = [r for r in results if r is not None]

        if successful_files:
            logger.info("Download completed successfully!")
            print(f"\nDownloaded files:")
            for file_path in successful_files:
                print(f"  - {file_path}")

                # Optional data validation
                if args.validate:
                    try:
                        df = load_raw_data(file_path)
                        is_valid = validate_data_quality(df, detailed=False)
                        status = "✅ VALID" if is_valid else "⚠️  ISSUES"
                        print(f"    Data quality: {status}")
                    except Exception as e:
                        print(f"    Validation failed: {e}")
        else:
            logger.error("No files were downloaded successfully")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == '__main__':
    main()
