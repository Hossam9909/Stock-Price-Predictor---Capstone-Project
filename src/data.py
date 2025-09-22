"""
Stock Data Download Module

This module provides functionality to download stock data from Yahoo Finance
using yfinance library. It supports downloading multiple tickers and saving
them as CSV files for further processing.

Usage:
    python src/data.py --tickers AAPL GOOGL --start 2020-01-01 --end 2023-12-31
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional
import yfinance as yf
import pandas as pd


def setup_logging():
    """Setup logging configuration for the data module."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD).

    Args:
        date_str (str): Date string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def download_ticker(ticker: str, start: str, end: str, out_dir: str = 'data/raw') -> Optional[str]:
    """
    Download stock data for a single ticker and save to CSV.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
        out_dir (str): Output directory path

    Returns:
        Optional[str]: Path to saved CSV file if successful, None otherwise
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

        # Download data
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None

        # Clean and prepare data
        df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
        df = df.rename(columns={'Adj Close': 'AdjClose'})

        # Remove any rows with all NaN values
        df = df.dropna(how='all')

        # Save to CSV
        out_path = os.path.join(out_dir, f'{ticker}.csv')
        df.to_csv(out_path)

        logger.info(f"Successfully saved {out_path} (rows={len(df)})")
        return out_path

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return None


def download_multiple_tickers(tickers: List[str], start: str, end: str, out_dir: str = 'data/raw') -> List[str]:
    """
    Download stock data for multiple tickers.

    Args:
        tickers (List[str]): List of ticker symbols
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
        out_dir (str): Output directory path

    Returns:
        List[str]: List of successfully downloaded file paths
    """
    logger = logging.getLogger(__name__)
    successful_downloads = []

    logger.info(f"Starting download for {len(tickers)} tickers")

    for ticker in tickers:
        result = download_ticker(ticker.upper(), start, end, out_dir)
        if result:
            successful_downloads.append(result)

    logger.info(
        f"Successfully downloaded {len(successful_downloads)}/{len(tickers)} tickers")
    return successful_downloads


def main():
    """Main function to handle CLI arguments and execute downloads."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Download stock data from Yahoo Finance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        required=True,
        help='Stock ticker symbols (e.g., AAPL GOOGL MSFT)'
    )
    parser.add_argument(
        '--start',
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end',
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--out',
        default='data/raw',
        help='Output directory for CSV files'
    )

    args = parser.parse_args()

    try:
        successful_files = download_multiple_tickers(
            args.tickers,
            args.start,
            args.end,
            args.out
        )

        if successful_files:
            logger.info("Download completed successfully!")
            print(f"\nDownloaded files:")
            for file_path in successful_files:
                print(f"  - {file_path}")
        else:
            logger.error("No files were downloaded successfully")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == '__main__':
    main()
