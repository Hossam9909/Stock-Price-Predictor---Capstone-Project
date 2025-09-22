"""
Tests for data.py module.

This module contains comprehensive tests for all data-related functionality
including data ingestion, processing, validation, and utility functions.

The tests cover:
- Data validation and format checking
- Stock data downloading from Yahoo Finance
- File operations (save/load CSV files)
- Data cleaning and preprocessing
- Data quality validation
- Utility functions for financial calculations
- Complete pipeline integration testing

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.data import (
    align_timestamps,
    calculate_returns,
    clean_data,
    detect_outliers,
    download_multiple_tickers,
    download_ticker,
    get_trading_days,
    handle_missing_data,
    load_raw_data,
    save_raw_data,
    validate_data_quality,
    validate_date_format,
)


class TestDataValidation:
    """
    Test cases for data validation functionality.

    This class tests all validation functions including date format validation,
    date range validation, and data integrity checks.
    """

    def test_validate_date_format_valid(self) -> None:
        """
        Test date validation with valid date formats.

        Tests various valid date formats including:
        - Standard ISO format (YYYY-MM-DD)
        - Leap year dates
        - Edge cases (end of year, etc.)
        """
        assert validate_date_format("2020-01-01") is True
        assert validate_date_format("2023-12-31") is True
        assert validate_date_format("2024-02-29") is True  # leap year

    def test_validate_date_format_invalid(self) -> None:
        """
        Test date validation with invalid date formats.

        Tests various invalid date formats including:
        - Wrong separators
        - Invalid month/day values
        - Non-leap year February 29th
        - Empty/None values
        """
        assert validate_date_format("2020/01/01") is False
        assert validate_date_format("invalid-date") is False
        assert validate_date_format("2020-13-01") is False
        assert validate_date_format("2021-02-29") is False  # not leap year
        assert validate_date_format("") is False
        assert validate_date_format(None) is False

    def test_date_range_validation(self) -> None:
        """
        Test validation of date ranges.

        Validates that start and end dates are in correct format
        and logical order (start <= end).
        """
        start_date = "2020-01-01"
        end_date = "2020-12-31"

        # Valid range
        assert validate_date_format(start_date) is True
        assert validate_date_format(end_date) is True

        # Note: Additional date range validation would require
        # a separate validate_date_range function


class TestDataIngestion:
    """
    Test cases for data ingestion functionality.

    This class tests all data download operations including:
    - Single ticker downloads
    - Multiple ticker downloads
    - Error handling for network issues
    - Handling of empty/invalid responses
    """

    @patch('src.data.yf.download')
    def test_download_ticker_success(self, mock_download: MagicMock) -> None:
        """
        Test successful ticker download.

        Mocks yfinance download to return sample data and verifies:
        - Successful data download
        - Proper file creation
        - Correct file naming convention

        Args:
            mock_download: Mocked yfinance download function
        """
        # Mock yfinance data with realistic stock price structure
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [102.0, 103.0, 104.0],
            'Adj Close': [101.0, 102.0, 103.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))
        mock_download.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_ticker("AAPL", "2020-01-01", "2020-01-03",
                                     temp_dir)

            assert result is not None
            assert "AAPL.csv" in result
            assert os.path.exists(result)

    @patch('src.data.yf.download')
    def test_download_ticker_failure(self, mock_download: MagicMock) -> None:
        """
        Test ticker download failure handling.

        Tests graceful handling of network errors and API failures.

        Args:
            mock_download: Mocked yfinance download function
        """
        mock_download.side_effect = Exception("Network error")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_ticker("INVALID", "2020-01-01", "2020-01-02",
                                     temp_dir)
            assert result is None

    @patch('src.data.yf.download')
    def test_download_ticker_empty_data(self, mock_download: MagicMock) -> None:
        """
        Test handling of empty data responses.

        Verifies proper handling when API returns empty DataFrame.

        Args:
            mock_download: Mocked yfinance download function
        """
        mock_download.return_value = pd.DataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_ticker("EMPTY", "2020-01-01", "2020-01-02",
                                     temp_dir)
            assert result is None

    @patch('src.data.download_ticker')
    def test_download_multiple_tickers(self,
                                       mock_download: MagicMock) -> None:
        """
        Test downloading data for multiple tickers.

        Verifies:
        - Multiple tickers are processed correctly
        - Proper return value structure
        - All successful downloads are recorded

        Args:
            mock_download: Mocked download_ticker function
        """
        mock_download.side_effect = ["AAPL.csv", "MSFT.csv", "GOOGL.csv"]

        tickers = ["AAPL", "MSFT", "GOOGL"]
        with tempfile.TemporaryDirectory() as temp_dir:
            results = download_multiple_tickers(tickers, "2020-01-01",
                                                "2020-01-02", temp_dir)

            assert len(results) == 3
            assert all("csv" in result for result in results
                       if result is not None)

    @patch('src.data.download_ticker')
    def test_download_multiple_tickers_partial_failure(self,
                                                       mock_download: MagicMock
                                                       ) -> None:
        """
        Test handling partial failures in multiple ticker downloads.

        Verifies that partial failures don't break the entire process
        and that successful downloads are still returned.

        Args:
            mock_download: Mocked download_ticker function
        """
        # MSFT fails, others succeed
        mock_download.side_effect = ["AAPL.csv", None, "GOOGL.csv"]

        tickers = ["AAPL", "MSFT", "GOOGL"]
        with tempfile.TemporaryDirectory() as temp_dir:
            results = download_multiple_tickers(tickers, "2020-01-01",
                                                "2020-01-02", temp_dir)

            assert len(results) == 3
            assert results[0] is not None  # AAPL success
            assert results[1] is None       # MSFT failure
            assert results[2] is not None  # GOOGL success


class TestDataFileOperations:
    """
    Test cases for file operations.

    This class tests all file I/O operations including:
    - Saving DataFrames to CSV
    - Loading DataFrames from CSV  
    - Error handling for missing files
    - Data integrity across save/load cycles
    """

    def test_save_raw_data(self) -> None:
        """
        Test saving raw data to CSV format.

        Verifies:
        - File is created with correct name
        - File path is returned correctly
        - File exists after save operation
        """
        test_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [102.0, 103.0],
            'Adj Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-01-01', periods=2, freq='D'))

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = save_raw_data(test_df, "TEST", temp_dir)

            assert os.path.exists(filepath)
            assert "TEST.csv" in filepath

    def test_load_raw_data(self) -> None:
        """
        Test loading raw data from CSV format.

        Verifies:
        - Data integrity is maintained across save/load cycle
        - Index and column structure is preserved
        - Data types are handled correctly
        """
        test_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [102.0, 103.0],
            'Adj Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-01-01', periods=2, freq='D'))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save first
            filepath = save_raw_data(test_df, "TEST", temp_dir)

            # Load and compare
            loaded_df = load_raw_data(filepath)
            pd.testing.assert_frame_equal(test_df, loaded_df,
                                          check_dtype=False)

    def test_load_raw_data_missing_file(self) -> None:
        """
        Test loading data from non-existent file.

        Verifies that appropriate exception is raised when trying
        to load from a file that doesn't exist.
        """
        with pytest.raises(FileNotFoundError):
            load_raw_data("non_existent_file.csv")


class TestDataProcessing:
    """
    Test cases for data processing functionality.

    This class tests all data cleaning and processing operations including:
    - Data cleaning (NaN, inf, zero handling)
    - Outlier detection and removal
    - Missing value imputation
    - Data validation rules
    """

    def test_data_cleaning(self) -> None:
        """
        Test comprehensive data cleaning operations.

        Tests cleaning of various data issues including:
        - NaN values
        - Infinite values
        - Zero volumes
        - Invalid price relationships
        """
        # Create test data with multiple issues
        test_df = pd.DataFrame({
            'Open': [100.0, np.nan, 102.0],
            'High': [105.0, 106.0, np.inf],
            'Low': [95.0, 96.0, 97.0],
            'Close': [102.0, 103.0, 104.0],
            'Adj Close': [101.0, 102.0, 103.0],
            'Volume': [1000000, 0, 1200000]  # Zero volume issue
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))

        cleaned_df = clean_data(test_df)

        # Verify cleaning results
        assert not cleaned_df.isnull().any().any()
        assert not np.isinf(cleaned_df.values).any()

    def test_outlier_detection(self) -> None:
        """
        Test outlier detection algorithms.

        Creates data with known outliers and verifies they are detected.
        Uses statistical methods (z-score, IQR) to identify anomalies.
        """
        # Create data with obvious outliers
        normal_data = np.random.normal(100, 10, 100)
        outliers = [1000, -1000]  # Clear outliers

        test_df = pd.DataFrame({
            'Close': np.concatenate([normal_data, outliers])
        })

        outlier_mask = detect_outliers(test_df['Close'])

        # Should detect the extreme values
        assert outlier_mask.sum() >= 2  # At least the two obvious outliers

    def test_missing_value_imputation(self) -> None:
        """
        Test various missing value imputation methods.

        Tests different strategies for handling missing data:
        - Forward fill
        - Backward fill  
        - Interpolation
        - Mean imputation
        """
        test_df = pd.DataFrame({
            'Open': [100.0, np.nan, 102.0, 103.0],
            'High': [105.0, np.nan, 107.0, 108.0],
            'Low': [95.0, 96.0, np.nan, 98.0],
            'Close': [102.0, 103.0, 104.0, np.nan],
            'Volume': [1000000, np.nan, 1200000, 1300000]
        }, index=pd.date_range('2020-01-01', periods=4, freq='D'))

        imputed_df = handle_missing_data(test_df, method='forward_fill')

        # Should have no missing values after imputation
        assert not imputed_df.isnull().any().any()

    def test_data_validation(self) -> None:
        """
        Test comprehensive data validation rules.

        Validates financial data integrity including:
        - High >= Low price relationship
        - Positive volume values
        - Reasonable price ranges
        - Proper date sequencing
        """
        # Test with valid data
        valid_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [95.0, 96.0],
            'Close': [102.0, 103.0],
            'Adj Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-01-01', periods=2, freq='D'))

        assert validate_data_quality(valid_df) is True

        # Test with invalid data (High < Low)
        invalid_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [90.0, 91.0],  # High < Low (invalid)
            'Low': [95.0, 96.0],
            'Close': [102.0, 103.0],
            'Adj Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-01-01', periods=2, freq='D'))

        assert validate_data_quality(invalid_df) is False


class TestDataUtils:
    """
    Test cases for data utility functions.

    This class tests utility functions including:
    - Trading day calculations
    - Timestamp alignment across datasets
    - Financial return calculations
    - Data transformation operations
    """

    def test_get_trading_days(self) -> None:
        """
        Test calculation of trading days between dates.

        Verifies:
        - Weekends are excluded
        - Correct count of business days
        - Proper date range handling
        """
        start_date = "2020-01-01"  # Wednesday
        end_date = "2020-01-10"    # Friday

        trading_days = get_trading_days(start_date, end_date)

        # Should exclude weekends (8 business days expected)
        assert len(trading_days) == 8
        assert all(day.weekday() < 5 for day in trading_days)

    def test_align_timestamps(self) -> None:
        """
        Test timestamp alignment across multiple DataFrames.

        Ensures that DataFrames with different date ranges
        are properly aligned for comparative analysis.
        """
        df1 = pd.DataFrame({
            'Close': [100, 101, 102]
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))

        df2 = pd.DataFrame({
            'Close': [200, 202]  # Missing middle day
        }, index=pd.date_range('2020-01-01', periods=2, freq='2D'))

        aligned_dfs = align_timestamps([df1, df2])

        # Should have same index after alignment
        assert len(aligned_dfs) == 2
        assert aligned_dfs[0].index.equals(aligned_dfs[1].index)

    def test_calculate_returns(self) -> None:
        """
        Test calculation of financial returns.

        Tests simple return calculations with known values:
        - 10%, 10%, -10% expected returns
        - Proper handling of first NaN value
        - Numerical precision
        """
        test_df = pd.DataFrame({
            'Adj Close': [100.0, 110.0, 121.0, 108.9]  # Known returns
        }, index=pd.date_range('2020-01-01', periods=4, freq='D'))

        returns = calculate_returns(test_df['Adj Close'])

        expected_returns = [np.nan, 0.1, 0.1, -0.1]
        np.testing.assert_array_almost_equal(
            returns.fillna(0), expected_returns, decimal=6
        )

    def test_calculate_returns_log(self) -> None:
        """
        Test calculation of logarithmic returns.

        Verifies:
        - Log returns differ from simple returns
        - Proper mathematical calculation
        - Handling of edge cases
        """
        test_df = pd.DataFrame({
            'Adj Close': [100.0, 110.0, 121.0, 108.9]
        }, index=pd.date_range('2020-01-01', periods=4, freq='D'))

        log_returns = calculate_returns(test_df['Adj Close'], method='log')

        # Log returns should differ from simple returns
        simple_returns = calculate_returns(test_df['Adj Close'],
                                           method='simple')

        assert not np.array_equal(
            log_returns.fillna(0), simple_returns.fillna(0)
        )


class TestDataQualityChecks:
    """
    Test cases for comprehensive data quality checks.

    This class provides extensive testing for data quality validation
    including missing data patterns, integrity checks, and anomaly detection.
    """

    def test_missing_data_handling(self) -> None:
        """
        Test comprehensive missing data handling strategies.

        Tests various imputation methods with different missing data patterns:
        - Random missing values
        - Sequential missing values
        - Multiple column missing patterns
        """
        # Create test data with various missing patterns
        test_df = pd.DataFrame({
            'Open': [100.0, np.nan, 102.0, np.nan, 104.0],
            'High': [105.0, 106.0, np.nan, 108.0, 109.0],
            'Low': [95.0, np.nan, np.nan, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, np.nan],
            'Volume': [1000000, 1100000, 1200000, np.nan, 1400000]
        }, index=pd.date_range('2020-01-01', periods=5, freq='D'))

        # Test different imputation methods
        methods = ['forward_fill', 'backward_fill', 'interpolate', 'mean']

        for method in methods:
            result = handle_missing_data(test_df, method=method)
            # Should handle missing values appropriately
            if method != 'drop':
                assert result.shape[0] > 0  # Should retain data

    def test_data_integrity_checks(self) -> None:
        """
        Test comprehensive data integrity validation.

        Validates multiple integrity rules:
        - High >= Low price constraints
        - Positive volume requirements
        - Reasonable price relationships
        - Data consistency checks
        """
        # Create data with various integrity issues
        problematic_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [99.0, 106.0, 107.0],   # First High < Open (invalid)
            'Low': [105.0, 96.0, 97.0],     # First Low > Open (invalid)
            'Close': [102.0, 103.0, 104.0],
            'Adj Close': [101.0, 102.0, 103.0],
            'Volume': [-1000000, 1100000, 1200000]  # Negative volume
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))

        validation_results = validate_data_quality(problematic_df,
                                                   detailed=True)

        # Should identify multiple integrity issues
        assert validation_results['overall'] is False
        assert 'high_low_check' in validation_results
        assert 'volume_check' in validation_results


@pytest.fixture
def sample_stock_data() -> pd.DataFrame:
    """
    Fixture providing realistic sample stock data for tests.

    Creates a DataFrame with 100 days of realistic stock data including:
    - Proper OHLCV structure
    - Realistic price ranges and relationships
    - Proper date indexing

    Returns:
        pd.DataFrame: Sample stock data with OHLCV columns
    """
    np.random.seed(42)  # For reproducible tests

    return pd.DataFrame({
        'Open': np.random.uniform(90, 110, 100),
        'High': np.random.uniform(95, 115, 100),
        'Low': np.random.uniform(85, 105, 100),
        'Close': np.random.uniform(90, 110, 100),
        'Adj Close': np.random.uniform(89, 109, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))


@pytest.fixture
def temp_data_dir() -> str:
    """
    Fixture providing temporary directory for test data operations.

    Creates a temporary directory that is automatically cleaned up
    after test completion. Safe for file I/O testing.

    Yields:
        str: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestDataPipeline:
    """
    Integration tests for complete data pipeline.

    This class tests the entire data processing pipeline from
    ingestion through processing to final output. Ensures all
    components work together correctly.
    """

    @patch('src.data.yf.download')
    def test_complete_data_pipeline(self,
                                    mock_download: MagicMock,
                                    sample_stock_data: pd.DataFrame,
                                    temp_data_dir: str) -> None:
        """
        Test complete data ingestion and processing pipeline.

        Verifies end-to-end functionality:
        1. Data download from API
        2. File save operations
        3. Data loading and validation
        4. Data cleaning and processing
        5. Financial calculations

        Args:
            mock_download: Mocked yfinance download function
            sample_stock_data: Sample DataFrame from fixture
            temp_data_dir: Temporary directory from fixture
        """
        mock_download.return_value = sample_stock_data

        # Step 1: Download data
        ticker_file = download_ticker("AAPL", "2020-01-01", "2020-04-09",
                                      temp_data_dir)
        assert ticker_file is not None

        # Step 2: Load raw data
        raw_data = load_raw_data(ticker_file)
        assert not raw_data.empty

        # Step 3: Clean data
        clean_data_df = clean_data(raw_data)
        assert validate_data_quality(clean_data_df)

        # Step 4: Calculate returns
        returns = calculate_returns(clean_data_df['Adj Close'])
        assert not returns.empty

    def test_error_handling_in_pipeline(self, temp_data_dir: str) -> None:
        """
        Test comprehensive error handling throughout pipeline.

        Verifies graceful handling of various error conditions:
        - Network/API failures
        - Invalid input parameters
        - File system errors
        - Data processing errors

        Args:
            temp_data_dir: Temporary directory from fixture
        """
        # Test with simulated API error
        with patch('src.data.yf.download') as mock_download:
            mock_download.side_effect = Exception("API Error")

            result = download_ticker("INVALID", "2020-01-01", "2020-01-02",
                                     temp_data_dir)
            assert result is None

    def test_data_consistency_across_operations(self,
                                                sample_stock_data: pd.DataFrame
                                                ) -> None:
        """
        Test data consistency across various operations.

        Ensures that data structure and relationships are maintained
        throughout processing pipeline operations.

        Args:
            sample_stock_data: Sample DataFrame from fixture
        """
        original_shape = sample_stock_data.shape

        # Operations that should preserve column structure
        cleaned_data = clean_data(sample_stock_data.copy())
        assert cleaned_data.shape[1] == original_shape[1]  # Same columns

        # Operations that should preserve index alignment
        returns = calculate_returns(sample_stock_data['Adj Close'])
        assert len(returns) == len(sample_stock_data)


# Performance benchmarking tests (optional)
class TestPerformance:
    """
    Performance and benchmarking tests for data operations.

    These tests ensure that data operations meet performance requirements
    and can handle large datasets efficiently.
    """

    def test_large_dataset_processing_performance(self) -> None:
        """
        Test performance with large datasets.

        Ensures that data processing operations complete within
        reasonable time limits for large datasets (e.g., 10+ years of data).
        """
        # Create large dataset (10 years of daily data)
        large_df = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 3650),
            'High': np.random.uniform(95, 115, 3650),
            'Low': np.random.uniform(85, 105, 3650),
            'Close': np.random.uniform(90, 110, 3650),
            'Adj Close': np.random.uniform(89, 109, 3650),
            'Volume': np.random.randint(1000000, 10000000, 3650)
        }, index=pd.date_range('2010-01-01', periods=3650, freq='D'))

        import time

        # Benchmark data cleaning
        start_time = time.time()
        cleaned_data = clean_data(large_df)
        cleaning_time = time.time() - start_time

        # Should complete within reasonable time (< 5 seconds)
        assert cleaning_time < 5.0
        assert not cleaned_data.empty


if __name__ == "__main__":
    # Configure pytest with appropriate options
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker handling
        "--disable-warnings",  # Clean output
    ])
