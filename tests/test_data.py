import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os
import tempfile

# Import YOUR existing functions
from src.data import (
    validate_date_format,
    download_ticker,
    load_raw_data,
    clean_data,
    validate_data_quality,
    calculate_returns,
    get_default_tickers,
    get_default_date_range,
    save_raw_data,
    detect_outliers,
    handle_missing_data,
    align_timestamps,
)

@pytest.fixture
def sample_dataframe():
    """Fixture for a sample stock data DataFrame."""
    return pd.DataFrame({
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 102, 105, 106],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 103, 101, 104, 105],
        'Adj Close': [102, 103, 101, 104, 105],
        'Volume': [1000, 1100, 900, 1200, 1050]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))


class TestDataValidation:
    def test_validate_date_format_valid(self):
        assert validate_date_format("2020-01-01") == True
        assert validate_date_format("2023-12-31") == True

    def test_validate_date_format_invalid(self):
        assert validate_date_format("invalid") == False
        assert validate_date_format("2020/01/01") == False


class TestConfigHelpers:
    def test_get_default_tickers(self):
        config = {'data': {'tickers': ['TSLA', 'NVDA']}}
        assert get_default_tickers(config) == ['TSLA', 'NVDA']

    def test_get_default_date_range(self):
        config = {'data': {'date_range': {'start_date': '2021-01-01', 'end_date': '2022-01-01'}}}
        start, end = get_default_date_range(config)
        assert start == '2021-01-01'
        assert end == '2022-01-01'

    @patch('src.data.datetime')
    def test_get_default_date_range_no_end_date(self, mock_dt):
        mock_dt.now.return_value.strftime.return_value = '2023-01-01'
        config = {'data': {'date_range': {'start_date': '2021-01-01', 'end_date': None}}}
        start, end = get_default_date_range(config)
        assert start == '2021-01-01'
        assert end == '2023-01-01'


class TestDataIO:
    @patch('src.data.yf.download')
    def test_download_ticker_success(self, mock_download, sample_dataframe):
        mock_download.return_value = sample_dataframe
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = download_ticker('TEST', '2023-01-01', '2023-01-05', out_dir=tmpdir)
            assert filepath is not None
            assert os.path.exists(filepath)
            df = pd.read_csv(filepath)
            assert len(df) == 5

    @patch('src.data.yf.download')
    def test_download_ticker_empty(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = download_ticker('EMPTY', '2023-01-01', '2023-01-05', out_dir=tmpdir)
            assert filepath is None

    def test_load_and_save_raw_data(self, sample_dataframe):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.csv')
            save_raw_data(sample_dataframe, 'test', tmpdir)
            
            loaded_df = load_raw_data(filepath)
            pd.testing.assert_frame_equal(sample_dataframe, loaded_df)

    def test_load_raw_data_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data('non_existent_file.csv')


class TestDataQuality:
    def test_clean_data_functionality(self):
        dirty_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 102, float('inf')],
            'Low': [98, 99, 100],
            'Close': [102, 100, 103],
            'Volume': [1000, 0, -500]
        })
        cleaned = clean_data(dirty_data)
        assert not cleaned.isin([float('inf')]).any().any()
        assert cleaned['Volume'].iloc[1] == 1000 # ffilled from previous
        assert pd.isna(cleaned['Volume'].iloc[2]) # No ffill for negative

    def test_clean_data_high_low_integrity(self):
        data = pd.DataFrame({'High': [100], 'Low': [101]}) # Invalid
        cleaned = clean_data(data)
        assert cleaned.empty

    def test_validate_data_quality_good_data(self, sample_dataframe):
        result = validate_data_quality(sample_dataframe, detailed=True)
        assert result['overall'] is True
        assert len(result['issues']) == 0

    def test_validate_data_quality_bad_data(self):
        bad_df = pd.DataFrame({
            'High': [100, 90], # High < Low in second row
            'Low': [95, 95],
            'Open': [98, 92],
            'Close': [99, 93],
            'Volume': [1000, -100] # Negative volume
        })
        result = validate_data_quality(bad_df, detailed=True)
        assert result['overall'] is False
        assert len(result['issues']) == 2


class TestDataProcessing:
    def test_detect_outliers_iqr(self):
        s = pd.Series([1, 10, 11, 12, 13, 14, 100])
        outliers = detect_outliers(s, method='iqr')
        assert outliers.tolist() == [True, False, False, False, False, False, True]

    def test_detect_outliers_zscore(self):
        s = pd.Series([1, 10, 11, 12, 13, 14, 100])
        outliers = detect_outliers(s, method='zscore', threshold=2.0)
        assert outliers.tolist() == [False, False, False, False, False, False, True]

    def test_handle_missing_data_ffill(self):
        df = pd.DataFrame({'A': [1, np.nan, 3]})
        filled_df = handle_missing_data(df, method='forward_fill')
        assert filled_df['A'].tolist() == [1.0, 1.0, 3.0]

    def test_calculate_returns_simple(self):
        prices = pd.Series([100, 110, 104.5])
        returns = calculate_returns(prices, method='simple')
        assert pd.isna(returns.iloc[0])
        assert pytest.approx(returns.iloc[1]) == 0.10  # (110-100)/100
        assert pytest.approx(returns.iloc[2]) == -0.05  # (104.5-110)/110

    def test_calculate_returns_log(self):
        prices = pd.Series([100, 110, 104.5])
        returns = calculate_returns(prices, method='log')
        assert pd.isna(returns.iloc[0])
        assert pytest.approx(returns.iloc[1]) == np.log(1.1)
        assert pytest.approx(returns.iloc[2]) == np.log(104.5/110)

    def test_align_timestamps(self):
        df1 = pd.DataFrame({'A': [1, 2]}, index=pd.to_datetime(['2023-01-01', '2023-01-02']))
        df2 = pd.DataFrame({'B': [3, 4]}, index=pd.to_datetime(['2023-01-02', '2023-01-03']))
        df3 = pd.DataFrame({'C': [5, 6]}, index=pd.to_datetime(['2023-01-02', '2023-01-04']))
        
        aligned = align_timestamps([df1, df2, df3])
        
        # The only common index is '2023-01-02'
        assert len(aligned) == 3
        for df in aligned:
            assert len(df) == 1
            assert df.index[0] == pd.to_datetime('2023-01-02')
        
        assert aligned[0]['A'].iloc[0] == 2
        assert aligned[1]['B'].iloc[0] == 3
        assert aligned[2]['C'].iloc[0] == 5


# Run tests: pytest tests/test_data.py -v
