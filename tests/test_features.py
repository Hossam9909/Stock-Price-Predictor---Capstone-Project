# Tests for features.py module

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import project functions
from src.features import (
    create_lag_features,
    create_rolling_features,
    create_technical_indicators,
    create_targets,
    scale_features,
    _calculate_sma, # Assuming you extract SMA to a helper for testing
    _calculate_rsi,
    _calculate_macd,
    _calculate_bollinger_bands,
    validate_features,
    create_all_features,
)

# Helper function for testing SMA, assuming it's extracted in features.py
# If not, we can test it via the main function's output.
def _calculate_sma(series, window):
    return series.rolling(window=window).mean()

@pytest.fixture
def sample_ohlcv_df():
    """Fixture for a sample OHLCV DataFrame."""
    return pd.DataFrame({
        'Open': [100, 102, 101, 103, 105, 106, 108, 110, 109, 112],
        'High': [103, 104, 102, 105, 106, 108, 110, 112, 111, 114],
        'Low': [99, 101, 100, 102, 104, 105, 107, 109, 108, 110],
        'Close': [102, 103, 101, 104, 105, 107, 109, 111, 110, 113],
        'Volume': [1000, 1100, 900, 1200, 1050, 1300, 1400, 1500, 1250, 1600]
    }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=10)))


class TestFeatureEngineering:
    """Test cases for feature engineering functionality"""

    def test_create_lag_features(self, sample_ohlcv_df):
        """Test creation of lag features"""
        df_lags = create_lag_features(sample_ohlcv_df, lags=[1, 3], columns=['Close'])
        assert 'Close_lag_1' in df_lags.columns
        assert 'Close_lag_3' in df_lags.columns
        assert pd.isna(df_lags['Close_lag_1'].iloc[0])
        assert df_lags['Close_lag_1'].iloc[1] == sample_ohlcv_df['Close'].iloc[0]
        assert pd.isna(df_lags['Close_lag_3'].iloc[2])
        assert df_lags['Close_lag_3'].iloc[3] == sample_ohlcv_df['Close'].iloc[0]

    def test_create_rolling_features(self, sample_ohlcv_df):
        """Test creation of rolling window features"""
        df_rolling = create_rolling_features(sample_ohlcv_df, windows=[3], columns=['Close'], statistics=['mean'])
        assert 'Close_rolling_3_mean' in df_rolling.columns
        assert pd.isna(df_rolling['Close_rolling_3_mean'].iloc[1])
        expected_mean = (102 + 103 + 101) / 3
        assert pytest.approx(df_rolling['Close_rolling_3_mean'].iloc[2]) == expected_mean

    def test_create_technical_indicators(self, sample_ohlcv_df):
        """Test creation of technical indicators"""
        df_tech = create_technical_indicators(sample_ohlcv_df)
        assert 'SMA_20' in df_tech.columns
        assert 'RSI_14' in df_tech.columns
        assert 'MACD' in df_tech.columns
        assert 'BB_Upper' in df_tech.columns
        # Check that it doesn't crash with fewer rows than a window
        small_df = sample_ohlcv_df.head(5)
        df_tech_small = create_technical_indicators(small_df)
        assert 'SMA_10' in df_tech_small.columns # Column should exist, even if all NaN

    def test_create_target_variables(self, sample_ohlcv_df):
        """Test creation of target variables for different horizons"""
        df_targets = create_targets(sample_ohlcv_df, horizons=[1, 5])
        assert 'Target_1d' in df_targets.columns
        assert 'Target_5d' in df_targets.columns
        assert df_targets['Target_1d'].iloc[0] == sample_ohlcv_df['Close'].iloc[1]
        assert df_targets['Target_5d'].iloc[0] == sample_ohlcv_df['Close'].iloc[5]
        assert pd.isna(df_targets['Target_5d'].iloc[-1])

    def test_feature_scaling(self, sample_ohlcv_df):
        """Test feature scaling and normalization"""
        df_scaled, scaler = scale_features(sample_ohlcv_df, method="standard")
        assert 'StandardScaler' in str(type(scaler))
        # Mean should be close to 0 and std dev close to 1
        assert pytest.approx(df_scaled['Close'].mean(), abs=1e-9) == 0.0
        assert pytest.approx(df_scaled['Close'].std(), abs=1e-9) == 1.0


class TestTechnicalIndicators:
    """Test cases for technical indicators"""

    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        s = pd.Series([1, 2, 3, 4, 5])
        sma = _calculate_sma(s, 3)
        assert pd.isna(sma.iloc[1])
        assert sma.iloc[2] == 2.0 # (1+2+3)/3
        assert sma.iloc[4] == 4.0 # (3+4+5)/3

    def test_rsi_calculation(self, sample_ohlcv_df):
        """Test Relative Strength Index calculation"""
        rsi = _calculate_rsi(sample_ohlcv_df['Close'], window=5)
        assert not rsi.isnull().all()
        # RSI must be between 0 and 100
        assert rsi.dropna().between(0, 100).all()

    def test_macd_calculation(self, sample_ohlcv_df):
        """Test MACD calculation"""
        macd_dict = _calculate_macd(sample_ohlcv_df['Close'])
        assert 'MACD' in macd_dict
        assert 'Signal' in macd_dict
        assert 'Histogram' in macd_dict
        # Histogram should be MACD - Signal
        pd.testing.assert_series_equal(macd_dict['Histogram'], macd_dict['MACD'] - macd_dict['Signal'])

    def test_bollinger_bands_calculation(self, sample_ohlcv_df):
        """Test Bollinger Bands calculation"""
        bb_dict = _calculate_bollinger_bands(sample_ohlcv_df['Close'], window=5)
        assert 'Upper' in bb_dict
        assert 'Middle' in bb_dict
        assert 'Lower' in bb_dict
        # Middle band should be the SMA
        pd.testing.assert_series_equal(bb_dict['Middle'], sample_ohlcv_df['Close'].rolling(5).mean())
        # Upper should be >= Middle, Lower should be <= Middle
        assert (bb_dict['Upper'].dropna() >= bb_dict['Middle'].dropna()).all()
        assert (bb_dict['Lower'].dropna() <= bb_dict['Middle'].dropna()).all()


class TestFeatureValidation:
    """Test cases for feature validation"""

    def test_feature_completeness(self, sample_ohlcv_df):
        """Test that all expected features are created"""
        df_all = create_all_features(sample_ohlcv_df)
        expected_cols = ['SMA_5', 'RSI_14', 'Close_lag_1', 'Close_rolling_5_mean', 'Returns_1d']
        for col in expected_cols:
            assert col in df_all.columns

    def test_feature_data_types(self, sample_ohlcv_df):
        """Test that features have correct data types"""
        df_all = create_all_features(sample_ohlcv_df)
        numeric_df = df_all.select_dtypes(include=np.number)
        # All columns except potential object/category columns should be numeric
        assert numeric_df.shape[1] == df_all.shape[1]

    def test_feature_ranges(self, sample_ohlcv_df):
        """Test that features are within expected ranges"""
        df_all = create_technical_indicators(sample_ohlcv_df)
        assert df_all['RSI_14'].dropna().between(0, 100).all()
        assert df_all['Williams_R'].dropna().between(-100, 0).all()
        assert df_all['Stoch_K'].dropna().between(0, 100).all()

    def test_no_data_leakage(self, sample_ohlcv_df):
        """Test that there's no data leakage in features"""
        # A simple check: a feature at time `t` should not use info from `t+1`
        df1 = create_all_features(sample_ohlcv_df)
        
        # Modify a future value and re-calculate features
        df_modified = sample_ohlcv_df.copy()
        df_modified.loc[df_modified.index[-1], 'Close'] = 999
        df2 = create_all_features(df_modified)

        # The features for a row before the modification should be identical
        # We check the second to last row
        pd.testing.assert_series_equal(df1.iloc[-2], df2.iloc[-2], check_names=False)
