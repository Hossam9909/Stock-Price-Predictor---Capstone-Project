# Tests for models.py module

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import project models and utilities
from src.models import (
    BasePredictor,
    NaiveLastValue,
    RandomWalkDrift,
    RandomForestPredictor,
    LightGBMPredictor,
    XGBPredictor,
    train_models_multi_horizon,
)
import tempfile
import os

lgb_available = True
try:
    import lightgbm
except ImportError:
    lgb_available = False

xgb_available = True
try:
    import xgboost
except ImportError:
    xgb_available = False

@pytest.fixture
def sample_features_and_target():
    """Fixture for sample features (X) and target (y)."""
    X = pd.DataFrame({
        'feature1': np.arange(10),
        'feature2': np.arange(10, 20)
    })
    y = pd.Series(np.arange(100, 110), name="target")
    return X, y


class TestBaselineModels:
    """Test cases for baseline models"""

    def test_naive_last_value_model(self, sample_features_and_target):
        """Test naive last value baseline model"""
        X, y = sample_features_and_target
        model = NaiveLastValue(horizon=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert model.is_fitted
        assert model.last_value == 109
        np.testing.assert_array_equal(preds, np.full(len(X), 109))

    def test_random_walk_drift_model(self, sample_features_and_target):
        """Test Random Walk with Drift baseline model"""
        X, y = sample_features_and_target
        # Add price column to X for drift calculation
        X['Adj Close'] = y
        model = RandomWalkDrift(horizon=5)
        model.fit(X, y)
        preds = model.predict(X)
        assert model.is_fitted
        # Prediction should be a constant value based on last price and drift
        assert len(np.unique(preds)) == 1
        assert preds[0] > 0

    def test_base_predictor_save_load(self, sample_features_and_target):
        """Test model saving and loading via BasePredictor"""
        X, y = sample_features_and_target
        model = RandomForestPredictor(horizon=1, rf_params={'n_estimators': 5})
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            model.save_model(path)
            
            new_model = RandomForestPredictor()
            assert not new_model.is_fitted
            new_model.load_model(path)
            
            assert new_model.is_fitted
            assert new_model.horizon == 1
            assert new_model.feature_names == list(X.columns)
            
            # Check if it can predict
            preds = new_model.predict(X)
            assert len(preds) == len(X)


class TestAdvancedModels:
    """Test cases for advanced ML models"""

    def test_random_forest_model(self, sample_features_and_target):
        """Test RandomForestPredictor training and prediction"""
        X, y = sample_features_and_target
        model = RandomForestPredictor(horizon=1, rf_params={'n_estimators': 5})
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(X)
        importances = model.feature_importances()
        assert len(importances) == X.shape[1]

    @pytest.mark.skipif(not lgb_available, reason="lightgbm not installed")
    def test_lightgbm_model(self, sample_features_and_target):
        """Test LightGBMPredictor training and prediction"""
        X, y = sample_features_and_target
        model = LightGBMPredictor(horizon=1, lgb_params={'n_estimators': 5})
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(X)

    @pytest.mark.skipif(not xgb_available, reason="xgboost not installed")
    def test_xgboost_model(self, sample_features_and_target):
        """Test XGBPredictor training and prediction"""
        X, y = sample_features_and_target
        model = XGBPredictor(horizon=1, xgb_params={'n_estimators': 5})
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestModelTrainingUtils:
    """Test cases for model utility functions"""

    def test_train_models_multi_horizon(self, sample_features_and_target):
        """Test the multi-horizon training utility"""
        X, y = sample_features_and_target
        df = X.copy()
        df['Target_1d'] = y.shift(-1)
        df['Target_3d'] = y.shift(-3)
        
        records = train_models_multi_horizon(
            df=df,
            feature_columns=['feature1', 'feature2'],
            horizons=[1, 3],
            model_type='rf',
            model_params={'n_estimators': 5}
        )
        
        assert 1 in records
        assert 3 in records
        assert records[1].model.is_fitted
        assert records[3].model.is_fitted
        assert records[1].horizon == 1
        assert records[3].horizon == 3
