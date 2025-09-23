"""
Stock Price Prediction Models Module

This module provides comprehensive functionality for training, evaluating, and
predicting stock prices using various machine learning models. It includes
baseline models, advanced ensemble methods, and proper time series validation.

Usage:
    python src/models.py --ticker AAPL --horizon 7 --model lightgbm
"""

import argparse
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import joblib

# Import existing data utilities - DO NOT recreate these functions
from src.data import (
    load_raw_data, clean_data, load_config, setup_logging,
    validate_data_quality, calculate_returns, align_timestamps
)


class BasePredictor:
    """Base class for all stock price predictors."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close'):
        """
        Initialize base predictor.

        Args:
            horizon: Prediction horizon in days
            target_col: Target column to predict
        """
        self.horizon = horizon
        self.target_col = target_col
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.scaler = None

    def create_targets(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable for given horizon.

        Args:
            df: Input DataFrame with price data

        Returns:
            Series with future prices (shifted backward)
        """
        return df[self.target_col].shift(-self.horizon)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BasePredictor':
        """Fit the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")

    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'horizon': self.horizon,
            'target_col': self.target_col,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)

    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.horizon = model_data['horizon']
        self.target_col = model_data['target_col']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']


class NaivePredictor(BasePredictor):
    """Naive predictor that uses last known value."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close'):
        super().__init__(horizon, target_col)
        self.last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaivePredictor':
        """
        Fit naive predictor (just stores last value).

        Args:
            X: Feature DataFrame (not used for naive prediction)
            y: Target series

        Returns:
            Self for method chaining
        """
        self.last_value = y.dropna().iloc[-1]
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using last known value.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (all equal to last value)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return np.full(len(X), self.last_value)


class RandomWalkPredictor(BasePredictor):
    """Random walk predictor with drift."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close'):
        super().__init__(horizon, target_col)
        self.drift = 0
        self.last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomWalkPredictor':
        """
        Fit random walk model by calculating average drift.

        Args:
            X: Feature DataFrame (contains price data)
            y: Target series

        Returns:
            Self for method chaining
        """
        # Calculate daily returns using existing utility
        if self.target_col in X.columns:
            returns = calculate_returns(X[self.target_col], method='simple')
            self.drift = returns.mean()
        else:
            self.drift = 0

        self.last_value = y.dropna().iloc[-1]
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using random walk with drift."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Predict: current_price * (1 + drift)^horizon
        predictions = self.last_value * ((1 + self.drift) ** self.horizon)
        return np.full(len(X), predictions)


class RFPredictor(BasePredictor):
    """Random Forest predictor for stock prices."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close', **rf_params):
        super().__init__(horizon, target_col)

        # Default Random Forest parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(rf_params)

        self.model = RandomForestRegressor(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RFPredictor':
        """
        Fit Random Forest model.

        Args:
            X: Feature DataFrame
            y: Target series

        Returns:
            Self for method chaining
        """
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        # Store feature names
        self.feature_names = list(X_clean.columns)

        # Fit model
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance_dict = dict(
            zip(self.feature_names, self.model.feature_importances_))
        return pd.Series(importance_dict).sort_values(ascending=False)


class LGBMPredictor(BasePredictor):
    """LightGBM predictor for stock prices."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close', **lgb_params):
        super().__init__(horizon, target_col)

        # Default LightGBM parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        default_params.update(lgb_params)

        self.params = default_params
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            num_boost_round: int = 1000, early_stopping_rounds: int = 50) -> 'LGBMPredictor':
        """
        Fit LightGBM model with early stopping.

        Args:
            X: Feature DataFrame
            y: Target series
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Self for method chaining
        """
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        # Store feature names
        self.feature_names = list(X_clean.columns)

        # Create train/validation split for early stopping
        split_idx = int(len(X_clean) * 0.8)

        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[valid_data],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance_dict = dict(
            zip(self.feature_names, self.model.feature_importance()))
        return pd.Series(importance_dict).sort_values(ascending=False)


class XGBPredictor(BasePredictor):
    """XGBoost predictor for stock prices."""

    def __init__(self, horizon: int = 1, target_col: str = 'Adj Close', **xgb_params):
        super().__init__(horizon, target_col)

        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(xgb_params)

        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            early_stopping_rounds: int = 50, eval_metric: str = 'rmse') -> 'XGBPredictor':
        """
        Fit XGBoost model with early stopping.

        Args:
            X: Feature DataFrame
            y: Target series
            early_stopping_rounds: Early stopping patience
            eval_metric: Evaluation metric for early stopping

        Returns:
            Self for method chaining
        """
        # Remove rows where target is NaN
        valid_mask = ~y.isna()
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()

        # Store feature names
        self.feature_names = list(X_clean.columns)

        # Create train/validation split
        split_idx = int(len(X_clean) * 0.8)

        X_train, X_val = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
        y_train, y_val = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

        # Fit model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
            verbose=False
        )

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")

        importance_dict = dict(
            zip(self.feature_names, self.model.feature_importances_))
        return pd.Series(importance_dict).sort_values(ascending=False)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic features for modeling.
    Uses existing data utilities for calculations.

    Args:
        df: Input DataFrame with OHLCV data

    Returns:
        DataFrame with additional features
    """
    df_features = df.copy()

    # Price-based features
    df_features['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    df_features['Price_Change'] = df['Close'].pct_change()
    df_features['Volume_Change'] = df['Volume'].pct_change()

    # Moving averages
    for window in [5, 10, 20]:
        df_features[f'SMA_{window}'] = df['Close'].rolling(
            window=window).mean()
        df_features[f'Close_SMA_Ratio_{window}'] = df['Close'] / \
            df_features[f'SMA_{window}']

    # Volatility features
    for window in [5, 10, 20]:
        returns = calculate_returns(df['Close'], method='simple')
        df_features[f'Volatility_{window}'] = returns.rolling(
            window=window).std()

    # Volume features
    for window in [5, 10]:
        df_features[f'Volume_SMA_{window}'] = df['Volume'].rolling(
            window=window).mean()
        df_features[f'Volume_Ratio_{window}'] = df['Volume'] / \
            df_features[f'Volume_SMA_{window}']

    # Lag features
    for lag in [1, 2, 3, 5]:
        df_features[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df_features[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        df_features[f'Return_Lag_{lag}'] = calculate_returns(
            df['Close']).shift(lag)

    return df_features


def walk_forward_validation(df: pd.DataFrame, model_class, model_params: dict,
                            horizon: int, initial_train_size: float = 0.7,
                            step_size: int = 30) -> Dict[str, Any]:
    """
    Perform walk-forward validation for time series.

    Args:
        df: Input DataFrame with features
        model_class: Model class to instantiate
        model_params: Parameters for model
        horizon: Prediction horizon
        initial_train_size: Initial training size as fraction
        step_size: Number of days to step forward

    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger(__name__)

    # Create features and target
    df_features = create_features(df)
    target = df_features['Adj Close'].shift(-horizon)

    # Remove non-feature columns for X
    feature_cols = [col for col in df_features.columns
                    if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    X = df_features[feature_cols].copy()

    # Remove rows with NaN values
    valid_mask = ~(X.isna().any(axis=1) | target.isna())
    X_clean = X[valid_mask].copy()
    y_clean = target[valid_mask].copy()

    if len(X_clean) == 0:
        logger.error("No valid data after cleaning")
        return {'error': 'No valid data'}

    # Initialize walk-forward validation
    initial_size = int(len(X_clean) * initial_train_size)
    predictions = []
    actuals = []
    dates = []

    current_start = 0

    while current_start + initial_size + step_size < len(X_clean):
        # Define train and test sets
        train_end = current_start + initial_size
        test_start = train_end
        test_end = test_start + step_size

        X_train = X_clean.iloc[current_start:train_end]
        y_train = y_clean.iloc[current_start:train_end]
        X_test = X_clean.iloc[test_start:test_end]
        y_test = y_clean.iloc[test_start:test_end]

        # Train model
        model = model_class(horizon=horizon, **model_params)
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            predictions.extend(pred)
            actuals.extend(y_test.values)
            dates.extend(y_clean.index[test_start:test_end])

        except Exception as e:
            logger.warning(
                f"Training failed for period {current_start}-{train_end}: {e}")

        # Move forward
        current_start += step_size

    if not predictions:
        return {'error': 'No successful predictions'}

    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    # Directional accuracy
    actual_direction = np.sign(np.diff(actuals))
    pred_direction = np.sign(np.diff(predictions))
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    results = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'n_predictions': len(predictions),
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates
    }

    logger.info(
        f"Walk-forward validation completed: RMSE={rmse:.4f}, MAE={mae:.4f}")

    return results


def train_model(ticker: str, model_type: str, horizon: int,
                data_dir: str = 'data/raw', config_path: str = 'config/config.yaml') -> BasePredictor:
    """
    Train a model for a specific ticker and horizon.

    Args:
        ticker: Stock ticker symbol
        model_type: Type of model ('naive', 'random_walk', 'rf', 'lightgbm', 'xgboost')
        horizon: Prediction horizon in days
        data_dir: Directory containing raw data
        config_path: Path to configuration file

    Returns:
        Trained model instance
    """
    logger = logging.getLogger(__name__)

    # Load configuration and data using existing functions
    config = load_config(config_path)

    # Load and clean data using existing functions
    filepath = os.path.join(data_dir, f'{ticker}.csv')
    df_raw = load_raw_data(filepath)
    df_clean = clean_data(df_raw)

    # Validate data quality using existing function
    if not validate_data_quality(df_clean):
        logger.warning(f"Data quality issues detected for {ticker}")

    # Create features
    df_features = create_features(df_clean)

    # Prepare target variable
    target = df_features['Adj Close'].shift(-horizon)

    # Select features (exclude OHLCV columns)
    feature_cols = [col for col in df_features.columns
                    if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    X = df_features[feature_cols].copy()

    # Remove rows with NaN values
    valid_mask = ~(X.isna().any(axis=1) | target.isna())
    X_clean = X[valid_mask].copy()
    y_clean = target[valid_mask].copy()

    # Get model parameters from config
    model_config = config.get('models', {})

    # Initialize and train model
    if model_type == 'naive':
        model = NaivePredictor(horizon=horizon)
    elif model_type == 'random_walk':
        model = RandomWalkPredictor(horizon=horizon)
    elif model_type == 'rf':
        rf_params = model_config.get('baseline', {}).get('random_forest', {})
        model = RFPredictor(horizon=horizon, **rf_params)
    elif model_type == 'lightgbm':
        lgb_params = model_config.get('advanced', {}).get('lightgbm', {})
        model = LGBMPredictor(horizon=horizon, **lgb_params)
    elif model_type == 'xgboost':
        xgb_params = model_config.get('advanced', {}).get('xgboost', {})
        model = XGBPredictor(horizon=horizon, **xgb_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    logger.info(
        f"Training {model_type} model for {ticker} (horizon={horizon})")
    model.fit(X_clean, y_clean)

    logger.info(f"Model training completed successfully")
    return model


def main():
    """Main function to handle CLI arguments and execute model training."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Train stock prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ticker',
        required=True,
        help='Stock ticker symbol'
    )
    parser.add_argument(
        '--model',
        choices=['naive', 'random_walk', 'rf', 'lightgbm', 'xgboost'],
        default='rf',
        help='Model type to train'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=1,
        help='Prediction horizon in days'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory containing raw data'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run walk-forward validation'
    )
    parser.add_argument(
        '--save',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--output-dir',
        default='models',
        help='Directory to save models (used if --save not specified)'
    )
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Display feature importance for tree-based models'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions on the latest available data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        logger.info(f"Starting model training for {args.ticker}")
        logger.info(f"Model: {args.model}, Horizon: {args.horizon} days")

        # Validate input arguments
        if args.horizon <= 0:
            raise ValueError("Horizon must be positive")

        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {args.data_dir}")

        # Check if data file exists
        data_file = os.path.join(args.data_dir, f'{args.ticker}.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Train model
        model = train_model(
            ticker=args.ticker,
            model_type=args.model,
            horizon=args.horizon,
            data_dir=args.data_dir,
            config_path=args.config
        )

        # Determine save path
        save_path = args.save
        if not save_path:
            # Auto-generate save path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{args.ticker}_{args.model}_h{args.horizon}_{timestamp}.joblib"
            save_path = os.path.join(args.output_dir, filename)

        # Save model
        try:
            model.save_model(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

        # Display feature importance if requested and available
        if args.feature_importance and hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                print(
                    f"\nTop 10 Feature Importance for {args.ticker} ({args.model}):")
                print("-" * 50)
                for feature, score in importance.head(10).items():
                    print(f"{feature:30} {score:10.6f}")
            except Exception as e:
                logger.warning(f"Could not display feature importance: {e}")

        # Make predictions if requested
        if args.predict:
            try:
                logger.info("Making predictions on latest data...")

                # Load and prepare latest data
                filepath = os.path.join(args.data_dir, f'{args.ticker}.csv')
                df_raw = load_raw_data(filepath)
                df_clean = clean_data(df_raw)
                df_features = create_features(df_clean)

                # Get latest features
                feature_cols = [col for col in df_features.columns
                                if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                latest_features = df_features[feature_cols].iloc[-1:].copy()

                # Remove any NaN values by forward filling
                latest_features = latest_features.fillna(
                    method='ffill').fillna(0)

                # Make prediction
                prediction = model.predict(latest_features)
                current_price = df_clean['Adj Close'].iloc[-1]

                print(f"\nPrediction Results for {args.ticker}:")
                print("-" * 40)
                print(f"Current Price: ${current_price:.2f}")
                print(
                    f"Predicted Price ({args.horizon} days): ${prediction[0]:.2f}")
                print(
                    f"Expected Change: {((prediction[0] - current_price) / current_price * 100):+.2f}%")

            except Exception as e:
                logger.error(f"Prediction failed: {e}")

        # Run validation if requested
        if args.validate:
            logger.info("Running walk-forward validation...")

            try:
                # Load data for validation
                filepath = os.path.join(args.data_dir, f'{args.ticker}.csv')
                df_raw = load_raw_data(filepath)
                df_clean = clean_data(df_raw)

                # Get model class and parameters
                config = load_config(args.config)
                model_config = config.get('models', {})

                model_classes = {
                    'naive': NaivePredictor,
                    'random_walk': RandomWalkPredictor,
                    'rf': RFPredictor,
                    'lightgbm': LGBMPredictor,
                    'xgboost': XGBPredictor
                }

                # Get model-specific parameters
                model_params = {}
                if args.model == 'rf':
                    model_params = model_config.get(
                        'baseline', {}).get('random_forest', {})
                elif args.model == 'lightgbm':
                    model_params = model_config.get(
                        'advanced', {}).get('lightgbm', {})
                elif args.model == 'xgboost':
                    model_params = model_config.get(
                        'advanced', {}).get('xgboost', {})

                results = walk_forward_validation(
                    df=df_clean,
                    model_class=model_classes[args.model],
                    model_params=model_params,
                    horizon=args.horizon
                )

                if 'error' not in results:
                    print(
                        f"\nValidation Results for {args.ticker} ({args.model}, horizon={args.horizon}):")
                    print("=" * 60)
                    print(
                        f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
                    print(
                        f"Mean Absolute Error (MAE):     {results['mae']:.4f}")
                    print(
                        f"Mean Absolute Percentage Error: {results['mape']:.2f}%")
                    print(
                        f"Directional Accuracy:          {results['directional_accuracy']:.2f}%")
                    print(
                        f"Number of Predictions:         {results['n_predictions']}")

                    # Performance interpretation
                    print(f"\nPerformance Interpretation:")
                    print("-" * 30)
                    if results['mape'] < 5:
                        print("📈 Excellent prediction accuracy")
                    elif results['mape'] < 10:
                        print("✅ Good prediction accuracy")
                    elif results['mape'] < 20:
                        print("⚠️  Moderate prediction accuracy")
                    else:
                        print(
                            "❌ Poor prediction accuracy - consider different model/features")

                    if results['directional_accuracy'] > 55:
                        print("📊 Good directional prediction capability")
                    else:
                        print("📉 Limited directional prediction capability")

                else:
                    logger.error(f"Validation failed: {results['error']}")

            except Exception as e:
                logger.error(f"Validation error: {e}")

        # Summary
        print(f"\n{'='*60}")
        print(f"MODEL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Ticker:          {args.ticker}")
        print(f"Model:           {args.model}")
        print(f"Horizon:         {args.horizon} days")
        print(f"Model saved to:  {save_path}")
        print(f"Training Status: ✅ COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")

        logger.info("Model training pipeline completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⚠️  Training was interrupted by user")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        print("Please check that the data file exists and the path is correct.")

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        print(f"❌ Error: {e}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
        print("Check the logs for more detailed error information.")

        # Additional debugging info if verbose
        if args.verbose:
            import traceback
            print(f"\nDetailed error trace:")
            traceback.print_exc()


if __name__ == '__main__':
    main()
