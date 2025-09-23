"""
src/models.py

Machine Learning Models for Stock Price Prediction
- Integrates with src.data, src.features and src.evaluate (no duplicated feature/validation code)
- Baselines: Naive last-value, RandomWalk (drift)
- Tree ensembles: RandomForest, LightGBM, XGBoost (if installed)
- Optional: simple LSTM (if TensorFlow/Keras installed)
- Model train/predict utilities, hyperparameter tuning, save/load, multi-horizon support
- Uses evaluate.walk_forward_validation for validation integration

Author: Stock Price Predictor Project
"""

from __future__ import annotations

import os
import json
import joblib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Optional libs
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

# Optional Keras / TensorFlow for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import load_model as keras_load_model
    keras_available = True
except Exception:
    keras_available = False

# Project imports (use existing functions)
from src.data import (
    load_raw_data,
    clean_data,
    load_config,
    setup_logging,
    validate_data_quality,
    calculate_returns,
    align_timestamps,
)
from src.features import (
    create_all_features,
    create_targets,
    apply_configured_features,
)
from src.evaluate import (
    walk_forward_validation,
    time_series_cross_validation,
    evaluate_regression_model,
)

warnings.filterwarnings("ignore")
logger = setup_logging()

# -----------------------------------------------------------------------------
# Dataclasses & Utilities
# -----------------------------------------------------------------------------


@dataclass
class ModelRecord:
    """Holds trained model + metadata."""
    name: str
    model: Any
    horizon: int
    feature_columns: List[str]
    trained: bool
    params: Dict[str, Any]


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# -----------------------------------------------------------------------------
# Base Predictor (shared API)
# -----------------------------------------------------------------------------


class BasePredictor:
    """Abstract base predictor class defining the interface."""

    def __init__(self, horizon: int = 1, target_col: str = "Adj Close") -> None:
        self.horizon = horizon
        self.target_col = target_col
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False

    def create_targets(self, df: pd.DataFrame) -> pd.Series:
        """Create target column (future price) for this horizon."""
        return df[self.target_col].shift(-self.horizon)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> "BasePredictor":
        """Fit model. Must be implemented by subclass."""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict. Must be implemented by subclass."""
        raise NotImplementedError

    def save_model(self, path: str) -> None:
        """Persist model to disk (joblib/pickle)."""
        _ensure_dir(path)
        payload = {
            "horizon": self.horizon,
            "target_col": self.target_col,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "model_type": self.__class__.__name__,
        }
        # Save model object separately for reliability
        try:
            joblib.dump({"meta": payload, "model": self.model}, path)
            logger.info(f"Saved model to {path}")
        except Exception:
            # fallback pickle
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"meta": payload, "model": self.model}, f)
            logger.info(f"Saved model (pickle fallback) to {path}")

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        try:
            obj = joblib.load(path)
            self.model = obj.get("model")
            meta = obj.get("meta", {})
            self.horizon = meta.get("horizon", self.horizon)
            self.target_col = meta.get("target_col", self.target_col)
            self.feature_names = meta.get("feature_names", self.feature_names)
            self.is_fitted = meta.get("is_fitted", self.is_fitted)
            logger.info(f"Loaded model from {path}")
        except Exception:
            import pickle
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.model = obj.get("model")
            meta = obj.get("meta", {})
            self.horizon = meta.get("horizon", self.horizon)
            self.target_col = meta.get("target_col", self.target_col)
            self.feature_names = meta.get("feature_names", self.feature_names)
            self.is_fitted = meta.get("is_fitted", self.is_fitted)
            logger.info(f"Loaded model (pickle fallback) from {path}")


# -----------------------------------------------------------------------------
# Baseline models
# -----------------------------------------------------------------------------


class NaiveLastValue(BasePredictor):
    """Predicts last observed target value for all future rows."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "NaiveLastValue":
        y_nonnull = y.dropna()
        if y_nonnull.empty:
            raise ValueError("No non-null targets to fit NaiveLastValue")
        self.last_value = float(y_nonnull.iloc[-1])
        self.is_fitted = True
        logger.info("Fitted NaiveLastValue baseline")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "is_fitted", False):
            raise ValueError("Model not fitted")
        return np.full(len(X), self.last_value, dtype=float)


class RandomWalkDrift(BasePredictor):
    """Random walk with estimated drift from historical returns."""

    def fit(self, X: pd.DataFrame, y: pd.Series, price_col: str = "Adj Close", **kwargs) -> "RandomWalkDrift":
        # compute returns from price column in X or y (prefer X if contains price)
        source = X[price_col] if price_col in X.columns else y
        returns = calculate_returns(pd.Series(source))
        self.drift = float(
            returns.mean()) if not returns.dropna().empty else 0.0
        # last observed value from y
        y_nonnull = y.dropna()
        if y_nonnull.empty:
            raise ValueError("No non-null targets to fit RandomWalkDrift")
        self.last_value = float(y_nonnull.iloc[-1])
        self.is_fitted = True
        logger.info(f"Fitted RandomWalkDrift with drift={self.drift:.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "is_fitted", False):
            raise ValueError("Model not fitted")
        # project last value forward using drift compounded by horizon
        val = float(self.last_value) * \
            ((1.0 + float(self.drift)) ** self.horizon)
        return np.full(len(X), val, dtype=float)


# -----------------------------------------------------------------------------
# Tree-based models
# -----------------------------------------------------------------------------


class RandomForestPredictor(BasePredictor):
    """RandomForestRegressor wrapper."""

    def __init__(self, horizon: int = 1, target_col: str = "Adj Close", rf_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(horizon=horizon, target_col=target_col)
        params = rf_params or {"n_estimators": 200,
                               "n_jobs": -1, "random_state": 42}
        self.model = RandomForestRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> "RandomForestPredictor":
        mask = ~y.isna()
        Xc = X.loc[mask].copy()
        yc = y.loc[mask].copy()
        if Xc.empty:
            raise ValueError("No training rows after dropping NaN")
        self.feature_names = list(Xc.columns)
        self.model.fit(Xc.values, yc.values)
        self.is_fitted = True
        logger.info("Trained RandomForest")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X.values)

    def feature_importances(self) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)


class LightGBMPredictor(BasePredictor):
    """LightGBM wrapper using sklearn API (LGBMRegressor)."""

    def __init__(self, horizon: int = 1, target_col: str = "Adj Close", lgb_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(horizon=horizon, target_col=target_col)
        if lgb is None:
            raise ImportError("lightgbm not installed")
        params = lgb_params or {"n_estimators": 500,
                                "learning_rate": 0.05, "random_state": 42}
        self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> "LightGBMPredictor":
        mask = ~y.isna()
        Xc = X.loc[mask].copy()
        yc = y.loc[mask].copy()
        if Xc.empty:
            raise ValueError("No training rows after dropping NaN")
        self.feature_names = list(Xc.columns)
        self.model.fit(Xc.values, yc.values, eval_set=[
                       (Xc.values, yc.values)], verbose=False)
        self.is_fitted = True
        logger.info("Trained LightGBM")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X.values)


class XGBPredictor(BasePredictor):
    """XGBoost wrapper (XGBRegressor)."""

    def __init__(self, horizon: int = 1, target_col: str = "Adj Close", xgb_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(horizon=horizon, target_col=target_col)
        if xgb is None:
            raise ImportError("xgboost not installed")
        params = xgb_params or {
            "n_estimators": 500, "learning_rate": 0.05, "random_state": 42, "n_jobs": -1}
        self.model = xgb.XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs) -> "XGBPredictor":
        mask = ~y.isna()
        Xc = X.loc[mask].copy()
        yc = y.loc[mask].copy()
        if Xc.empty:
            raise ValueError("No training rows after dropping NaN")
        self.feature_names = list(Xc.columns)
        self.model.fit(Xc.values, yc.values, eval_set=[
                       (Xc.values, yc.values)], verbose=False)
        self.is_fitted = True
        logger.info("Trained XGBoost")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X.values)


# -----------------------------------------------------------------------------
# Optional LSTM
# -----------------------------------------------------------------------------


def build_simple_lstm(input_shape: Tuple[int, int], units: int = 32, dropout: float = 0.1, dense_units: int = 16) -> Any:
    """Build small LSTM model via Keras. Raises if Keras unavailable."""
    if not keras_available:
        raise ImportError("Keras/TensorFlow not available")
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


class LSTMPredictor(BasePredictor):
    """Wrapper around a Keras LSTM model. Expects pre-made sequences for training/prediction."""

    def __init__(self, horizon: int = 1, target_col: str = "Adj Close", params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(horizon=horizon, target_col=target_col)
        if not keras_available:
            raise ImportError("Keras/TensorFlow not available")
        self.params = params or {
            "n_timesteps": 10, "units": 32, "dropout": 0.1, "epochs": 50, "batch_size": 32}
        self.model = None

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray, **fit_kwargs) -> "LSTMPredictor":
        """X_seq shape -> (n_samples, n_timesteps, n_features)"""
        if not keras_available:
            raise ImportError("Keras/TensorFlow not available")
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = build_simple_lstm(input_shape, units=self.params.get(
            "units", 32), dropout=self.params.get("dropout", 0.1))
        es = EarlyStopping(patience=fit_kwargs.get(
            "patience", 5), restore_best_weights=True)
        self.model.fit(X_seq, y_seq, epochs=self.params.get(
            "epochs", 50), batch_size=self.params.get("batch_size", 32), callbacks=[es], verbose=0)
        self.is_fitted = True
        logger.info("Trained LSTM")
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("LSTM not trained")
        preds = self.model.predict(X_seq)
        return preds.reshape(-1)


# -----------------------------------------------------------------------------
# Hyperparameter Tuning Helpers
# -----------------------------------------------------------------------------


def tune_random_forest(X: np.ndarray, y: np.ndarray, param_dist: Optional[Dict[str, Iterable]] = None, n_iter: int = 20, cv: int = 3, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """Randomized search for RandomForest hyperparams. Returns best_estimator, best_params."""
    default_dist = {"n_estimators": [100, 200, 400], "max_depth": [
        None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
    dist = param_dist or default_dist
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rs = RandomizedSearchCV(base, dist, n_iter=n_iter, cv=cv,
                            scoring="neg_mean_squared_error", n_jobs=-1, random_state=random_state)
    rs.fit(X, y)
    logger.info(f"RandomForest tuning done: best_score={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_


def tune_lightgbm(X: np.ndarray, y: np.ndarray, param_dist: Optional[Dict[str, Iterable]] = None, n_iter: int = 20, cv: int = 3, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """Randomized search for LightGBM. Requires lightgbm installed."""
    if lgb is None:
        raise ImportError("lightgbm not installed")
    default_dist = {"n_estimators": [100, 200, 400], "learning_rate": [
        0.01, 0.05, 0.1], "num_leaves": [31, 50, 100]}
    dist = param_dist or default_dist
    base = lgb.LGBMRegressor(random_state=random_state)
    rs = RandomizedSearchCV(base, dist, n_iter=n_iter, cv=cv,
                            scoring="neg_mean_squared_error", n_jobs=-1, random_state=random_state)
    rs.fit(X, y)
    logger.info(f"LightGBM tuning done: best_score={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_


def tune_xgboost(X: np.ndarray, y: np.ndarray, param_dist: Optional[Dict[str, Iterable]] = None, n_iter: int = 20, cv: int = 3, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """Randomized search for XGBoost. Requires xgboost installed."""
    if xgb is None:
        raise ImportError("xgboost not installed")
    default_dist = {"n_estimators": [100, 200, 400], "learning_rate": [
        0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}
    dist = param_dist or default_dist
    base = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
    rs = RandomizedSearchCV(base, dist, n_iter=n_iter, cv=cv,
                            scoring="neg_mean_squared_error", n_jobs=-1, random_state=random_state)
    rs.fit(X, y)
    logger.info(f"XGBoost tuning done: best_score={rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_


# -----------------------------------------------------------------------------
# Multi-horizon training & prediction helpers
# -----------------------------------------------------------------------------


def _prepare_features_if_needed(df: pd.DataFrame, config_path: Optional[str] = None) -> pd.DataFrame:
    """
    If df already contains Target_*d columns assume features are prepared.
    Otherwise call apply_configured_features from src.features using config_path.
    """
    if any(col.startswith("Target_") for col in df.columns):
        return df.copy()
    cfg = config_path or "config/config.yaml"
    try:
        df_prepped = apply_configured_features(df, config_path=cfg)
        return df_prepped
    except Exception as e:
        logger.warning(f"apply_configured_features failed: {e}")
        # fallback to create_all_features with defaults
        df_prepped = create_all_features(df)
        df_prepped = create_targets(df_prepped, horizons=[1, 7, 14, 28])
        return df_prepped


def train_models_multi_horizon(df: pd.DataFrame,
                               feature_columns: List[str],
                               horizons: List[int],
                               model_type: str = "random_forest",
                               model_params: Optional[Dict[int,
                                                           Dict[str, Any]]] = None,
                               save_dir: Optional[str] = None,
                               config_path: Optional[str] = None) -> Dict[int, ModelRecord]:
    """
    Train one model per horizon. model_params can be a dict keyed by horizon or a single param dict.
    Returns dict: horizon -> ModelRecord.
    """
    df_prepped = _prepare_features_if_needed(df, config_path=config_path)
    records: Dict[int, ModelRecord] = {}
    model_params = model_params or {}

    for h in horizons:
        target_col = f"Target_{h}d"
        if target_col not in df_prepped.columns:
            logger.warning(
                f"Target {target_col} not found; skipping horizon {h}")
            continue

        data_h = df_prepped[feature_columns + [target_col]].dropna()
        if data_h.empty:
            logger.warning(f"No rows for horizon {h} after dropna")
            continue

        X = data_h[feature_columns]
        y = data_h[target_col]

        # instantiate model
        params_for_h = model_params.get(h) if isinstance(model_params, dict) and h in model_params else (
            model_params if isinstance(model_params, dict) and not any(isinstance(k, int) for k in model_params.keys()) else {})
        predictor: BasePredictor
        try:
            if model_type.lower() in ("rf", "random_forest"):
                predictor = RandomForestPredictor(
                    horizon=h, target_col=target_col, rf_params=params_for_h)
            elif model_type.lower() in ("lgb", "lightgbm"):
                predictor = LightGBMPredictor(
                    horizon=h, target_col=target_col, lgb_params=params_for_h)
            elif model_type.lower() in ("xgb", "xgboost"):
                predictor = XGBPredictor(
                    horizon=h, target_col=target_col, xgb_params=params_for_h)
            elif model_type.lower() in ("naive",):
                predictor = NaiveLastValue(horizon=h, target_col=target_col)
            elif model_type.lower() in ("rw", "random_walk"):
                predictor = RandomWalkDrift(horizon=h, target_col=target_col)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to instantiate model for horizon {h}: {e}")
            continue

        # Fit
        try:
            predictor.fit(X, y)
            predictor.feature_names = list(X.columns)
            predictor.is_fitted = True
            rec = ModelRecord(name=predictor.__class__.__name__, model=predictor, horizon=h,
                              feature_columns=list(X.columns), trained=True, params=params_for_h or {})
            records[h] = rec
            logger.info(
                f"Trained model for horizon {h}: {rec.name}, rows={len(y)}")
            # optional save
            if save_dir:
                _ensure_dir(save_dir + "/.keep")
                save_path = os.path.join(save_dir, f"{rec.name}_h{h}.joblib")
                predictor.save_model(save_path)
        except Exception as e:
            logger.error(f"Training failed for horizon {h}: {e}")

    return records


def predict_multi_horizon(records: Dict[int, ModelRecord], df: pd.DataFrame, feature_columns: List[str]) -> Dict[int, np.ndarray]:
    """
    Given trained ModelRecords produce predictions aligned to df.index.
    Returns dict horizon -> numpy array (with np.nan where no prediction).
    """
    n = len(df)
    forecasts: Dict[int, np.ndarray] = {}
    for h, rec in records.items():
        preds = np.full(n, np.nan, dtype=float)
        predictor: BasePredictor = rec.model
        # find rows with complete features
        rows = df[feature_columns].dropna()
        if rows.empty:
            forecasts[h] = preds
            continue
        try:
            p = predictor.predict(rows)
            # align values to df index positions of rows
            for i, idx in enumerate(rows.index):
                pos = df.index.get_loc(idx)
                preds[pos] = float(p[i])
        except Exception as e:
            logger.error(f"Prediction error for horizon {h}: {e}")
        forecasts[h] = preds
    return forecasts


# -----------------------------------------------------------------------------
# Integration with evaluate.walk_forward_validation
# -----------------------------------------------------------------------------


def evaluate_with_walk_forward(df: pd.DataFrame,
                               feature_columns: List[str],
                               horizons: List[int],
                               model_type: str = "random_forest",
                               model_params: Optional[Dict[int,
                                                           Dict[str, Any]]] = None,
                               min_train_size: int = 252,
                               step_size: int = 21,
                               config_path: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Run walk-forward validation using src.evaluate.walk_forward_validation for each horizon.
    Returns dict: horizon -> validation result (as produced by evaluate.walk_forward_validation).
    """
    df_prepped = _prepare_features_if_needed(df, config_path=config_path)
    results: Dict[int, Dict[str, Any]] = {}

    for h in horizons:
        model_params_for_h = model_params.get(h) if isinstance(
            model_params, dict) and h in model_params else (model_params or {})
        # build a model instance to pass to the validator; walk_forward_validation expects model with fit/predict
        try:
            if model_type.lower() in ("rf", "random_forest"):
                model_inst = RandomForestPredictor(
                    horizon=h, target_col=f"Target_{h}d", rf_params=model_params_for_h)
            elif model_type.lower() in ("lgb", "lightgbm"):
                model_inst = LightGBMPredictor(
                    horizon=h, target_col=f"Target_{h}d", lgb_params=model_params_for_h)
            elif model_type.lower() in ("xgb", "xgboost"):
                model_inst = XGBPredictor(
                    horizon=h, target_col=f"Target_{h}d", xgb_params=model_params_for_h)
            elif model_type.lower() in ("naive",):
                model_inst = NaiveLastValue(
                    horizon=h, target_col=f"Target_{h}d")
            elif model_type.lower() in ("rw", "random_walk"):
                model_inst = RandomWalkDrift(
                    horizon=h, target_col=f"Target_{h}d")
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            logger.error(
                f"Failed to create model instance for horizon {h}: {e}")
            results[h] = {"error": str(e)}
            continue

        try:
            wf_res = walk_forward_validation(
                data=df_prepped,
                model=model_inst,
                horizons=[h],
                feature_columns=feature_columns,
                target_column_template="Target_{}d",
                min_train_size=min_train_size,
                step_size=step_size,
            )
            # wf_res is dict keyed by horizon
            results[h] = wf_res.get(h, {"error": "no_result"})
            logger.info(f"Walk-forward completed for horizon {h}")
        except Exception as e:
            logger.error(f"Walk-forward validation error for horizon {h}: {e}")
            results[h] = {"error": str(e)}

    return results


# -----------------------------------------------------------------------------
# Utilities: save/load multiple model records metadata
# -----------------------------------------------------------------------------


def save_model_records(records: Dict[int, ModelRecord], path: str) -> None:
    _ensure_dir(path)
    meta = {h: {"name": r.name, "horizon": r.horizon, "trained": r.trained,
                "params": r.params} for h, r in records.items()}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved model records metadata to {path}")


# -----------------------------------------------------------------------------
# CLI wrapper (keeps original CLI convenience but delegates to module functions)
# -----------------------------------------------------------------------------
def cli_main() -> None:
    """Simple CLI entrypoint to train a single model/horizon (keeps compatibility)."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Train stock prediction model (wrapper)")
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--model", choices=["naive", "random_walk", "rf", "lightgbm", "xgboost"], default="rf")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--save", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        import logging as _logging
        _logging.getLogger().setLevel(_logging.DEBUG)

    logger.info(
        f"Starting CLI training for {args.ticker} model={args.model} horizon={args.horizon}")

    # load & prepare data
    data_file = os.path.join(args.data_dir, f"{args.ticker}.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    df_raw = load_raw_data(data_file)
    df_clean = clean_data(df_raw)

    # apply features via project function
    df_fe = _prepare_features_if_needed(df_clean, config_path=args.config)

    # pick features automatically (exclude OHLCV)
    feature_cols = [c for c in df_fe.columns if c not in [
        "Open", "High", "Low", "Close", "Adj Close", "Volume"] and not c.startswith("Target_")]

    # train single-horizon model via train_models_multi_horizon wrapper
    records = train_models_multi_horizon(df_fe, feature_cols, horizons=[
                                         args.horizon], model_type=args.model, save_dir=args.output_dir, config_path=args.config)
    rec = records.get(args.horizon)
    if rec:
        model_path = args.save or os.path.join(
            args.output_dir, f"{rec.name}_{args.horizon}.joblib")
        rec.model.save_model(model_path)
        logger.info(f"Saved trained model to {model_path}")
    else:
        logger.error("No model trained")

    # optionally validate using walk-forward
    if args.validate:
        wf_results = evaluate_with_walk_forward(df_fe, feature_cols, horizons=[
                                                args.horizon], model_type=args.model, config_path=args.config)
        logger.info(f"Validation summary: {wf_results.get(args.horizon)}")

    print("Done.")


# -----------------------------------------------------------------------------
# If executed as script, run CLI wrapper
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cli_main()
