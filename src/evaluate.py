"""
Model Evaluation Module - USES existing data.py functions

This module provides comprehensive evaluation metrics and validation strategies 
specifically designed for stock price prediction models and time-series forecasting.

Author: Stock Price Predictor Project
Dependencies: src.data (for config + logging)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# Import project utilities
from src.data import load_raw_data
from src.utils import load_config, setup_logging, plot_predictions_vs_actual, plot_error_distribution

warnings.filterwarnings("ignore")

# Initialize logger using project's logging setup
logger = setup_logging()


# -----------------------------------------------------------------------------
# BASIC METRICS
# -----------------------------------------------------------------------------


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    Returns percentage (e.g., 2.5 means 2.5%).
    """
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100.0)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    Returns percentage.
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom_safe = np.where(denom < epsilon, epsilon, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom_safe) * 100.0)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def calculate_within_tolerance(y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.05, epsilon: float = 1e-8) -> float:
    """
    Hit-rate within ±tolerance fraction (e.g., 0.05 = ±5%).
    Returns percentage of predictions within tolerance.
    """
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    within = np.abs(y_true - y_pred) / denom <= tolerance
    return float(np.mean(within) * 100.0)


# -----------------------------------------------------------------------------
# DIRECTIONAL & TRADING METRICS
# -----------------------------------------------------------------------------


def calculate_directional_accuracy(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prev: Optional[np.ndarray] = None) -> float:
    """
    Directional accuracy: percent of times the predicted direction matches the true direction.

    - If y_prev provided (same length as y_true), directions are computed as sign(y - y_prev).
      This is useful when y_true/y_pred are price levels and you want to compare movement from previous price.
    - If y_prev None and len(y_true) >= 2, directions are computed as sign(diff(y_true)) and sign(diff(y_pred)),
      and the first observation is dropped to align lengths.
    - If arrays are already returns (signed), function works by sign(y_true) vs sign(y_pred).

    Returns percentage.
    """
    if y_prev is not None:
        if len(y_prev) != len(y_true):
            raise ValueError(
                "y_prev must have the same length as y_true if provided.")
        true_dir = np.sign(y_true - y_prev)
        pred_dir = np.sign(y_pred - y_prev)
    else:
        # If arrays look like returns (small values), use sign directly
        if np.all(np.abs(y_true) < 1.0) and np.all(np.abs(y_pred) < 1.0):
            true_dir = np.sign(y_true)
            pred_dir = np.sign(y_pred)
        else:
            # Use differences with shift
            if len(y_true) < 2:
                return float("nan")
            true_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))

    # Align lengths
    if len(true_dir) != len(pred_dir):
        minlen = min(len(true_dir), len(pred_dir))
        true_dir = true_dir[:minlen]
        pred_dir = pred_dir[:minlen]

    correct = np.sum(true_dir == pred_dir)
    return float(correct / len(true_dir) * 100.0) if len(true_dir) > 0 else float("nan")


def calculate_profit_loss(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          initial_capital: float = 10000.0,
                          transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    Simulate a simple trading strategy that takes position = sign(y_pred).
    y_true and y_pred should represent returns (or price changes). If they are prices,
    pass differences externally.

    Returns dictionary with basic P&L statistics.
    """
    # Interpret inputs: if values are price levels, convert to returns by diff
    # (Assume user passed returns if length matches)
    positions = np.sign(y_pred)  # -1 short, 0 flat, +1 long
    returns = positions * y_true

    # transaction costs based on changes in positions
    position_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    trade_costs = position_changes * transaction_cost
    # align returns and trade_costs (trade costs apply when changing positions before that return)
    net_returns = returns.copy()
    if len(net_returns) > 0:
        net_returns[1:] = net_returns[1:] - trade_costs[1:]
    # cumulative
    if len(net_returns) == 0:
        return {
            "total_return": 0.0,
            "final_capital": float(initial_capital),
            "total_trades": 0,
            "avg_return_per_trade": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
        }

    cumulative = np.cumprod(1 + net_returns)
    final_capital = initial_capital * cumulative[-1]
    total_trades = int(np.sum(position_changes > 0))
    avg_return_per_trade = float(np.mean(net_returns) * 100.0)
    win_rate = float(np.sum(net_returns > 0) / len(net_returns) * 100.0)
    sharpe = float(np.mean(net_returns) / np.std(net_returns)
                   ) if np.std(net_returns) > 0 else 0.0

    return {
        "total_return": float((final_capital - initial_capital) / initial_capital * 100.0),
        "final_capital": float(final_capital),
        "total_trades": total_trades,
        "avg_return_per_trade": avg_return_per_trade,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
    }


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a return series (percent)."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown) * 100.0)


# -----------------------------------------------------------------------------
# COMPREHENSIVE EVALUATION WRAPPERS
# -----------------------------------------------------------------------------


def evaluate_regression_model(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate a single model's predictions with a variety of metrics.
    Supports price-level inputs or returns. The caller should pass appropriate arrays.
    """
    metrics: Dict[str, float] = {
        "model_name": model_name,
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "smape": calculate_smape(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        # Directional accuracy: attempt to infer direction sensibly
        "directional_accuracy": calculate_directional_accuracy(y_true, y_pred),
        # ±5% hit rate
        "hit_rate_5pct": calculate_within_tolerance(y_true, y_pred, tolerance=0.05),
    }
    logger.info(
        f"Evaluated regression metrics for {model_name}: RMSE={metrics['rmse']:.4f}")
    return metrics


def evaluate_trading_strategy(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              initial_capital: float = 10000.0,
                              transaction_cost: float = 0.001) -> Dict[str, float]:
    """Wrapper that evaluates trading P&L metrics for predicted signals."""
    res = calculate_profit_loss(
        y_true, y_pred, initial_capital, transaction_cost)
    logger.info(
        f"Evaluated trading strategy: total_return={res['total_return']:.2f}%")
    return res


# -----------------------------------------------------------------------------
# VALIDATION STRATEGIES (WALK-FORWARD, EXPANDING, ROLLING)
# -----------------------------------------------------------------------------
# All validation functions expect:
# - data: pd.DataFrame indexed by datetime, containing feature columns and target columns
# - feature_columns: list of column names for X
# - target_col: name of the target column (e.g., 'Target_7d' or 'Returns_1d')
# -----------------------------------------------------------------------------

def _ensure_sorted_by_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy sorted by index (date)."""
    return df.sort_index().copy()


def walk_forward_validation(data: pd.DataFrame,
                            model: Any,
                            horizons: List[int],
                            feature_columns: List[str],
                            target_column_template: str,
                            min_train_size: int = 252,
                            step_size: int = 21,
                            retrain: bool = True) -> Dict[int, Dict[str, Any]]:
    """
    Walk-forward validation for multiple horizons.

    Args:
        data: DataFrame indexed by date containing feature columns and multiple target columns named by template.
              target_column_template should be a format string with one placeholder for horizon, e.g. "Target_{}d".
        model: estimator with fit(X, y) and predict(X)
        horizons: list of integer horizons to evaluate (days)
        feature_columns: columns used as features
        target_column_template: e.g., "Target_{}d" -> will be formatted with horizon
        min_train_size: minimum training window (rows)
        step_size: how many rows to advance each iteration
        retrain: if True retrain model in every iteration (common). If False reuse fitted model.

    Returns:
        dict keyed by horizon -> dict with metrics, trading_metrics, predictions_df, n_predictions
    """
    logger.info("Starting walk-forward validation")
    data_sorted = _ensure_sorted_by_index(data)

    results: Dict[int, Dict[str, Any]] = {}
    max_horizon = max(horizons)

    n = len(data_sorted)
    start_idx = min_train_size

    # arrays for features and index
    X_all = data_sorted[feature_columns].values
    dates = data_sorted.index.to_list()

    # iterate over horizons and collect predictions
    horizon_storage: Dict[int, Dict[str, List]] = {
        h: {"dates": [], "actuals": [], "preds": []} for h in horizons}

    while start_idx + max_horizon <= n - 1:
        # training window indices
        X_train = X_all[:start_idx]
        # For training target we need to use the same target column for one chosen horizon for fitting.
        # We'll fit on the shortest horizon target by default (1d) unless user passes a target for each horizon separately.
        # To be robust, fit separate models per horizon here (common in multi-horizon setups).
        for h in horizons:
            try:
                # prepare train target column name
                t_col = target_column_template.format(h)
                # ensure column exists
                if t_col not in data_sorted.columns:
                    continue

                y_train = data_sorted[t_col].values[:start_idx]

                # single-row test feature / target (at index start_idx + h - 1)
                test_idx = start_idx + h - 1
                if test_idx >= n:
                    continue

                X_test = X_all[test_idx:test_idx + 1]
                y_test = data_sorted[t_col].values[test_idx:test_idx + 1]
                date_test = dates[test_idx]

                # Fit a fresh model per horizon (safer)
                if retrain:
                    try:
                        model.fit(X_all[:start_idx],
                                  data_sorted[t_col].values[:start_idx])
                    except Exception:
                        # If fitting on horizon-specific target fails (shapes), try fallback with 1d target
                        # This is defensive; log and continue
                        logger.warning(
                            f"Model fit failed on horizon {h} at idx {start_idx}")
                        continue

                y_pred = model.predict(X_test)
                # store
                horizon_storage[h]["dates"].append(date_test)
                horizon_storage[h]["actuals"].append(float(y_test[0]))
                horizon_storage[h]["preds"].append(float(y_pred[0]))
            except Exception as e:
                logger.warning(
                    f"Walk-forward iteration failed for horizon {h} at idx {start_idx}: {e}")

        start_idx += step_size

    # Build results dict per horizon
    for h in horizons:
        preds = np.array(horizon_storage[h]["preds"])
        actuals = np.array(horizon_storage[h]["actuals"])
        dates_h = horizon_storage[h]["dates"]

        if len(preds) == 0:
            results[h] = {"n_predictions": 0,
                          "error": "No predictions for horizon"}
            continue

        metrics = evaluate_regression_model(
            actuals, preds, model_name=f"WalkForward_h{h}")
        trading = evaluate_trading_strategy(actuals, preds)
        preds_df = pd.DataFrame(
            {"date": dates_h, "actual": actuals, "predicted": preds})
        results[h] = {
            "n_predictions": len(preds),
            "metrics": metrics,
            "trading_metrics": trading,
            "predictions_df": preds_df,
        }
        logger.info(
            f"Walk-forward horizon {h}: n={len(preds)}, RMSE={metrics['rmse']:.4f}")

    return results


def expanding_window_validation(data: pd.DataFrame,
                                model: Any,
                                feature_columns: List[str],
                                target_column: str,
                                min_train_size: int = 252,
                                test_size: int = 21,
                                retrain: bool = True) -> Dict[str, Any]:
    """
    Expanding window validation (anchored at start, grow training window).

    Returns aggregated metrics and predictions DataFrame.
    """
    logger.info("Starting expanding-window validation")
    data_sorted = _ensure_sorted_by_index(data)
    n = len(data_sorted)

    X_all = data_sorted[feature_columns].values
    dates = data_sorted.index.to_list()

    predictions: List[float] = []
    actuals: List[float] = []
    pred_dates: List[pd.Timestamp] = []

    train_end = min_train_size
    while train_end + test_size <= n:
        X_train = X_all[:train_end]
        y_train = data_sorted[target_column].values[:train_end]

        X_test = X_all[train_end:train_end + test_size]
        y_test = data_sorted[target_column].values[train_end:train_end + test_size]
        dates_test = dates[train_end:train_end + test_size]

        try:
            if retrain:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.extend([float(x) for x in y_pred])
            actuals.extend([float(x) for x in y_test])
            pred_dates.extend(dates_test)
            logger.debug(
                f"Expanding window: train_end={train_end}, produced {len(y_pred)} preds")
        except Exception as e:
            logger.warning(
                f"Expanding window failed at train_end {train_end}: {e}")

        train_end += test_size

    if len(predictions) == 0:
        return {"error": "No predictions generated"}

    preds = np.array(predictions)
    acts = np.array(actuals)
    metrics = evaluate_regression_model(
        acts, preds, model_name="ExpandingWindow")
    trading = evaluate_trading_strategy(acts, preds)
    preds_df = pd.DataFrame(
        {"date": pred_dates, "actual": acts, "predicted": preds})
    logger.info(
        f"Expanding validation produced {len(preds)} predictions, RMSE={metrics['rmse']:.4f}")

    return {"n_predictions": len(preds), "metrics": metrics, "trading_metrics": trading, "predictions_df": preds_df}


def rolling_window_validation(data: pd.DataFrame,
                              model: Any,
                              feature_columns: List[str],
                              target_column: str,
                              train_size: int = 252,
                              test_size: int = 21,
                              step_size: int = 21,
                              retrain: bool = True) -> Dict[str, Any]:
    """
    Rolling (sliding) window validation.

    Returns aggregated metrics and DataFrame of predictions.
    """
    logger.info("Starting rolling-window validation")
    data_sorted = _ensure_sorted_by_index(data)
    n = len(data_sorted)

    X_all = data_sorted[feature_columns].values
    dates = data_sorted.index.to_list()

    preds_list: List[float] = []
    acts_list: List[float] = []
    pred_dates: List[pd.Timestamp] = []

    start_idx = 0
    while start_idx + train_size + test_size <= n:
        X_train = X_all[start_idx:start_idx + train_size]
        y_train = data_sorted[target_column].values[start_idx:start_idx + train_size]

        X_test = X_all[start_idx +
                       train_size:start_idx + train_size + test_size]
        y_test = data_sorted[target_column].values[start_idx +
                                                   train_size:start_idx + train_size + test_size]
        dates_test = dates[start_idx +
                           train_size:start_idx + train_size + test_size]

        try:
            if retrain:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds_list.extend([float(x) for x in y_pred])
            acts_list.extend([float(x) for x in y_test])
            pred_dates.extend(dates_test)
            logger.debug(
                f"Rolling window start={start_idx} -> preds={len(y_pred)}")
        except Exception as e:
            logger.warning(f"Rolling window failed at start {start_idx}: {e}")

        start_idx += step_size

    if len(preds_list) == 0:
        return {"error": "No predictions generated"}

    preds = np.array(preds_list)
    acts = np.array(acts_list)
    metrics = evaluate_regression_model(
        acts, preds, model_name="RollingWindow")
    trading = evaluate_trading_strategy(acts, preds)
    preds_df = pd.DataFrame(
        {"date": pred_dates, "actual": acts, "predicted": preds})
    logger.info(
        f"Rolling validation produced {len(preds)} predictions, RMSE={metrics['rmse']:.4f}")

    return {"n_predictions": len(preds), "metrics": metrics, "trading_metrics": trading, "predictions_df": preds_df}


# -----------------------------------------------------------------------------
# TIME-SERIES CROSS-VALIDATION (TimeSeriesSplit)
# -----------------------------------------------------------------------------


def time_series_cross_validation(model: Any,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 n_splits: int = 5,
                                 test_size: Optional[int] = None,
                                 scoring: str = "rmse") -> Dict[str, Any]:
    """
    Perform time-series aware cross-validation using sklearn TimeSeriesSplit.

    Args:
        model: estimator with fit/predict
        X: full features array
        y: full target array
        n_splits: number of folds
        test_size: optional test window size for each split (if provided)
        scoring: which metric to collect ('rmse','mae','mape')

    Returns:
        dict with per-fold metrics and aggregated results
    """
    logger.info("Starting time-series cross-validation")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []
    fold = 0

    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = calculate_rmse(y_test, y_pred)
            mae = calculate_mae(y_test, y_pred)
            mape = calculate_mape(y_test, y_pred)
            fold_metrics.append(
                {"fold": fold, "rmse": rmse, "mae": mae, "mape": mape})
            logger.debug(f"CV fold {fold}: rmse={rmse:.4f}, mae={mae:.4f}")
        except Exception as e:
            logger.warning(f"CV fold {fold} failed: {e}")

    if len(fold_metrics) == 0:
        return {"error": "No successful CV folds"}

    df_folds = pd.DataFrame(fold_metrics)
    aggregated = df_folds.mean(numeric_only=True).to_dict()
    logger.info(
        f"Time-series CV completed: mean_rmse={aggregated.get('rmse', float('nan')):.4f}")
    return {"folds": df_folds, "aggregated": aggregated}


# -----------------------------------------------------------------------------
# MODEL COMPARISON & STATISTICAL TESTS
# -----------------------------------------------------------------------------


def compare_models(validation_results: Dict[str, Dict[str, Any]],
                   primary_metric: str = "rmse") -> pd.DataFrame:
    """
    Compare multiple models' validation outputs and return a sorted dataframe.

    validation_results should be a dict keyed by model_name -> validation output
    where validation output contains a top-level 'metrics' dict (from evaluate_regression_model)
    """
    logger.info("Comparing models")
    rows = []
    for model_name, res in validation_results.items():
        metrics = res.get("metrics")
        if not metrics:
            continue
        row = metrics.copy()
        row["model_name"] = model_name
        row["n_predictions"] = res.get("n_predictions", 0)
        # include trading metrics if present
        trading = res.get("trading_metrics", {})
        for k, v in trading.items():
            row[f"trading_{k}"] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    ascending = primary_metric.lower() in ["rmse", "mae", "mape", "smape"]
    if primary_metric not in df.columns:
        logger.warning(
            f"Primary metric {primary_metric} not in comparison DataFrame columns")
        return df
    df_sorted = df.sort_values(
        primary_metric, ascending=ascending).reset_index(drop=True)
    logger.info("Model comparison complete")
    return df_sorted


def statistical_significance_test(y_true: np.ndarray,
                                  y_pred_1: np.ndarray,
                                  y_pred_2: np.ndarray,
                                  test_type: str = "diebold_mariano") -> Dict[str, Any]:
    """
    Statistical test between two model forecasts:
    - 'paired_t' does a paired t-test on absolute errors
    - 'diebold_mariano' computes a simplified DM statistic on squared errors
    """
    logger.info(f"Running statistical test: {test_type}")
    e1 = np.abs(y_true - y_pred_1)
    e2 = np.abs(y_true - y_pred_2)
    if test_type == "paired_t":
        stat, p = stats.ttest_rel(e1, e2)
        return {"test": "paired_t", "statistic": float(stat), "p_value": float(p), "significant": float(p) < 0.05}
    elif test_type == "diebold_mariano":
        # loss differential (use squared errors by default)
        d = (e1 ** 2) - (e2 ** 2)
        mean_d = np.mean(d)
        sd_d = np.std(d, ddof=1)
        n = len(d)
        if sd_d == 0 or n <= 1:
            return {"test": "diebold_mariano", "error": "No variance in loss differential"}
        dm_stat = mean_d / (sd_d / np.sqrt(n))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        return {"test": "diebold_mariano", "statistic": float(dm_stat), "p_value": float(p_value), "significant": float(p_value) < 0.05}
    else:
        raise ValueError(
            "Unsupported test_type: choose 'paired_t' or 'diebold_mariano'.")


# -----------------------------------------------------------------------------
# REPORTING & VISUALIZATION
# -----------------------------------------------------------------------------


def create_evaluation_report(model_name: str,
                             validation_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
    """
    Create a human-readable evaluation report summarizing validation_results.
    validation_results may be the output of any validation method (walk/exp/rolling)
    or a per-horizon dict.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(f"MODEL EVALUATION REPORT: {model_name}")
    lines.append("=" * 80)

    # If validation_results contains multiple horizons (walk-forward style)
    if isinstance(validation_results, dict) and all(isinstance(k, int) for k in validation_results.keys()):
        for h, r in sorted(validation_results.items()):
            lines.append(f"\n--- Horizon: {h} days ---")
            metrics = r.get("metrics", {})
            if metrics:
                lines.append(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                lines.append(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
                lines.append(f"MAPE: {metrics.get('mape', 'N/A'):.2f}%")
                lines.append(f"SMAPE: {metrics.get('smape', 'N/A'):.2f}%")
                lines.append(
                    f"Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
                lines.append(
                    f"Hit Rate ±5%: {metrics.get('hit_rate_5pct', 'N/A'):.2f}%")
            trading = r.get("trading_metrics")
            if trading:
                lines.append(
                    f"Total Return: {trading.get('total_return', 'N/A'):.2f}%")
                lines.append(
                    f"Final Capital: ${trading.get('final_capital', 'N/A'):.2f}")
                lines.append(
                    f"Win Rate: {trading.get('win_rate', 'N/A'):.2f}%")
    else:
        # Single validation dictionary
        metrics = validation_results.get("metrics", {})
        trading = validation_results.get("trading_metrics", {})
        lines.append("\nREGRESSION METRICS:")
        if metrics:
            lines.append(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            lines.append(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
            lines.append(f"MAPE: {metrics.get('mape', 'N/A'):.2f}%")
            lines.append(f"SMAPE: {metrics.get('smape', 'N/A'):.2f}%")
            lines.append(
                f"Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")
            lines.append(
                f"Hit Rate ±5%: {metrics.get('hit_rate_5pct', 'N/A'):.2f}%")
        if trading:
            lines.append("\nTRADING METRICS:")
            lines.append(
                f"Total Return: {trading.get('total_return', 'N/A'):.2f}%")
            lines.append(
                f"Final Capital: ${trading.get('final_capital', 'N/A'):.2f}")
            lines.append(f"Win Rate: {trading.get('win_rate', 'N/A'):.2f}%")

    report_text = "\n".join(lines)
    if save_path:
        try:
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"Saved evaluation report to {save_path}")
            report_text += f"\n\nSaved to {save_path}"
        except Exception as e:
            logger.error(f"Failed to save report to {save_path}: {e}")

    return report_text


def plot_validation_results(validation_results: Dict[str, Any],
                            title: str = "Validation Results",
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Plot predictions vs actuals and residuals.
    Accepts a validation_results dict that contains a 'predictions_df' DataFrame
    with columns ['date','actual','predicted'].
    """
    try:
        if isinstance(validation_results, dict) and "predictions_df" in validation_results:
            df = validation_results["predictions_df"].copy()
            df["date"] = pd.to_datetime(df["date"])
        elif isinstance(validation_results, pd.DataFrame):
            df = validation_results.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
        else:
            logger.warning("No predictions_df available to plot.")
            return

        if "date" not in df.columns:
            df = df.reset_index().rename(columns={'index': 'date'})

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)

        # Time series
        axs[0, 0].plot(df["date"], df["actual"],
                       label="Actual", color="blue", alpha=0.8)
        axs[0, 0].plot(df["date"], df["predicted"], label="Predicted",
                       color="red", linestyle="--", alpha=0.8)
        axs[0, 0].set_title("Predictions vs. Actual Over Time")
        axs[0, 0].set_xlabel("Date")
        axs[0, 0].set_ylabel("Value")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)

        # Scatter Actual vs Predicted
        axs[0, 1].scatter(df["actual"], df["predicted"], alpha=0.6)
        mn = min(df["actual"].min(), df["predicted"].min())
        mx = max(df["actual"].max(), df["predicted"].max())
        axs[0, 1].plot([mn, mx], [mn, mx], "r--", label="Perfect Prediction")
        axs[0, 1].set_title("Scatter Plot: Actual vs. Predicted")
        axs[0, 1].set_xlabel("Actual Values")
        axs[0, 1].set_ylabel("Predicted Values")
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)
        axs[0, 1].legend()

        # Residuals over time
        residuals = df["actual"] - df["predicted"]
        axs[1, 0].plot(df["date"], residuals)
        axs[1, 0].axhline(0, color="r", linestyle="--")
        axs[1, 0].set_title("Residuals over time")
        axs[1, 0].set_xlabel("Date")
        axs[1, 0].set_ylabel("Error (Actual - Predicted)")
        axs[1, 0].grid(True)

        # Residuals histogram
        sns.histplot(residuals, bins=50, kde=True, ax=axs[1, 1])
        axs[1, 1].axvline(residuals.mean(), color='r',
                          linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        axs[1, 1].set_title("Distribution of Residuals")
        axs[1, 1].set_xlabel("Prediction Error")
        axs[1, 1].set_ylabel("Frequency")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved validation plots to {save_path}")
        if show:
            plt.show()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Plotting failed: {e}")


# -----------------------------------------------------------------------------
# HIGH-LEVEL VALIDATION PIPELINE
# -----------------------------------------------------------------------------


def run_comprehensive_validation(data: pd.DataFrame,
                                 model: Any,
                                 feature_columns: List[str],
                                 target_column_template: str,
                                 validation_methods: List[str] = [
                                     "walk_forward", "expanding", "rolling"],
                                 horizons: List[int] = [1, 7, 14, 28],
                                 config_path: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
    """
    Orchestrate a suite of validation strategies and return structured results.

    Args:
        data: DataFrame indexed by date
        model: estimator
        feature_columns: list of feature column names
        target_column_template: format string for per-horizon targets, e.g. "Target_{}d"
        validation_methods: list of methods to run
        horizons: list of horizons (for walk-forward)
        config_path: optional path to config to read parameters (not required)
        **kwargs forwarded to specific validation functions

    Returns:
        dictionary of results keyed by validation method
    """
    logger.info("Running comprehensive validation pipeline")
    results: Dict[str, Any] = {}

    if config_path:
        try:
            cfg = load_config(config_path)
            logger.info(f"Loaded validation config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config at {config_path}: {e}")

    if "walk_forward" in validation_methods:
        logger.info("Executing walk-forward validation")
        try:
            results["walk_forward"] = walk_forward_validation(
                data,
                model,
                horizons=horizons,
                feature_columns=feature_columns,
                target_column_template=target_column_template,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Walk-forward validation error: {e}")

    if "expanding" in validation_methods:
        logger.info("Executing expanding-window validation")
        target_single = target_column_template.format(
            horizons[0]) if "{" in target_column_template else target_column_template
        try:
            results["expanding"] = expanding_window_validation(
                data,
                model,
                feature_columns=feature_columns,
                target_column=target_single,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Expanding validation error: {e}")

    if "rolling" in validation_methods:
        logger.info("Executing rolling-window validation")
        target_single = target_column_template.format(
            horizons[0]) if "{" in target_column_template else target_column_template
        try:
            results["rolling"] = rolling_window_validation(
                data,
                model,
                feature_columns=feature_columns,
                target_column=target_single,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Rolling validation error: {e}")

    logger.info("Comprehensive validation pipeline completed")
    return results
