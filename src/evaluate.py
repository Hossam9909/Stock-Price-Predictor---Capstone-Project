"""
Model Evaluation Module - NEW MODULE (NO OVERLAP with data.py)

This module provides comprehensive evaluation metrics and validation strategies 
specifically designed for stock price prediction models and time-series forecasting.

Author: Stock Price Predictor Project
Dependencies: Uses src.data for data loading only (NO DUPLICATION)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# REGRESSION METRICS (NEW FUNCTIONALITY)
# =============================================================================


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        epsilon (float): Small value to avoid division by zero

    Returns:
        float: MAPE value as percentage
    """
    # Avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: SMAPE value as percentage
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared coefficient of determination.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: R-squared value
    """
    return r2_score(y_true, y_pred)

# =============================================================================
# FINANCIAL METRICS (NEW FUNCTIONALITY)
# =============================================================================


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).

    Args:
        y_true (np.ndarray): True price changes
        y_pred (np.ndarray): Predicted price changes

    Returns:
        float: Directional accuracy as percentage
    """
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    correct_directions = np.sum(true_direction == pred_direction)
    return (correct_directions / len(y_true)) * 100


def calculate_profit_loss(y_true: np.ndarray, y_pred: np.ndarray,
                          initial_capital: float = 10000.0,
                          transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    Calculate profit/loss metrics for a simple trading strategy.

    Args:
        y_true (np.ndarray): True price changes
        y_pred (np.ndarray): Predicted price changes
        initial_capital (float): Starting capital
        transaction_cost (float): Transaction cost as percentage

    Returns:
        Dict[str, float]: Trading performance metrics
    """
    positions = np.sign(y_pred)  # 1 for buy, -1 for sell, 0 for hold
    returns = positions * y_true

    # Apply transaction costs
    position_changes = np.diff(np.concatenate([[0], positions]))
    trade_costs = np.abs(position_changes) * transaction_cost
    net_returns = returns[1:] - trade_costs

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + net_returns)
    final_capital = initial_capital * \
        cumulative_returns[-1] if len(
            cumulative_returns) > 0 else initial_capital

    return {
        'total_return': (final_capital - initial_capital) / initial_capital * 100,
        'final_capital': final_capital,
        'total_trades': np.sum(np.abs(position_changes) > 0),
        'avg_return_per_trade': np.mean(net_returns) * 100,
        'win_rate': np.sum(net_returns > 0) / len(net_returns) * 100 if len(net_returns) > 0 else 0,
        'sharpe_ratio': np.mean(net_returns) / np.std(net_returns) if np.std(net_returns) > 0 else 0
    }


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from return series.

    Args:
        returns (np.ndarray): Return series

    Returns:
        float: Maximum drawdown as percentage
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown) * 100

# =============================================================================
# COMPREHENSIVE EVALUATION (NEW FUNCTIONALITY)
# =============================================================================


def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model") -> Dict[str, float]:
    """
    Comprehensive evaluation of regression model performance.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model for reporting

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred)
    }

    return metrics


def evaluate_trading_strategy(y_true: np.ndarray, y_pred: np.ndarray,
                              initial_capital: float = 10000.0,
                              transaction_cost: float = 0.001) -> Dict[str, float]:
    """
    Evaluate trading strategy performance.

    Args:
        y_true (np.ndarray): True price changes
        y_pred (np.ndarray): Predicted price changes
        initial_capital (float): Starting capital
        transaction_cost (float): Transaction cost as percentage

    Returns:
        Dict[str, float]: Trading performance metrics
    """
    return calculate_profit_loss(y_true, y_pred, initial_capital, transaction_cost)

# =============================================================================
# TIME-SERIES VALIDATION STRATEGIES (NEW FUNCTIONALITY)
# =============================================================================


def walk_forward_validation(data: pd.DataFrame, model, horizons: List[int],
                            feature_columns: List[str], target_column: str,
                            min_train_size: int = 252, step_size: int = 21) -> Dict[str, Any]:
    """
    Perform walk-forward validation for time-series models.

    Args:
        data (pd.DataFrame): Time-series data with features and target
        model: Model object with fit() and predict() methods
        horizons (List[int]): Forecast horizons to evaluate
        feature_columns (List[str]): List of feature column names
        target_column (str): Target column name
        min_train_size (int): Minimum training window size
        step_size (int): Number of periods to move forward each iteration

    Returns:
        Dict[str, Any]: Validation results for each horizon
    """
    results = {horizon: {'predictions': [], 'actuals': [], 'dates': []}
               for horizon in horizons}

    # Ensure data is sorted by date
    data_sorted = data.sort_index()

    # Get feature and target data
    X = data_sorted[feature_columns].values
    y = data_sorted[target_column].values
    dates = data_sorted.index

    # Walk forward through the data
    start_idx = min_train_size

    while start_idx < len(data_sorted) - max(horizons):
        # Training data
        X_train = X[:start_idx]
        y_train = y[:start_idx]

        # Fit model
        try:
            model.fit(X_train, y_train)

            # Make predictions for different horizons
            for horizon in horizons:
                if start_idx + horizon - 1 < len(X):
                    # Test data for this horizon
                    X_test = X[start_idx + horizon - 1:start_idx + horizon]
                    y_test = y[start_idx + horizon - 1:start_idx + horizon]
                    test_date = dates[start_idx + horizon - 1]

                    # Predict
                    y_pred = model.predict(X_test)

                    # Store results
                    results[horizon]['predictions'].extend(y_pred)
                    results[horizon]['actuals'].extend(y_test)
                    results[horizon]['dates'].append(test_date)

        except Exception as e:
            print(
                f"Warning: Model fitting failed at index {start_idx}: {str(e)}")

        # Move forward
        start_idx += step_size

    # Calculate metrics for each horizon
    validation_metrics = {}
    for horizon in horizons:
        if len(results[horizon]['predictions']) > 0:
            y_true = np.array(results[horizon]['actuals'])
            y_pred = np.array(results[horizon]['predictions'])

            validation_metrics[horizon] = {
                'n_predictions': len(y_pred),
                'metrics': evaluate_regression_model(y_true, y_pred, f"Horizon_{horizon}"),
                'trading_metrics': evaluate_trading_strategy(y_true, y_pred),
                'predictions_df': pd.DataFrame({
                    'date': results[horizon]['dates'],
                    'actual': results[horizon]['actuals'],
                    'predicted': results[horizon]['predictions']
                })
            }

    return validation_metrics


def expanding_window_validation(data: pd.DataFrame, model,
                                feature_columns: List[str], target_column: str,
                                min_train_size: int = 252, test_size: int = 21) -> Dict[str, Any]:
    """
    Perform expanding window validation (anchored walk-forward).

    Args:
        data (pd.DataFrame): Time-series data
        model: Model object with fit() and predict() methods
        feature_columns (List[str]): Feature column names
        target_column (str): Target column name
        min_train_size (int): Minimum training window size
        test_size (int): Size of test set for each iteration

    Returns:
        Dict[str, Any]: Validation results
    """
    data_sorted = data.sort_index()
    X = data_sorted[feature_columns].values
    y = data_sorted[target_column].values
    dates = data_sorted.index

    predictions = []
    actuals = []
    test_dates = []

    # Start with minimum training size
    train_end = min_train_size

    while train_end + test_size <= len(data_sorted):
        # Expanding training window (always starts from beginning)
        X_train = X[:train_end]
        y_train = y[:train_end]

        # Fixed test window
        X_test = X[train_end:train_end + test_size]
        y_test = y[train_end:train_end + test_size]
        test_period_dates = dates[train_end:train_end + test_size]

        try:
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test)
            test_dates.extend(test_period_dates)

        except Exception as e:
            print(
                f"Warning: Model fitting failed at train_end {train_end}: {str(e)}")

        # Move to next test period
        train_end += test_size

    # Calculate metrics
    if len(predictions) > 0:
        y_true = np.array(actuals)
        y_pred = np.array(predictions)

        return {
            'n_predictions': len(y_pred),
            'metrics': evaluate_regression_model(y_true, y_pred, "Expanding_Window"),
            'trading_metrics': evaluate_trading_strategy(y_true, y_pred),
            'predictions_df': pd.DataFrame({
                'date': test_dates,
                'actual': actuals,
                'predicted': predictions
            })
        }
    else:
        return {'error': 'No predictions generated'}


def rolling_window_validation(data: pd.DataFrame, model,
                              feature_columns: List[str], target_column: str,
                              train_size: int = 252, test_size: int = 21,
                              step_size: int = 21) -> Dict[str, Any]:
    """
    Perform rolling window validation (sliding window).

    Args:
        data (pd.DataFrame): Time-series data
        model: Model object with fit() and predict() methods  
        feature_columns (List[str]): Feature column names
        target_column (str): Target column name
        train_size (int): Size of training window
        test_size (int): Size of test window
        step_size (int): Step size for rolling

    Returns:
        Dict[str, Any]: Validation results
    """
    data_sorted = data.sort_index()
    X = data_sorted[feature_columns].values
    y = data_sorted[target_column].values
    dates = data_sorted.index

    predictions = []
    actuals = []
    test_dates = []

    # Rolling window validation
    start_idx = 0

    while start_idx + train_size + test_size <= len(data_sorted):
        # Rolling training window
        X_train = X[start_idx:start_idx + train_size]
        y_train = y[start_idx:start_idx + train_size]

        # Test window
        X_test = X[start_idx + train_size:start_idx + train_size + test_size]
        y_test = y[start_idx + train_size:start_idx + train_size + test_size]
        test_period_dates = dates[start_idx +
                                  train_size:start_idx + train_size + test_size]

        try:
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store results
            predictions.extend(y_pred)
            actuals.extend(y_test)
            test_dates.extend(test_period_dates)

        except Exception as e:
            print(
                f"Warning: Model fitting failed at start_idx {start_idx}: {str(e)}")

        # Move window forward
        start_idx += step_size

    # Calculate metrics
    if len(predictions) > 0:
        y_true = np.array(actuals)
        y_pred = np.array(predictions)

        return {
            'n_predictions': len(y_pred),
            'metrics': evaluate_regression_model(y_true, y_pred, "Rolling_Window"),
            'trading_metrics': evaluate_trading_strategy(y_true, y_pred),
            'predictions_df': pd.DataFrame({
                'date': test_dates,
                'actual': actuals,
                'predicted': predictions
            })
        }
    else:
        return {'error': 'No predictions generated'}

# =============================================================================
# MODEL COMPARISON (NEW FUNCTIONALITY)
# =============================================================================


def compare_models(validation_results: Dict[str, Dict],
                   primary_metric: str = 'rmse') -> pd.DataFrame:
    """
    Compare multiple models based on validation results.

    Args:
        validation_results (Dict[str, Dict]): Dictionary of model validation results
        primary_metric (str): Primary metric for ranking models

    Returns:
        pd.DataFrame: Comparison table sorted by primary metric
    """
    comparison_data = []

    for model_name, results in validation_results.items():
        if 'metrics' in results:
            metrics = results['metrics'].copy()
            metrics['model_name'] = model_name
            metrics['n_predictions'] = results.get('n_predictions', 0)

            # Add trading metrics if available
            if 'trading_metrics' in results:
                trading_metrics = results['trading_metrics']
                metrics.update(
                    {f"trading_{k}": v for k, v in trading_metrics.items()})

            comparison_data.append(metrics)

    if not comparison_data:
        return pd.DataFrame()

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by primary metric (ascending for error metrics, descending for accuracy/return metrics)
    ascending = primary_metric.lower() in ['rmse', 'mae', 'mape', 'smape']
    comparison_df = comparison_df.sort_values(
        primary_metric, ascending=ascending)

    return comparison_df


def statistical_significance_test(y_true: np.ndarray,
                                  y_pred_1: np.ndarray,
                                  y_pred_2: np.ndarray,
                                  test_type: str = 'diebold_mariano') -> Dict[str, float]:
    """
    Test statistical significance between two model predictions.

    Args:
        y_true (np.ndarray): True values
        y_pred_1 (np.ndarray): Predictions from model 1
        y_pred_2 (np.ndarray): Predictions from model 2
        test_type (str): Type of test ('diebold_mariano' or 'paired_t')

    Returns:
        Dict[str, float]: Test results
    """
    from scipy import stats

    # Calculate prediction errors
    errors_1 = np.abs(y_true - y_pred_1)
    errors_2 = np.abs(y_true - y_pred_2)

    if test_type == 'paired_t':
        # Paired t-test on absolute errors
        statistic, p_value = stats.ttest_rel(errors_1, errors_2)

        return {
            'test_type': 'paired_t_test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'model_1_better': statistic > 0  # Model 1 has higher errors
        }

    elif test_type == 'diebold_mariano':
        # Simplified Diebold-Mariano test
        diff = errors_1 - errors_2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        if std_diff > 0:
            dm_statistic = mean_diff / (std_diff / np.sqrt(len(diff)))
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_statistic)))

            return {
                'test_type': 'diebold_mariano',
                'statistic': dm_statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'model_1_better': dm_statistic > 0
            }
        else:
            return {
                'test_type': 'diebold_mariano',
                'error': 'No variance in prediction differences'
            }

# =============================================================================
# UTILITY FUNCTIONS (NEW FUNCTIONALITY)
# =============================================================================


def create_evaluation_report(model_name: str, validation_results: Dict[str, Any],
                             save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive evaluation report.

    Args:
        model_name (str): Name of the model
        validation_results (Dict[str, Any]): Validation results
        save_path (str, optional): Path to save the report

    Returns:
        str: Evaluation report as string
    """
    report = []
    report.append(f"{'='*60}")
    report.append(f"MODEL EVALUATION REPORT: {model_name}")
    report.append(f"{'='*60}")

    if 'metrics' in validation_results:
        metrics = validation_results['metrics']
        report.append(f"\n📊 REGRESSION METRICS:")
        report.append(f"  • RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        report.append(f"  • MAE: {metrics.get('mae', 'N/A'):.4f}")
        report.append(f"  • MAPE: {metrics.get('mape', 'N/A'):.2f}%")
        report.append(f"  • SMAPE: {metrics.get('smape', 'N/A'):.2f}%")
        report.append(f"  • R²: {metrics.get('r2', 'N/A'):.4f}")
        report.append(
            f"  • Directional Accuracy: {metrics.get('directional_accuracy', 'N/A'):.2f}%")

    if 'trading_metrics' in validation_results:
        trading = validation_results['trading_metrics']
        report.append(f"\n💰 TRADING METRICS:")
        report.append(
            f"  • Total Return: {trading.get('total_return', 'N/A'):.2f}%")
        report.append(
            f"  • Final Capital: ${trading.get('final_capital', 'N/A'):.2f}")
        report.append(
            f"  • Total Trades: {trading.get('total_trades', 'N/A')}")
        report.append(f"  • Win Rate: {trading.get('win_rate', 'N/A'):.2f}%")
        report.append(
            f"  • Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A'):.4f}")
        report.append(
            f"  • Avg Return per Trade: {trading.get('avg_return_per_trade', 'N/A'):.4f}%")

    report.append(f"\n📈 VALIDATION INFO:")
    report.append(
        f"  • Number of Predictions: {validation_results.get('n_predictions', 'N/A')}")

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        report_text += f"\n\n📁 Report saved to: {save_path}"

    return report_text


def plot_validation_results(validation_results: Dict[str, Any],
                            model_name: str = "Model",
                            save_path: Optional[str] = None) -> None:
    """
    Create validation result plots (requires matplotlib).

    Args:
        validation_results (Dict[str, Any]): Validation results
        model_name (str): Model name for title
        save_path (str, optional): Path to save plots
    """
    try:
        import matplotlib.pyplot as plt

        if 'predictions_df' in validation_results:
            df = validation_results['predictions_df']

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Validation Results: {model_name}', fontsize=16)

            # Time series plot
            axes[0, 0].plot(df['date'], df['actual'],
                            label='Actual', alpha=0.7)
            axes[0, 0].plot(df['date'], df['predicted'],
                            label='Predicted', alpha=0.7)
            axes[0, 0].set_title('Actual vs Predicted Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Scatter plot
            axes[0, 1].scatter(df['actual'], df['predicted'], alpha=0.6)
            min_val = min(df['actual'].min(), df['predicted'].min())
            max_val = max(df['actual'].max(), df['predicted'].max())
            axes[0, 1].plot([min_val, max_val], [
                            min_val, max_val], 'r--', alpha=0.8)
            axes[0, 1].set_title('Actual vs Predicted Scatter')
            axes[0, 1].set_xlabel('Actual')
            axes[0, 1].set_ylabel('Predicted')
            axes[0, 1].grid(True, alpha=0.3)

            # Residuals plot
            residuals = df['actual'] - df['predicted']
            axes[1, 0].plot(df['date'], residuals)
            axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 0].set_title('Residuals Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].grid(True, alpha=0.3)

            # Residuals histogram
            axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
            axes[1, 1].set_title('Residuals Distribution')
            axes[1, 1].set_xlabel('Residuals')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"📊 Plots saved to: {save_path}")
            else:
                plt.show()

    except ImportError:
        print("Warning: matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")


# =============================================================================
# VALIDATION PIPELINE (NEW FUNCTIONALITY)
# =============================================================================

def run_comprehensive_validation(data: pd.DataFrame, model,
                                 feature_columns: List[str], target_column: str,
                                 validation_methods: List[str] = [
                                     'walk_forward', 'expanding', 'rolling'],
                                 horizons: List[int] = [1, 5, 10],
                                 **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive validation using multiple methods.

    Args:
        data (pd.DataFrame): Time-series data
        model: Model object
        feature_columns (List[str]): Feature column names
        target_column (str): Target column name
        validation_methods (List[str]): Validation methods to use
        horizons (List[int]): Forecast horizons for walk-forward validation
        **kwargs: Additional arguments for validation methods

    Returns:
        Dict[str, Dict[str, Any]]: Comprehensive validation results
    """
    results = {}

    if 'walk_forward' in validation_methods:
        print("🚶 Running walk-forward validation...")
        try:
            wf_results = walk_forward_validation(
                data, model, horizons, feature_columns, target_column, **kwargs
            )
            results['walk_forward'] = wf_results
        except Exception as e:
            print(f"Error in walk-forward validation: {str(e)}")

    if 'expanding' in validation_methods:
        print("📈 Running expanding window validation...")
        try:
            exp_results = expanding_window_validation(
                data, model, feature_columns, target_column, **kwargs
            )
            results['expanding'] = exp_results
        except Exception as e:
            print(f"Error in expanding window validation: {str(e)}")

    if 'rolling' in validation_methods:
        print("🔄 Running rolling window validation...")
        try:
            roll_results = rolling_window_validation(
                data, model, feature_columns, target_column, **kwargs
            )
            results['rolling'] = roll_results
        except Exception as e:
            print(f"Error in rolling window validation: {str(e)}")

    return results
