"""
Utility Module for Stock Price Prediction Project

This module provides general-purpose utility functions and constants that 
support the main data, features, evaluation, and modeling modules.

This is a STANDALONE UTILITIES MODULE - no dependencies on other project modules.

Contents:
    - Project constants and enums
    - File path utilities
    - General helper functions
    - Plotting utilities
    - Report generation helpers
    - Error handling utilities
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# -------------------------------------------------------------------
# Project Constants and Enums
# -------------------------------------------------------------------

class ModelType(str, Enum):
    """Enumeration of supported model types."""
    NAIVE = "naive"
    RANDOM_WALK = "random_walk"
    RF = "rf"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    LSTM = "lstm"


class FileFormat(str, Enum):
    """Enumeration of supported file formats for saving results."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


DEFAULT_RESULTS_DIR = "results"
DEFAULT_FIGURES_DIR = "figures"


# -------------------------------------------------------------------
# File Path Utilities
# -------------------------------------------------------------------

def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists; create it if not.

    Args:
        path: Directory path

    Returns:
        Path object of created/existing directory
    """
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root() -> Path:
    """
    Get the project root directory path.

    Returns:
        Path object of the project root
    """
    return Path(__file__).resolve().parents[1]


def get_project_path(*subpaths: str) -> Path:
    """
    Get absolute project path by joining subpaths to project root.

    Args:
        subpaths: Subdirectories or filenames to append

    Returns:
        Path object of the constructed path
    """
    return get_project_root().joinpath(*subpaths)


# -------------------------------------------------------------------
# General Helper Functions
# -------------------------------------------------------------------

def save_results_to_json(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save results dictionary to a JSON file.

    Args:
        results: Dictionary of results
        filepath: File path to save JSON
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, default=str)
        logging.info(f"Results saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save results to {filepath}: {e}")
        raise


def load_results_from_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results dictionary from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary of loaded results

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)
        logging.info(f"Results loaded from {filepath}")
        return results
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {filepath}: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to load results from {filepath}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], fmt: FileFormat = FileFormat.CSV) -> None:
    """
    Save a DataFrame to a specified format.

    Args:
        df: DataFrame to save
        filepath: Destination path
        fmt: File format (csv, excel, json)
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)

    try:
        if fmt == FileFormat.CSV:
            df.to_csv(filepath, index=True)
        elif fmt == FileFormat.EXCEL:
            df.to_excel(filepath, index=True)
        elif fmt == FileFormat.JSON:
            df.to_json(filepath, orient="records", indent=4)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")
        logging.info(f"DataFrame saved to {filepath} ({fmt})")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {filepath}: {e}")
        raise


# -------------------------------------------------------------------
# Plotting Utilities
# -------------------------------------------------------------------

def plot_predictions_vs_actual(
    actual: Union[List[float], pd.Series, np.ndarray],
    predicted: Union[List[float], pd.Series, np.ndarray],
    title: str = "Predictions vs Actual"
) -> plt.Figure:
    """
    Plot predictions vs actual values.

    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to pandas Series if needed for better plotting
    if isinstance(actual, (list, np.ndarray)):
        actual = pd.Series(actual)
    if isinstance(predicted, (list, np.ndarray)):
        predicted = pd.Series(predicted)

    ax.plot(actual.values, label="Actual",
            color="blue", linewidth=2, alpha=0.8)
    ax.plot(predicted.values, label="Predicted", color="red",
            linestyle="--", linewidth=2, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "rmse") -> plt.Figure:
    """
    Plot model comparison bar chart.

    Args:
        results_df: DataFrame with model results (must have 'model_name' column and metric column)
        metric: Metric to plot

    Returns:
        Matplotlib Figure object
    """
    if 'model_name' not in results_df.columns:
        raise ValueError("results_df must contain 'model_name' column")
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results_df columns")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by metric (ascending for error metrics, descending for accuracy metrics)
    ascending = metric.lower() in ['rmse', 'mae', 'mape', 'smape']
    df_sorted = results_df.sort_values(metric, ascending=ascending)

    bars = ax.bar(df_sorted['model_name'], df_sorted[metric],
                  color='steelblue', alpha=0.7, edgecolor='navy')

    ax.set_title(
        f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show

    Returns:
        Matplotlib Figure object
    """
    if 'importance' not in importance_df.columns:
        # Try to infer importance column
        numeric_cols = importance_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            importance_col = numeric_cols[0]
        else:
            raise ValueError(
                "importance_df must contain 'importance' column or single numeric column")
    else:
        importance_col = 'importance'

    # Get feature names
    if 'feature' in importance_df.columns:
        feature_col = 'feature'
    elif importance_df.index.name:
        importance_df = importance_df.reset_index()
        feature_col = importance_df.columns[0]
    else:
        importance_df = importance_df.reset_index()
        feature_col = 'index'

    # Sort and select top features
    df_sorted = importance_df.sort_values(
        importance_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    bars = ax.barh(range(len(df_sorted)), df_sorted[importance_col],
                   color='forestgreen', alpha=0.7, edgecolor='darkgreen')

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted[feature_col])
    ax.invert_yaxis()  # Highest importance at top

    ax.set_title(f'Top {top_n} Feature Importances',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance', fontsize=12)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}', ha='left', va='center', fontsize=9)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def create_time_series_plot(data: pd.DataFrame, columns: List[str],
                            title: str = "Time Series Plot") -> plt.Figure:
    """
    Create a time series plot for specified columns.

    Args:
        data: DataFrame with datetime index
        columns: List of column names to plot
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have datetime index")

    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))

    for i, col in enumerate(columns):
        ax.plot(data.index, data[col], label=col, color=colors[i], linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig


def plot_error_distribution(
    errors: Union[List[float], pd.Series, np.ndarray],
    title: str = "Prediction Error Distribution"
) -> plt.Figure:
    """
    Plot distribution of prediction errors.

    Args:
        errors: Prediction errors
        title: Plot title

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to numpy array for consistency
    if isinstance(errors, (pd.Series, list)):
        errors = np.array(errors)

    # Remove any NaN values
    errors_clean = errors[~np.isnan(errors)]

    if len(errors_clean) == 0:
        ax.text(0.5, 0.5, "No valid errors to plot",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    sns.histplot(errors_clean, kde=True, bins=30,
                 ax=ax, color="purple", alpha=0.7)

    # Add statistics
    mean_err = np.mean(errors_clean)
    std_err = np.std(errors_clean)
    ax.axvline(mean_err, color='red', linestyle='--',
               label=f'Mean: {mean_err:.4f}')
    ax.axvline(mean_err + std_err, color='orange', linestyle=':', alpha=0.7,
               label=f'±1 Std: {std_err:.4f}')
    ax.axvline(mean_err - std_err, color='orange', linestyle=':', alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Error", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Report Generation Utilities
# -------------------------------------------------------------------

def generate_performance_report(results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive performance report from results.

    Args:
        results: Dictionary containing model evaluation results

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("="*80)
    lines.append("STOCK PRICE PREDICTION - PERFORMANCE REPORT")
    lines.append("="*80)
    lines.append("")

    # Summary section
    if 'summary' in results:
        lines.append("SUMMARY")
        lines.append("-" * 40)
        summary = results['summary']
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key:30}: {value:.4f}")
            else:
                lines.append(f"{key:30}: {value}")
        lines.append("")

    # Model performance
    if 'models' in results:
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 40)
        for model_name, metrics in results['models'].items():
            lines.append(f"\n{model_name}:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {metric:25}: {value:.4f}")
                    else:
                        lines.append(f"  {metric:25}: {value}")
        lines.append("")

    # Trading metrics
    if 'trading' in results:
        lines.append("TRADING PERFORMANCE")
        lines.append("-" * 40)
        trading = results['trading']
        for key, value in trading.items():
            if isinstance(value, (int, float)):
                if 'return' in key.lower() or 'rate' in key.lower():
                    lines.append(f"{key:30}: {value:.2f}%")
                elif 'capital' in key.lower():
                    lines.append(f"{key:30}: ${value:,.2f}")
                else:
                    lines.append(f"{key:30}: {value:.4f}")
            else:
                lines.append(f"{key:30}: {value}")
        lines.append("")

    # Additional sections
    for section_name, section_data in results.items():
        if section_name not in ['summary', 'models', 'trading'] and isinstance(section_data, dict):
            lines.append(f"{section_name.upper()}")
            lines.append("-" * 40)
            for key, value in section_data.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{key:30}: {value:.4f}")
                else:
                    lines.append(f"{key:30}: {value}")
            lines.append("")

    lines.append("="*80)
    lines.append("Report generated at: " + time.strftime('%Y-%m-%d %H:%M:%S'))

    return "\n".join(lines)


def create_html_report(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Create an HTML report from results.

    Args:
        results: Dictionary containing results
        output_path: Path to save HTML file
    """
    output_path = Path(output_path)
    ensure_dir_exists(output_path.parent)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Price Prediction Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 20px 0; }}
            .metrics {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Stock Price Prediction Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Results Summary</h2>
            <div class="metrics">
                <pre>{generate_performance_report(results)}</pre>
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"HTML report saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to create HTML report: {e}")
        raise


def format_metrics_table(metrics_dict: Dict[str, Any]) -> str:
    """
    Format a metrics dictionary as a table string.

    Args:
        metrics_dict: Dictionary of metric name -> value

    Returns:
        Formatted table string
    """
    if not metrics_dict:
        return "No metrics to display"

    # Calculate column widths
    max_name_len = max(len(str(k)) for k in metrics_dict.keys())
    max_value_len = max(len(f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
                        for v in metrics_dict.values())

    # Create table
    lines = []
    header = f"{'Metric':<{max_name_len}} | {'Value':<{max_value_len}}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        lines.append(f"{name:<{max_name_len}} | {value_str:<{max_value_len}}")

    return "\n".join(lines)


# -------------------------------------------------------------------
# Error Handling and Utility Decorators
# -------------------------------------------------------------------

def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logging.info(
                f"{func.__name__} completed in {duration:.2f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logging.error(
                f"{func.__name__} failed after {duration:.2f} seconds: {e}")
            raise
    return wrapper


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Decorator to retry function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logging.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"{func.__name__} failed after {max_attempts} attempts")

            raise last_exception
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result if successful, None otherwise
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error executing {func.__name__}: {e}")
        return None


def validate_input_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns and basic data quality.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Check if DataFrame is empty
    if df.empty:
        logger.error("DataFrame is empty")
        return False

    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Check for all-null columns
    null_cols = [col for col in required_columns if df[col].isnull().all()]
    if null_cols:
        logger.error(f"Columns with all null values: {null_cols}")
        return False

    # Check data types
    for col in required_columns:
        if col in df.columns and df[col].dtype == 'object':
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                logger.warning(f"Column '{col}' contains non-numeric data")

    logger.info("Input data validation passed")
    return True


# Initialize logging for this module (minimal setup)
logger = logging.getLogger(__name__)
