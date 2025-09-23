"""
Utility Module for Stock Price Prediction Project

This module provides general-purpose utility functions and constants that 
support the main data, features, evaluation, and modeling modules.

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def get_project_path(*subpaths: str) -> Path:
    """
    Get absolute project path by joining subpaths.

    Args:
        subpaths: Subdirectories or filenames to append

    Returns:
        Path object of the constructed path
    """
    base_path = Path(__file__).resolve().parents[1]
    return base_path.joinpath(*subpaths)


# -------------------------------------------------------------------
# General Helper Functions
# -------------------------------------------------------------------

def setup_logging():
    """Setup logging configuration for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)

    # Try multiple possible config file locations
    possible_paths = [
        config_path,
        "config.yaml",
        "config/config.yaml",
        os.path.join("config", "config.yaml")
    ]

    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    import yaml
                    config = yaml.safe_load(file)
                    logger.info(f"Configuration loaded from {path}")
                    return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {path}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")
            continue

    # If no config file found, return defaults
    logger.warning(
        f"No config file found. Tried: {possible_paths}. Using defaults.")
    return {
        'data': {
            'tickers': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'date_range': {
                'start_date': '2020-01-01',
                'end_date': None
            },
            'raw_data_dir': 'data/raw',
            'target_column': 'Close'  # More flexible default
        },
        'logging': {
            'level': 'INFO'
        }
    }


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
    actual: Union[List[float], pd.Series],
    predicted: Union[List[float], pd.Series],
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label="Actual", color="blue", linewidth=2)
    ax.plot(predicted, label="Predicted",
            color="red", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_error_distribution(
    errors: Union[List[float], pd.Series],
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
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors, kde=True, bins=30, ax=ax, color="purple")
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


# -------------------------------------------------------------------
# Report Generation Utilities
# -------------------------------------------------------------------

def generate_text_report(metrics: Dict[str, Any], title: str = "Model Evaluation Report") -> str:
    """
    Generate a formatted text report from metrics.

    Args:
        metrics: Dictionary of evaluation metrics
        title: Report title

    Returns:
        Formatted string report
    """
    lines = [f"{'='*60}", title, f"{'='*60}"]
    for key, value in metrics.items():
        lines.append(f"{key:30}: {value}")
    report = "\n".join(lines)
    return report


def save_text_report(report: str, filepath: Union[str, Path]) -> None:
    """
    Save a text report to file.

    Args:
        report: Report string
        filepath: Destination file path
    """
    filepath = Path(filepath)
    ensure_dir_exists(filepath.parent)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        logging.info(f"Report saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save report to {filepath}: {e}")
        raise


# -------------------------------------------------------------------
# Error Handling Utilities
# -------------------------------------------------------------------

def safe_execute(func, *args, **kwargs) -> Optional[Any]:
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


# Initialize logging for this module
logger = setup_logging()
