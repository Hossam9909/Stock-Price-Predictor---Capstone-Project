#!/usr/bin/env python3
"""
Stock Price Predictor Pipeline Runner

This script runs the complete end-to-end pipeline for stock price prediction.
It can be used to run individual components or the entire pipeline.

Usage:
    python scripts/run_pipeline.py --step data
    python scripts/run_pipeline.py --step all
    python scripts/run_pipeline.py --config config/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
from src.utils import load_config, setup_logging
from src.data import download_multiple_tickers, get_default_tickers, get_default_date_range
from src.features import process_stock_features
from src.models import train_models_multi_horizon, evaluate_with_walk_forward
from src.evaluate import create_evaluation_report, plot_validation_results

def run_data_pipeline(config: dict):
    """Run data ingestion and processing"""
    logging.info("--- Running Data Pipeline ---")
    tickers = get_default_tickers(config)
    start_date, end_date = get_default_date_range(config)
    raw_dir = config.get('paths', {}).get('data_raw', 'data/raw')

    download_multiple_tickers(tickers, start_date, end_date, raw_dir)
    logging.info("--- Data Pipeline Finished ---")


def run_feature_engineering(config: dict):
    """Run feature engineering pipeline"""
    logging.info("--- Running Feature Engineering Pipeline ---")
    tickers = get_default_tickers(config)
    raw_dir = config.get('paths', {}).get('data_raw', 'data/raw')
    processed_dir = config.get('paths', {}).get('data_processed', 'data/processed')
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        raw_path = Path(raw_dir) / f"{ticker}.csv"
        if raw_path.exists():
            features_df = process_stock_features(ticker, str(raw_path), config_path=config['config_path'])
            features_df.to_csv(Path(processed_dir) / f"{ticker}_features.csv")
    logging.info("--- Feature Engineering Pipeline Finished ---")


def run_model_training(config: dict):
    """Run model training pipeline"""
    logging.info("--- Running Model Training Pipeline ---")
    tickers = get_default_tickers(config)
    processed_dir = config.get('paths', {}).get('data_processed', 'data/processed')
    models_dir = config.get('paths', {}).get('models_dir', 'experiments/models')
    model_types = config.get('models', {}).get('types', ['rf'])
    horizons = config.get('horizons', [1, 7, 14])

    for ticker in tickers:
        features_path = Path(processed_dir) / f"{ticker}_features.csv"
        if not features_path.exists():
            logging.warning(f"Feature file not found for {ticker}, skipping training.")
            continue

        logging.info(f"Training models for {ticker}...")
        df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)

        # Define feature columns (exclude OHLC, Volume, and Targets)
        exclude_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"}
        feature_cols = [c for c in df_features.columns if not c.startswith("Target_") and c not in exclude_cols]

        for model_type in model_types:
            logging.info(f"  Training model type: {model_type}")
            # train_models_multi_horizon handles training for all horizons
            trained_records = train_models_multi_horizon(
                df=df_features,
                feature_columns=feature_cols,
                horizons=horizons,
                model_type=model_type,
                save_dir=None,  # We will save manually to control filename
                config_path=config['config_path']
            )

            # Save each trained model with the correct filename convention
            for h, record in trained_records.items():
                model_filename = f"{ticker}_{model_type}_h{h}_{pd.Timestamp.now():%Y%m%d}.joblib"
                save_path = Path(models_dir) / model_filename
                record.model.save_model(str(save_path))

    logging.info("--- Model Training Pipeline Finished ---")


def run_evaluation(config: dict):
    """Run model evaluation pipeline using walk-forward validation."""
    logging.info("--- Running Evaluation Pipeline ---")
    tickers = get_default_tickers(config)
    processed_dir = config.get('paths', {}).get('data_processed', 'data/processed')
    results_dir = config.get('paths', {}).get('results_dir', 'experiments/results')
    figures_dir = config.get('paths', {}).get('figures_dir', 'experiments/figures')
    model_types = config.get('models', {}).get('types', ['rf'])
    horizons = config.get('horizons', [1, 7, 14])

    # Ensure output directories exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        features_path = Path(processed_dir) / f"{ticker}_features.csv"
        if not features_path.exists():
            logging.warning(f"Feature file not found for {ticker}, skipping evaluation.")
            continue

        logging.info(f"Evaluating models for {ticker}...")
        df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)

        # Define feature columns
        exclude_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"}
        feature_cols = [c for c in df_features.columns if not c.startswith("Target_") and c not in exclude_cols]

        for model_type in model_types:
            logging.info(f"  Evaluating model type: {model_type}")

            validation_results = evaluate_with_walk_forward(
                df=df_features,
                feature_columns=feature_cols,
                horizons=horizons,
                model_type=model_type,
                config_path=config['config_path']
            )

            report_path = Path(results_dir) / f"{ticker}_{model_type}_summary_report.txt"
            create_evaluation_report(
                model_name=f"{ticker} - {model_type}",
                validation_results=validation_results,
                save_path=str(report_path)
            )

            for h, res in validation_results.items():
                if "predictions_df" in res:
                    fig_path = Path(figures_dir) / f"{ticker}_{model_type}_h{h}_validation_plot.png"
                    plot_validation_results(
                        validation_results=res,
                        title=f"Validation: {ticker} - {model_type} - {h}-day Horizon",
                        save_path=str(fig_path),
                        show=False
                    )

    logging.info("--- Evaluation Pipeline Finished ---")


def run_complete_pipeline(config: dict):
    """Run the complete pipeline"""
    run_data_pipeline(config)
    run_feature_engineering(config)
    run_model_training(config)
    run_evaluation(config)


def main():
    """Main function to handle command line arguments and run pipeline"""
    parser = argparse.ArgumentParser(
        description="Stock Price Predictor Pipeline Runner"
    )

    parser.add_argument(
        "--step",
        choices=["data", "features", "train", "evaluate", "all"],
        default="all",
        help="Pipeline step to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['config_path'] = args.config # Store path for later use

    # Setup logging
    setup_logging(config)

    # Log start of pipeline
    logging.info(f"Starting pipeline step: {args.step}")
    logging.info(f"Using config: {args.config}")

    try:
        if args.step == "data":
            run_data_pipeline(config)
        elif args.step == "features":
            run_feature_engineering(config)
        elif args.step == "train":
            run_model_training(config)
        elif args.step == "evaluate":
            run_evaluation(config)
        elif args.step == "all":
            run_complete_pipeline(config)

        logging.info(f"Pipeline step '{args.step}' completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
