"""
Streamlit Web Application for Stock Price Prediction

This app provides an interactive interface to:
- Select stock ticker, date range, prediction horizon, and model
- Load and preprocess stock data using src.data
- Generate features using src.features
- Load trained ML models from experiments/models
- Display real-time predictions and historical performance
- Show evaluation metrics and data quality indicators

Author: Stock Price Prediction Project
"""

from src.evaluate import evaluate_regression_model, evaluate_trading_strategy
from src.models import (
    NaiveLastValue,
    RandomWalkDrift,
    RandomForestPredictor,
    LGBMPredictor,
    XGBPredictor,
)
from src.features import create_all_features
from src.data import load_raw_data, clean_data, validate_data_quality
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Ensure src is on the path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "src")))

# Import project modules
from src.utils import load_config


# =============================
# APP CONFIGURATION
# =============================
st.set_page_config(
    page_title="📈 Stock Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

config = load_config("config/config.yaml")


# =============================
# HELPER FUNCTIONS
# =============================

MODEL_MAPPING = {
    "naive": NaiveLastValue,
    "random_walk": RandomWalkDrift,
    "rf": RandomForestPredictor,
    "lightgbm": LGBMPredictor,
    "xgboost": XGBPredictor,
}


def load_trained_model(model_type: str, horizon: int, ticker: str):
    """Load a trained model from experiments/models/ directory."""
    model_dir = "experiments/models"
    model_files = [
        f for f in os.listdir(model_dir)
        if f.startswith(f"{ticker}_{model_type}_h{horizon}")
    ]

    if not model_files:
        st.error(
            f"No trained model found for {ticker}, {model_type}, horizon={horizon}")
        return None

    # Load latest model file
    model_files.sort(reverse=True)
    model_path = os.path.join(model_dir, model_files[0])

    try:
        model_class = MODEL_MAPPING.get(model_type)
        if not model_class:
            st.error(f"Unsupported model type: {model_type}")
            return None
        model = model_class(horizon=horizon)
        model.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def plot_price_chart(df: pd.DataFrame, predictions: pd.Series = None):
    """Plot historical stock prices and predictions."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Adj Close"],
        mode="lines", name="Actual Price"
    ))

    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=predictions.index, y=predictions.values,
            mode="lines+markers", name="Predicted Price"
        ))

    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )
    return fig


# =============================
# SIDEBAR - USER INPUT
# =============================
st.sidebar.header("⚙️ Configuration")

ticker = st.sidebar.selectbox("Select Stock Ticker", config["data"]["tickers"])
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

horizon = st.sidebar.selectbox("Prediction Horizon (days)", config["horizons"])
model_type = st.sidebar.selectbox(
    "Select Model",
    ["naive", "random_walk", "rf", "lightgbm", "xgboost"]
)


# =============================
# MAIN APP LOGIC
# =============================
st.title("📈 Stock Price Prediction Dashboard")

try:
    # Load and clean data
    file_path = os.path.join(config["paths"]["data_raw"], f"{ticker}.csv")
    df_raw = load_raw_data(file_path)
    df_clean = clean_data(df_raw)

    # Filter by date
    df_filtered = df_clean.loc[str(start_date):str(end_date)]

    # Data quality check
    quality_ok = validate_data_quality(df_filtered)
    if quality_ok:
        st.success("✅ Data quality check passed")
    else:
        st.warning("⚠️ Potential data quality issues detected")

    # Create features
    df_features = create_all_features(df_filtered)

    # Load trained model
    model = load_trained_model(model_type, horizon, ticker)

    if model:
        # Prepare features and target
        target = df_features["Adj Close"].shift(-horizon)
        feature_cols = [col for col in df_features.columns if col not in [
            "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        X = df_features[feature_cols].fillna(0)

        # Predict on latest available data
        latest_features = X.iloc[[-1]]
        prediction = model.predict(latest_features)[0]

        st.subheader("🔮 Prediction")
        st.metric(
            label=f"Predicted Price ({horizon} days ahead)",
            value=f"${prediction:.2f}",
            delta=f"{((prediction - df_features['Adj Close'].iloc[-1]) / df_features['Adj Close'].iloc[-1] * 100):+.2f}%"
        )

        # Historical performance
        valid_mask = ~(X.isna().any(axis=1) | target.isna())
        X_clean, y_clean = X[valid_mask], target[valid_mask]
        preds = model.predict(X_clean)

        # Metrics
        metrics = evaluate_regression_model(
            y_clean.values, preds, model_name=model_type)
        st.subheader("📊 Model Metrics")
        st.json(metrics)

        # Trading metrics
        trading = evaluate_trading_strategy(y_clean.values, preds)
        st.subheader("💰 Trading Strategy Metrics")
        st.json(trading)

        # Plot actual vs predicted
        st.subheader("📉 Historical Predictions")
        preds_series = pd.Series(preds, index=y_clean.index)
        st.plotly_chart(plot_price_chart(
            df_filtered, preds_series), use_container_width=True)

    else:
        st.warning("No model available for the selected configuration.")

except FileNotFoundError:
    st.error(
        f"Data file for {ticker} not found in {config['paths']['data_raw']}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
