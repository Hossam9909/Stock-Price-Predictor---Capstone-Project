"""Simple Streamlit app prototype to load a model and show predictions."""
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.title('Stock Price Indicator — Demo')

st.sidebar.write('Model & data inputs')
model_path = st.sidebar.text_input('Model path', 'models/rf_example.pkl')
ticker = st.sidebar.text_input('Ticker', 'AAPL')
date_input = st.sidebar.date_input('Query date', datetime.today())

if st.button('Load model and predict'):
    try:
        model = joblib.load(model_path)
        st.success('Model loaded.')
        st.write('This demo requires a prepared feature vector for the query date.')
        st.write('Use the notebooks to prepare feature vectors; this app loads the model and would accept features for prediction.')
    except Exception as e:
        st.error(f'Failed to load model: {e}')
