# Makefile for Stock Price Predictor Project

.PHONY: help setup install clean data features train evaluate all test lint format

# Default target
help:
	@echo "Available targets:"
	@echo "  setup     - Create directories and setup environment"
	@echo "  install   - Install dependencies"
	@echo "  clean     - Clean generated files and cache"
	@echo "  data      - Download and process data"
	@echo "  features  - Generate features"
	@echo "  train     - Train all models"
	@echo "  evaluate  - Evaluate models and generate reports"
	@echo "  all       - Run complete pipeline"
	@echo "  test      - Run tests"
	@echo "  lint      - Run linting"
	@echo "  format    - Format code"
	@echo "  app       - Run Streamlit app"

# Setup project structure
setup:
	@echo "Setting up project structure..."
	mkdir -p data/raw data/processed data/external
	mkdir -p experiments/models experiments/results experiments/figures
	mkdir -p logs
	mkdir -p config
	mkdir -p tests

# Install dependencies
install:
	pip install -r requirements.txt

# Create conda environment
env-create:
	conda env create -f environment.yml

# Activate conda environment
env-activate:
	conda activate stock-predictor

# Clean generated files
clean:
	rm -rf data/processed/*
	rm -rf experiments/*
	rm -rf logs/*
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete

# Data pipeline steps
data:
	@echo "Running data ingestion..."
	python src/data.py

features:
	@echo "Running feature engineering..."
	python src/features.py

train:
	@echo "Training models..."
	python src/models.py

evaluate:
	@echo "Evaluating models..."
	python src/evaluate.py

# Run complete pipeline
all: data features train evaluate

# Testing
test:
	pytest tests/ -v

# Code quality
lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Run Streamlit app
app:
	streamlit run app/app.py

# Install pre-commit hooks
pre-commit-install:
	pre-commit install

# Run pre-commit hooks
pre-commit:
	pre-commit run --all-files