.PHONY: test setup install data features train evaluate all lint format clean help app

# Python and environment settings
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest

# Default configuration file
CONFIG_FILE := config/config.yaml

help:
	@echo ""
	@echo "Available commands:"
	@echo "  help       Show this help message"
	@echo "  setup      Create directories and setup environment"
	@echo "  install    Install dependencies"
	@echo "  test       Run all tests with coverage"
	@echo "  lint       Run linting checks"
	@echo "  format     Format code using black and isort"
	@echo "  data       Download and process data"
	@echo "  features   Generate features"
	@echo "  train      Train all models"
	@echo "  evaluate   Evaluate models and generate reports"
	@echo "  all        Run complete pipeline"
	@echo "  app        Run the Streamlit web application"
	@echo "  clean      Clean generated files"
	@echo ""

setup:
	mkdir -p data/raw data/processed data/external
	mkdir -p experiments/models experiments/results experiments/figures
	mkdir -p logs
	@echo "✅ Project directories created"

install:
	@echo "--- Setting up virtual environment and installing dependencies ---"
	@test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed into $(VENV_DIR)"

test:
	@echo "--- Running tests ---"
	@$(PYTEST) tests/ -v --tb=short --disable-warnings

test-coverage:
	@$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "--- Running linters ---"
	@$(PIP) install -q flake8 mypy
	@flake8 src/ tests/
	@mypy src/

format:
	@echo "--- Formatting code ---"
	@$(PIP) install -q black isort
	@black src/ tests/
	@isort src/ tests/

data:
	@$(PYTHON) scripts/run_pipeline.py --step data --config $(CONFIG_FILE)

features:
	@$(PYTHON) scripts/run_pipeline.py --step features --config $(CONFIG_FILE)

train:
	@$(PYTHON) scripts/run_pipeline.py --step train --config $(CONFIG_FILE)

evaluate:
	@$(PYTHON) scripts/run_pipeline.py --step evaluate --config $(CONFIG_FILE)

all: setup data features train evaluate

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf data/raw/*.csv logs/*.log
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

app:
	@$(PYTHON) -m streamlit run app/app.py