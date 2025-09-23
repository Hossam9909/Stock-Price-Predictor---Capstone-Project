# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.1.0] - 2024-05-24

### Added
- Complete project structure with modular `src`, `tests`, and `scripts`.
- Configuration-driven pipelines (`config/config.yaml`).
- Data ingestion and cleaning module (`src/data.py`).
- Comprehensive feature engineering module (`src/features.py`).
- Baseline (Naive, Random Walk) and advanced (RF, LGBM, XGB) models (`src/models.py`).
- Robust evaluation module with walk-forward validation (`src/evaluate.py`).
- Interactive Streamlit web application (`app/app.py`).
- Automation via `Makefile.mk` and `scripts/run_pipeline.py`.
- Full testing suite for core modules.
- Code quality enforcement with `pre-commit`, `black`, `flake8`, and `mypy`.