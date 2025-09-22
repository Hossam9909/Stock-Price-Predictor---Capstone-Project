# Stock Price Indicator — Capstone Project

This repository contains code and notebooks for a Capstone Project that predicts the **Adjusted Close** price of stocks at multiple future horizons (1, 7, 14, 28 days). The goal is to demonstrate the full data science lifecycle: data ingestion, exploratory data analysis, feature engineering, model building, validation with walk-forward evaluation, and presentation of results. Optionally a simple Streamlit web UI is provided for making predictions locally.

## Quick start

1. Clone the repo:
```bash
git clone <your-repo-url>
cd capstone-stock-predictor
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows (PowerShell)

pip install -r requirements.txt
```

3. Download sample data:
```bash
python -m src.data --download --tickers AAPL MSFT GOOG --start 2015-01-01 --end 2024-12-31
```

4. Run notebooks in `notebooks/` top-to-bottom.

## Repo structure

```
capstone-stock-predictor/
├─ data/                  # scripts to fetch data + small sample CSVs (do NOT commit large raw files)
├─ notebooks/             # analysis and modeling notebooks
├─ src/                   # reusable code: data, features, models, evaluation
├─ app/                   # optional Streamlit app prototype
├─ experiments/           # saved experiment results, model artifacts (gitignored)
├─ requirements.txt
├─ .gitignore
├─ README.md
└─ report/                # final write-up or blog post markdown
```

## Deliverables checklist

- [ ] Project Definition: overview, problem statement, metrics.
- [ ] Analysis: EDA notebooks + visualizations.
- [ ] Methodology: feature engineering & models documented.
- [ ] Results: evaluation tables/plots, walk-forward validation.
- [ ] Conclusion: reflection, limitations, improvements.
- [ ] Repo: README, notebooks, src code, and report.

## Notes

- Use `yfinance` or another public data source.
- Avoid lookahead leakage; use walk-forward validation.
- This project is educational and not financial advice.
