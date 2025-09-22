"""Evaluation utilities: simple metrics and a rolling split helper."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rolling_splits(dates, n_splits=5, test_size=90):
    """Yield (train_start, train_end, test_start, test_end) tuples as timestamps."""
    dates = pd.to_datetime(dates).sort_values()
    n = len(dates)
    step = max(1, (n - test_size) // n_splits)
    for i in range(0, n - test_size, step):
        train_start = dates[0]
        train_end = dates[i + step - 1]
        test_start = dates[i + step]
        test_end = dates[i + step + test_size - 1] if (i + step + test_size -1) < n else dates[-1]
        yield train_start, train_end, test_start, test_end
