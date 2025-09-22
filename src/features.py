"""Feature engineering helpers."""
import pandas as pd

def make_features(df, lags=[1,2,3,5], roll_windows=[7,14]):
    d = df.copy()
    for lag in lags:
        d[f'lag_{lag}'] = d['AdjClose'].shift(lag)
    for w in roll_windows:
        d[f'rolmean_{w}'] = d['AdjClose'].shift(1).rolling(window=w).mean()
        d[f'rolstd_{w}'] = d['AdjClose'].shift(1).rolling(window=w).std()
    d['ret_1'] = d['AdjClose'].pct_change(1)
    d['dayofweek'] = d.index.dayofweek
    return d.dropna()
