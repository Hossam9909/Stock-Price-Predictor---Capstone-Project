"""Simple model wrappers for training and inference."""
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class SklearnWrapper:
    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
