# src/classifiers/logistic.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .base import ClassifierBase

class Logistic(ClassifierBase):
    def __init__(self, C=1.0, max_iter=1000, solver="lbfgs", class_weight=None):
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, solver=solver, class_weight=class_weight)
        )
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        proba = self.model.predict_proba(X)
        return proba[:,1] if proba.shape[1]==2 else proba
