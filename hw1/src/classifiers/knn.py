# src/classifiers/knn.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from .base import ClassifierBase
import numpy as np

class KNN(ClassifierBase):
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.model = make_pipeline(StandardScaler(),
                                   KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        proba = self.model.predict_proba(X)
        return proba[:,1] if proba.shape[1]==2 else proba  # 二分類回傳陽性機率
