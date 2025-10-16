from sklearn.ensemble import RandomForestClassifier
from .base import ClassifierBase

class RF(ClassifierBase):
    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            random_state=random_state, n_jobs=-1)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        proba = self.model.predict_proba(X)
        return proba[:,1] if proba.shape[1]==2 else proba
