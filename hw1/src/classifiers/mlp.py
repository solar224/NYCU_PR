from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from .base import ClassifierBase

class MLP(ClassifierBase):
    def __init__(self, hidden_layer_sizes=(128,), alpha=1e-3, max_iter=1000, random_state=42):
        self.model = make_pipeline(StandardScaler(),
                                   MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                 alpha=alpha, max_iter=max_iter,
                                                 early_stopping=True,
                                                 n_iter_no_change=20,
                                                 random_state=random_state))

    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        # MLP 常見只用 predict_proba
        proba = self.model.predict_proba(X)
        return proba[:,1] if proba.shape[1]==2 else proba
