# src/classifiers/svm.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .base import ClassifierBase

class SVM(ClassifierBase):
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42):
        self.model = make_pipeline(StandardScaler(),
                                   SVC(kernel=kernel, C=C, gamma=gamma,
                                       probability=probability, random_state=random_state))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        svc = self.model.named_steps["svc"]
        return svc.decision_function(self.model[:-1].transform(X)) if hasattr(svc,"decision_function") else self.model.predict_proba(X)[:,1]
