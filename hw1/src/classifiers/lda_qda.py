from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from .base import ClassifierBase
import numpy as np

class LDA(ClassifierBase):
    def __init__(self, solver="svd"):
        self.model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                                   LinearDiscriminantAnalysis(solver=solver))
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        est = self.model.named_steps.get("lineardiscriminantanalysis")
        try: return est.decision_function(self.model[:-1].transform(X))
        except: 
            proba = self.model.predict_proba(X)
            return proba[:,1] if proba.shape[1]==2 else proba

class QDA(ClassifierBase):
    def __init__(self, reg_param=0.01):
        self.model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    def fit(self, X, y): self.model.fit(X, y); return self
    def predict(self, X): return self.model.predict(X)
    def discriminant(self, X):
        try: return self.model.decision_function(X)
        except:
            proba = self.model.predict_proba(X)
            return proba[:,1] if proba.shape[1]==2 else proba
