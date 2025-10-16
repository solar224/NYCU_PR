# src/classifiers/base.py
class ClassifierBase:
    def fit(self, X_train, y_train): raise NotImplementedError
    def predict(self, X_test): raise NotImplementedError
    def discriminant(self, X_test): raise NotImplementedError  # decision score 或機率
