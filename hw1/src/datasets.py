# src/datasets.py
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.datasets import fetch_openml

def _from_sklearn(loader, name):
    d = loader()
    X = d["data"].astype(float)
    y = d["target"].astype(int)
    meta = dict(name=name, n_classes=len(np.unique(y)),
                is_binary=len(np.unique(y))==2,
                feature_names=getattr(d, "feature_names", [f"f{i}" for i in range(X.shape[1])]))
    return X, y, meta
def _from_openml_pima():
    df = fetch_openml(name="diabetes", version=1, as_frame=True).frame 
    y = (df['class'] == 'tested_positive').astype(int).to_numpy()
    X = df.drop(columns=['class']).astype(float).to_numpy()
    meta = dict(name="pima_openml", n_classes=2, is_binary=True, feature_names=list(df.columns.drop('class')))
    return X, y, meta

def _from_csv(path, target_col):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in CSV columns: {list(df.columns)}")
    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[target_col]).astype(float).to_numpy()
    meta = dict(name=path, n_classes=len(np.unique(y)),
                is_binary=len(np.unique(y))==2,
                feature_names=[c for c in df.columns if c != target_col])
    return X, y, meta

def load_dataset(name: str):
    if ":" in name:
        path, target = name.split(":", 1)
        if path.lower().endswith(".csv"):
            return _from_csv(path, target)

    key = name.lower()
    if key in ["breast_cancer", "cancer", "bc"]:
        return _from_sklearn(load_breast_cancer, "breast_cancer")
    if key == "wine":
        return _from_sklearn(load_wine, "wine")
    if key == "iris":
        return _from_sklearn(load_iris, "iris")
    if key == "digits":
        return _from_sklearn(load_digits, "digits")
    if key in ["pima", "pima_openml", "diabetes_pima"]:
        return _from_openml_pima()

    raise ValueError(f"Unknown dataset: {name}")
