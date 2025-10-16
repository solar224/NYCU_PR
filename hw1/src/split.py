# src/split.py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42

def stratified_holdout(X, y, test_size=0.2, rs=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=rs)

def stratified_kfold(n_splits=5, rs=RANDOM_STATE):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
