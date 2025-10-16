# src/experiment.py
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from src.datasets import load_dataset
from src.classifiers.svm import SVM
from src.classifiers.knn import KNN
from src.classifiers.rf import RF
from src.classifiers.mlp import MLP

def get_model_and_param_grid(name):
    if name == "svm":
        model = SVM()
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 'auto', 0.01, 0.1],
            'svc__kernel': ['rbf', 'linear']
        }
    elif name == "knn":
        model = KNN()
        param_grid = {
            'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
            'kneighborsclassifier__weights': ['uniform', 'distance']
        }
    elif name == "rf":
        model = RF()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
        }
    elif name == "mlp":
        model = MLP()
        param_grid = {
            'mlpclassifier__hidden_layer_sizes': [(64,), (128,), (64, 32)],
            'mlpclassifier__alpha': [1e-5, 1e-4, 1e-3],
        }
    else:
        raise ValueError(f"Unknown model for GridSearchCV: {name}")
    return model.model, param_grid
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Name of the dataset to use.")
    ap.add_argument("--model", required=True, help="Model to tune (e.g., svm, knn, rf, mlp).")
    ap.add_argument("--scoring", default="accuracy", help="Scoring metric for GridSearchCV.")
    ap.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    args = ap.parse_args()

    print(f"Loading dataset: {args.dataset}")
    X, y, _ = load_dataset(args.dataset)

    print(f"Initializing model and parameter grid for: {args.model}")
    model, param_grid = get_model_and_param_grid(args.model)

    print(f"Starting GridSearchCV with {args.cv}-fold CV...")
    grid_search = GridSearchCV(model, param_grid, cv=args.cv, scoring=args.scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    print("\n--- Grid Search Results ---")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best {args.scoring} score: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    main()