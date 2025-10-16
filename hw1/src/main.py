# src/main.py
import argparse, os, numpy as np
from src.datasets import load_dataset
from src.split import stratified_holdout
from src.evaluate import save_confmat, binary_roc_auc, summarize_metrics, dump_table
from src.classifiers.svm import SVM
from src.classifiers.knn import KNN
from src.classifiers.logistic import Logistic
from src.classifiers.lda_qda import LDA, QDA
from src.classifiers.mlp import MLP
from src.classifiers.rf import RF
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_models(names):
    out={}
    for n in names:
        if n=="svm": out[n]=SVM(kernel="rbf", C=1.0, gamma="scale", probability=False)
        elif n=="knn": out[n]=KNN(n_neighbors=5, weights="uniform")
        elif n=="logreg": out[n]=Logistic(C=1.0)
        elif n=="lda": out[n]=LDA(solver="svd")
        elif n=="qda": out[n]=QDA(reg_param=0.01)
        elif n=="mlp": out[n]=MLP(hidden_layer_sizes=(128,), alpha=1e-4, max_iter=500)
        elif n=="rf": out[n]=RF(n_estimators=300)
        else: raise ValueError(f"unknown model {n}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="breast_cancer|wine|iris|digits or path.csv:Target")
    ap.add_argument("--models", nargs="+", default=["svm","knn","logreg"])
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--cv", type=int, default=0, help="0=off, else k-fold")

    args = ap.parse_args()

    X, y, meta = load_dataset(args.dataset)
    Xtr, Xte, ytr, yte = stratified_holdout(X, y, test_size=0.2)
    if args.cv and args.cv > 1:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        for name, mdl in get_models(args.models).items():
            accs, f1s = [], []
            for tr, va in skf.split(X, y):
                mdl.fit(X[tr], y[tr])
                pred = mdl.predict(X[va])
                accs.append((pred==y[va]).mean())
                from sklearn.metrics import f1_score
                f1s.append(f1_score(y[va], pred, average="macro"))
            row = {"dataset": meta["name"], "model": name,
                   "cv": args.cv,
                   "cv_acc_mean": float(np.mean(accs)), "cv_acc_std": float(np.std(accs)),
                   "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s))}
            dump_table(row, os.path.join(args.outdir, "tables", f"{meta['name']}_cv.csv"))
            print(row)

    models = get_models(args.models)
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        yhat = mdl.predict(Xte)
        disc = mdl.discriminant(Xte)

        fig_dir = os.path.join(args.outdir, "figures")
        save_confmat(yte, yhat, classes=[str(i) for i in sorted(set(y))],
                     out_png=os.path.join(fig_dir, f"{meta['name']}_{name}_confmat.png"))
        aucv=None
        if meta["is_binary"] and disc.ndim==1:
            aucv = binary_roc_auc(yte, disc, out_png=os.path.join(fig_dir, f"{meta['name']}_{name}_roc.png"))

        row = {"dataset": meta["name"], "model": name, **summarize_metrics(yte, yhat, disc if disc.ndim==1 else None, meta["is_binary"])}
        dump_table(row, os.path.join(args.outdir, "tables", f"{meta['name']}.csv"))
        print(row)

if __name__=="__main__":
    main()
