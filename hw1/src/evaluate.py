# src/evaluate.py
import matplotlib
matplotlib.use("Agg")
import os, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, f1_score

def save_confmat(y_true, y_pred, classes, out_png):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    fig.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)

def binary_roc_auc(y_true, scores, out_png):
    fpr, tpr, _ = roc_curve(y_true, scores)
    A = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={A:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    fig.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return A

def summarize_metrics(y_true, y_pred, scores=None, is_binary=False):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    out = {"accuracy": acc, "macro_f1": f1m}
    if is_binary and scores is not None and np.ndim(scores)==1:
        fpr, tpr, _ = roc_curve(y_true, scores); out["auc"] = auc(fpr, tpr)
    return out

def dump_table(row_dict, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        base = pd.read_csv(csv_path)
        df = pd.concat([base, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
