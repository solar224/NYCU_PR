import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 匯入您專案中必要的模組
from src.datasets import load_dataset
from src.split import stratified_holdout
from src.main import get_models # 直接從 main.py 匯入 get_models 函式

def plot_roc_comparison(datasets, output_dir="reports/figures"):
    """
    為指定的資料集訓練所有模型，並在同一張圖上繪製 ROC 曲線進行比較。
    """
    if not datasets:
        print("No datasets specified.")
        return

    # 創建一個 1xN 的子圖，N 是資料集的數量
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 7))
    if len(datasets) == 1:
        axes = [axes] # 確保 axes 是一個可迭代的列表

    # 為每個資料集生成一個子圖
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]
        print(f"Processing dataset: {dataset_name}...")

        # 1. 載入並分割資料
        X, y, meta = load_dataset(dataset_name)
        Xtr, Xte, ytr, yte = stratified_holdout(X, y, test_size=0.2)

        # 檢查是否為二元分類問題
        if not meta.get("is_binary"):
            print(f"Skipping '{dataset_name}' as it is not a binary classification dataset.")
            ax.text(0.5, 0.5, 'Not a binary classification task', ha='center', va='center', fontsize=12)
            ax.set_title(f"ROC Curves - {dataset_name}", fontsize=14, weight='bold')
            continue

        # 2. 取得所有模型
        models = get_models(["svm", "knn", "logreg", "lda", "qda", "mlp", "rf"])

        # 3. 訓練模型並繪製 ROC 曲線
        for name, mdl in models.items():
            try:
                mdl.fit(Xtr, ytr)
                # 取得判別分數
                scores = mdl.discriminant(Xte)

                # 計算 ROC 曲線
                fpr, tpr, _ = roc_curve(yte, scores)
                roc_auc = auc(fpr, tpr)

                # 繪製
                ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Could not plot ROC for model '{name}' on dataset '{dataset_name}'. Reason: {e}")

        # 4. 美化圖表
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f"ROC Curve Comparison - {dataset_name}", fontsize=14, weight='bold')
        ax.legend(loc="lower right")
        ax.grid(True)

    # 調整整體佈局並儲存
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "roc_comparison_plot.png")
    plt.savefig(output_path)
    
    print(f"\nCombined ROC plot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a comparison plot of ROC curves for specified datasets.")
    parser.add_argument("--datasets", nargs='+', default=["pima_openml", "breast_cancer"], help="List of binary classification datasets to plot.")
    parser.add_argument("--output_dir", type=str, default="reports/figures", help="Directory where the output plot will be saved.")
    args = parser.parse_args()
    
    plot_roc_comparison(args.datasets, args.output_dir)

if __name__ == "__main__":
    main()