```
pr-assignment1/
  data/                      # 
  src/
    datasets.py              # 載入前處理與特徵工程
    split.py                 # Stratified split + CV
    classifiers/
      base.py                # interface define
      knn.py                 # k-NN
      svm.py                 # SVM (linear/RBF)
      logistic.py            # 邏輯迴歸
      lda_qda.py             # LDA/QDA
      mlp.py                 # MLPClassifier 
      rf.py                  # RandomForest 
    evaluate.py              # 混淆矩陣、ROC、AUC、報表
    experiment.py            # 實驗分析
    main.py                  # 
  reports/
    figures/                 # 圖片輸出：混淆矩陣、ROC、學習曲線
    tables/                  # CSV
  README.md
```
