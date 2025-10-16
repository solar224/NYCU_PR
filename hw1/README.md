pr-assignment1/
  data/                      # 放本地快取
  src/
    datasets.py              # 載入/前處理與特徵工程
    split.py                 # Stratified split + CV
    classifiers/
      base.py                # 介面定義
      knn.py                 # k-NN
      svm.py                 # SVM (linear/RBF)
      logistic.py            # 邏輯迴歸
      lda_qda.py             # LDA/QDA
      mlp.py                 # MLPClassifier (允許)
      rf.py                  # RandomForest (可選)
    evaluate.py              # 混淆矩陣、ROC、AUC、報表
    experiment.py            # 單一資料集多模型多超參數
    main.py                  # 入口：挑資料集→跑實驗→存圖與結果
  reports/
    figures/                 # 圖片輸出：混淆矩陣、ROC、學習曲線
    tables/                  # 指標彙整 CSV
  README.md
