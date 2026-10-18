# HW2：多元線性回歸（Multiple Linear Regression）

邱勇嘉 · 5115056022 · 依 CRISP-DM 流程，以兩個 Kaggle 資料集完成多元線性回歸的特徵選擇、模型評估與預測區間分析。

- **線上儀表板（GitHub Pages）**：https://hobartxiii.github.io/HW2-Multiple-Linear-regression/
- **主程式**：[`5115056022_hw2.ipynb`](5115056022_hw2.ipynb)（完整分析，所有輸出已保留在 notebook 中）
- **報告**：[`report/5115056022_hw2_report.pdf`](report/5115056022_hw2_report.pdf)（CRISP-DM 分析流程、GPT 輔助內容、NotebookLM 研究摘要、網路主流解法比較；附錄為 GPT 對話過程）
- **GPT 對話過程**：[`report/5115056022_hw2_GPT對話過程.pdf`](report/5115056022_hw2_GPT對話過程.pdf)（ChatGPT 與 Claude Code 的對話，與報告附錄相同）

## 資料集

| 題目 | 資料集 | 特徵數 | 目標 |
|---|---|---|---|
| A | [Kaggle Playground S4E5 — Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5)（檔案取自 [naiyakhalid/flood-prediction-dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)） | 20（數值） | `FloodProbability` |
| B | [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) | 19（6 數值＋13 類別） | `Exam_Score` |

題目 A 原本考慮 50,000 筆的 `flood.csv`，但檢查發現其目標值是 20 個特徵的精確線性組合（OLS 測試 R² = 1.000），不具分析意義，因此改用競賽的 `train.csv`（1,117,957 筆）。

## 分析流程（CRISP-DM）

1. **Business Understanding**：A 預測地區洪水機率，協助防災資源排序；B 預測學生期末考成績，找出可介入的影響因素。
2. **Data Understanding**：缺失值、敘述統計、目標分布、相關係數；B 的類別特徵以箱型圖比較。
3. **Data Preparation**：80/20 切分（seed 42）；只用訓練集計算中位數／眾數補值與 one-hot 編碼（drop first）。
4. **Modeling**：比較 5 種特徵選擇方法（SelectKBest、RFE、Sequential Forward、Lasso path、Permutation importance），以 5-fold CV RMSE 選出「誤差在最佳值容許範圍內的最少特徵」；再比較 OLS（全部／選出特徵）、RidgeCV、LassoCV。
5. **Evaluation**：MAE、MSE、RMSE、R²、Adjusted R²、CV；預測圖附 95% 信賴區間與預測區間；檢查線性回歸五大假設（殘差圖、Durbin-Watson、Breusch-Pagan、Jarque-Bera、VIF）。
6. **Deployment**：模型存檔與新資料預測（含 CI/PI）、Kaggle 提交、互動式網頁。

## 主要結果（notebook）

| | 題目 A：Flood | 題目 B：Student |
|---|---|---|
| 特徵選擇 | 20 個全部保留（每個特徵貢獻相同） | 27 個編碼欄位保留 21 個 |
| 測試集 R² / RMSE | 0.845 / 0.020 | 0.764 / 1.83 分 |
| 95% PI 實際覆蓋率 | 94.6% | 99.5%（受少數高分離群值影響而偏寬） |
| Kaggle | Public 0.84458 · Private 0.84426（約 1,850 / 2,794 名） | — |

- **A**：20 個係數全部約為 0.00565，洪水機率幾乎等於「20 項風險評分總和」的線性函數；排行榜分數呈兩群，0.8446 附近正是線性模型的上限，前段解法以「特徵總和」等工程特徵＋GBDT 達到 0.869。
- **B**：98% 的測試殘差落在 ±1.3 分內，誤差幾乎全部來自約 0.5% 的高分離群學生；出席率、讀書時數、資源取得、家長參與為最主要因素。

## 互動網頁（`web/`）

純前端靜態網站：Tailwind CSS ＋ Flowbite ＋ ApexCharts ＋ PapaParse，回歸運算全部在瀏覽器中完成。

- 下拉選單切換兩個題目；可輸入任意 CSV 網址（例：[50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)）或拖曳上傳。
- 四個頁面：資料總覽（敘述統計、相關係數熱力圖）、模型結果（特徵選擇、係數）、評估（預測圖＋CI/PI、殘差圖、五大假設）、預測（即時預測與區間）。
- 網址參數：`?problem=flood`、`?problem=student`、`?url=<CSV 網址>`。
- 網頁的 Flood 資料為 20,000 筆抽樣，且為另一次隨機切分，數字與 notebook 略有差異；報告以 notebook 為準。
- 正確性驗證：`web/tests/compare_statsmodels.py` 以相同切分比對 statsmodels，係數、標準誤、p 值、CI/PI、R²、DW、JB、BP、VIF 的最大相對誤差約 2e-12（測試門檻 1e-6）。

## 重現方式

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn kaggle
kaggle auth login
kaggle datasets download -d naiyakhalid/flood-prediction-dataset -p data/flood-prediction-dataset --unzip
kaggle datasets download -d lainguyn123/student-performance-factors -p data/student-performance-factors --unzip
jupyter nbconvert --to notebook --execute 5115056022_hw2.ipynb
```

本機預覽網頁：`cd web && python -m http.server`，開啟 http://localhost:8000 。
