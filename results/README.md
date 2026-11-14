# results/ －實驗結果（規劃中）

這個資料夾之後會放 `evaluate.py` 與 `agent.py` 產生的實驗結果。**目前尚無任何模型實驗結果。**

## 已放入的檔案

| 檔案 | 內容 |
|---|---|
| `tsmc_crosscheck.csv` | 資料正確性核對（非實驗結果）：台積電 FinMind 數值與 Yahoo 股市逐項比對，8/8 一致，附來源網址。核對日期 2026-10-31 |

## 資料可行性驗證（已完成）

| 檔案 | 內容 |
|---|---|
| `verify_summary.md` | `verify_data.py` 的執行時間與結果總結 |
| `verify_sample_coverage.csv` | 抽樣 30 檔（上市中 20、已下市 10）的名單，以及每檔 7 種資料是否抓到、筆數、起訖日期 |
| `verify_dataset_summary.csv` | 各資料集在上市中與已下市公司的抓取率與最早日期 |
| `tsmc_crosscheck.csv` | 台積電 EPS、月營收、收盤價與 Yahoo 股市的核對結果 |

由 `verify_data.py` 產生（亂數種子 42），重新執行會得到相同的抽樣名單。

## 全市場資料收集（已完成）

| 檔案 | 內容 |
|---|---|
| `collect_summary.md` | `collect_data.py` 的執行期間、各資料集抓取率（上市中／已下市）、合併後列數與日期範圍、收集後發現的問題、全市場 Demo 總表統計。只有彙總數字，原始資料依 FinMind 授權不上傳 |
| `dataset_summary.csv` | 正式資料集（`build_dataset.py`，2026-11-14）各特徵年的統計：母體、上市未滿一年／興櫃／流動性排除筆數、流動性門檻、可排名筆數、`y_raw`、`y_main` 正例與 UNKNOWN 筆數、正例率。只有統計數字，不含原始資料 |
| `rule_thresholds.csv` | 兩條領域規則（`rules.py`）在三輪 walk-forward 的門檻（只用訓練特徵年的特徵分布）與訓練期觸發率、不適用率。不含標記或報酬 |
| `shares_crosscheck.csv` | 市值改用發行股數後（2026-11-14），與原本「股本 ÷ 10 再修正減資、分割」相差超過 20% 的 64 筆「公司 × 年份」：兩種算法的市值（億元）與比值。只有計算後的市值，不含原始資料 |
| `return_crosscheck.csv` | 含息報酬與 Yahoo 股市還原股價的抽樣核對（8 檔）：本專題算出的 5 年含息報酬（當時以 60 日終點平均計算，DESIGN v0.6 起正式改為 20 日；核對對象是公司行動因子，與終點天數無關）、只看股價報酬、Yahoo 還原股價報酬與差異原因。只有計算後的報酬數字，不含原始股價 |

## 資料管線 Demo 截圖（已完成）

| 檔案 | 內容 |
|---|---|
| `demo/1_target.png` | 分頁①「我們要找什麼樣的股票」：主要標記說明與自動挑出的三個例子 |
| `demo/2_time_machine.png` | 分頁②「時光機」：台積電 2020 年特徵日的 10 項特徵與 5 年後結果 |
| `demo/3_table.png` | 分頁③「30 檔總表」：每檔每年一列的特徵、報酬、營收 CAGR 與判定 |

由 `streamlit run app.py` 開啟網頁後截圖（2026-11-02），資料為 `verify_data.py` 的 30 檔抽樣快取加 0050。計算規則見 `DESIGN.md` 第 11.3 節。網頁現在有全市場資料時會改用全市場，截圖尚未更新。這是資料管線示範，不是模型實驗結果。

## 規劃中的實驗結果

| 實驗 | 預計檔案 | 內容 |
|---|---|---|
| E1 | `e1_ranking_metrics.csv` | 各模型與 Baseline（含重複隨機（1,000 次））的 Precision@20/50、Recall@50、PR-AUC |
| E2 | `e2_portfolio.csv`、`e2_portfolio.png` | 前 50 檔等權組合 vs 0050 的累積報酬與最大回撤 |
| E3 | `e3_shap_ranking.csv`、`e3_shap.png` | SHAP 特徵排名；主要正例、高報酬但未過成長門檻、其餘可判定者三組的特徵中位數與命中率，另按是否下市分層 |
| E4 | `e4_agent_eval.csv` | Agent 報告 vs 單純 LLM 的數字正確率、證據引用率；（延伸）使用者評分 |
| E5 | `e5_sensitivity.csv` | （延伸）觀察期 3 年 vs 5 年、主要 vs 對照標記、營收門檻 5% / 10% / 15% 的 Precision@50 比較 |

每次更新結果時，請同步更新 `DESIGN.md` 第 11 節與 Change Log。
