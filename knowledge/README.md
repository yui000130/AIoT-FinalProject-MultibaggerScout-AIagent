# knowledge/ －規則說明與知識庫

放每條領域規則的固定說明、門檻依據與文獻出處，Agent 在報告中解釋規則時直接引用。（RAG 知識庫不納入本專題，列為 Future Work。）

## 目前內容

| 檔案 | 內容 | 程式 |
|---|---|---|
| `rule_earnings_quality.md` | 規則一：盈餘品質紅旗（應計占平均總資產偏高），Sloan (1996) | `rules.check_earnings_quality` |
| `feature_evidence.md` | 10 項特徵的近年文獻依據（JKP 2023、Bui 等 2023 台股、Bessembinder 2020 長期極端贏家） | — |
| `lit_foreign_ownership.md` | 文獻筆記：Huang & Shiu (2009) 外資持股與報酬（列 Future Work） | — |
| `rule_overvaluation.md` | 規則二：估值過高（帳面／市值極低或本益比極高），Lakonishok et al. (1994) | `rules.check_overvaluation` |

門檻只用各輪訓練特徵年的特徵分布計算（不看標記），結果見 `results/rule_thresholds.csv`（執行 `python rules.py` 產生）。

## 之後可加入

- 多倍股相關文獻重點整理（例如 Yartseva 2025、Bessembinder et al. 2023）
- 財務指標定義與計算方式（ROE、FCF 殖利率、營業現金流／淨利等）

## 撰寫規範

- 一份文件一個主題，使用 Markdown
- 每份文件註明出處與查證狀態（是否讀過全文）
- 只放可公開分享的內容，不放受版權限制的全文

