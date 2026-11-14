# 文獻筆記：外資持股與報酬（列 Future Work）

> 2026-11-14 讀完全文。2026-11-14 決定**不加入**，列為 Future Work（見 DESIGN 2.3）。這份是讀書筆記。

## Huang, R. D., & Shiu, C.-Y. (2009)

Local Effects of Foreign Ownership in an Emerging Financial Market: Evidence from Qualified Foreign Institutional Investors in Taiwan. *Financial Management*, 38(3), 567–602.

| 項目 | 原文 |
|---|---|
| 樣本 | 台股（TSE），1994 Q3–2001 年，523 家；上市滿 6 個月；資料來源 TEJ（p. 572） |
| 變數 | 外資（QFII）持股 ÷ 流通在外股數，取**前一季底**（Table III 說明） |
| 分組 | 每季依外資持股比例分五組，持有 3 個月後重新分組（p. 578） |
| 主要結果 | 最高組減最低組的四因子 alpha：等權每月 1.22%（t = 3.19）、市值加權每月 1.75%（t = 3.84）（Table III） |
| 穩健性 | 控制出口比例、規模、資訊透明度後仍存在（p. 582–584，結果未列表） |
| 經營績效 | 前一年外資持股與之後的 ROA、Tobin's Q、研發增加都顯著正相關（Table IV、V） |
| 作者解釋 | 外資的長期資訊優勢與監督角色 |
| 作者提到的疑慮 | 「假外資」：本地投資人透過境外公司買進，看起來像外資（p. 597）；外資不能放空可能也有影響（p. 584） |

## 用在本專題要注意

1. **持有期間不同**：原文是每季換股、看之後 3 個月的報酬；本專題看 5 年。原文的經營績效結果（之後一年 ROA 較高）比較接近長期，但仍不是 5 年報酬的直接證據。
2. **變數口徑不同**：原文只算 QFII；FinMind `TaiwanStockShareholding` 的 `ForeignInvestmentSharesRatio` 是全體外資（含陸資以外的所有外國投資人），且 QFII 制度後來已取消，無法完全對應。
3. **樣本年代**：1994–2001 年，外資持股普遍很低（平均 2.2%，Table I）；2013–2020 年外資持股高很多，效果大小可能不同。
4. 搜尋時沒有找到 2010 年後、看台股長期報酬的同儕審查研究（只找到碩博士論文）。

## 若之後要固定成特徵（建議口徑）

外資持股比例 ＝ 特徵日（4 月底）當天或之前最後一筆 `ForeignInvestmentShares ÷ NumberOfSharesIssued`；只用特徵日以前的資料；放進 E1 消融比較，不用結果挑選。
