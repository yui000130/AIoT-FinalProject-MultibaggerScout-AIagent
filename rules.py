"""
rules.py －領域規則檢查（DESIGN.md 第 7 節）

每條規則接收單一公司在特徵日的特徵，回傳：
  {"rule": 規則名稱, "triggered": 是否觸發（None ＝ 不適用）, "evidence": 用到的數字與門檻,
   "reason": 一句話說明, "doc": knowledge/ 中的固定說明}
結果作為報告中的反對證據或注意事項，不改變模型分數。共 2 條規則，都只用 build_dataset.py 的特徵判斷。

門檻（2026-11-14 事先固定，依文獻採十分位極端組）：
  - 只用**訓練視窗特徵年**的特徵分布計算（不使用任何標記或報酬），每輪 walk-forward 各算一次
  - 盈餘品質紅旗：應計 ÷ 平均總資產 ≥ 訓練分布第 90 百分位（Sloan, 1996 的最高應計組；2026-11-14 由營業現金流／淨利改為原文口徑）
  - 估值過高：帳面／市值 ≤ 訓練分布第 10 百分位，或本益比（> 0）≥ 第 90 百分位（Lakonishok et al., 1994 的魅力股組）
固定說明與文獻出處見 knowledge/rule_earnings_quality.md、knowledge/rule_overvaluation.md。

輸入：單一公司當期特徵（dict 或 pandas.Series，欄名同 data/dataset.parquet）＋ compute_thresholds() 的門檻
輸出：規則結果清單
"""

import math

import pandas as pd

ACCRUAL = "應計/平均總資產"   # 規則輸入，不是模型特徵（build_dataset.accrual_ratio）
BM = "帳面/市值"
PE = "本益比"
LOW_PCT, HIGH_PCT = 0.10, 0.90   # 十分位極端組（事先固定）
DOCS = {"盈餘品質紅旗": "knowledge/rule_earnings_quality.md", "估值過高": "knowledge/rule_overvaluation.md"}


def compute_thresholds(train_rows):
    """只用訓練視窗的特徵（不看標記）計算門檻。train_rows：訓練特徵年的 DataFrame。"""
    acc = train_rows[ACCRUAL].dropna()
    pe = train_rows[PE].dropna()
    pe = pe[pe > 0]
    bm = train_rows[BM].dropna()
    return {
        "訓練特徵年": sorted(int(y) for y in train_rows["特徵年"].unique()),
        f"{ACCRUAL} 第 90 百分位": float(acc.quantile(HIGH_PCT)),
        f"{BM} 第 10 百分位": float(bm.quantile(LOW_PCT)),
        f"{PE} 第 90 百分位": float(pe.quantile(HIGH_PCT)),
        "樣本數": {ACCRUAL: len(acc), BM: len(bm), PE: len(pe)},
    }


def _val(features, k):
    v = features.get(k) if isinstance(features, dict) else features[k]
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)


def check_earnings_quality(features, th):
    """盈餘品質紅旗：應計（帳上獲利但尚未收到現金的部分）占總資產的比例落在訓練期最高 10%。"""
    name, cut = "盈餘品質紅旗", th[f"{ACCRUAL} 第 90 百分位"]
    v = _val(features, ACCRUAL)
    out = {"rule": name, "doc": DOCS[name], "evidence": {ACCRUAL: v, "門檻（訓練期第 90 百分位）": round(cut, 3)}}
    if v is None:
        return {**out, "triggered": None, "reason": "金融保險業或缺財報資料，規則不適用"}
    if v >= cut:
        return {**out, "triggered": True,
                "reason": f"應計占平均總資產 {v:.1%}，落在訓練期最高 10%（≥ {cut:.1%}），獲利中尚未收到現金的部分偏高"}
    return {**out, "triggered": False, "reason": f"應計占平均總資產 {v:.1%}，低於訓練期第 90 百分位 {cut:.1%}"}


def check_overvaluation(features, th):
    """估值過高：帳面／市值極低，或本益比極高（魅力股）。"""
    name = "估值過高"
    bm_cut, pe_cut = th[f"{BM} 第 10 百分位"], th[f"{PE} 第 90 百分位"]
    bm, pe = _val(features, BM), _val(features, PE)
    out = {"rule": name, "doc": DOCS[name],
           "evidence": {BM: bm, PE: pe, f"{BM} 門檻（第 10 百分位）": round(bm_cut, 3),
                        f"{PE} 門檻（第 90 百分位）": round(pe_cut, 1)}}
    if bm is None and pe is None:
        return {**out, "triggered": None, "reason": "帳面／市值與本益比皆缺值，規則不適用"}
    hits = []
    if bm is not None and bm <= bm_cut:
        hits.append(f"帳面／市值 {bm:.3f} ≤ {bm_cut:.3f}（訓練期最低 10%）")
    if pe is not None and pe >= pe_cut:
        hits.append(f"本益比 {pe:.1f} ≥ {pe_cut:.1f}（訓練期最高 10%）")
    if hits:
        return {**out, "triggered": True, "reason": "；".join(hits)}
    return {**out, "triggered": False, "reason": "帳面／市值與本益比都不在訓練期的極端 10%"}


def run_all_rules(features, th):
    """執行所有規則，回傳結果清單。"""
    return [check_earnings_quality(features, th), check_overvaluation(features, th)]


if __name__ == "__main__":
    # 每輪 walk-forward 的門檻與訓練期觸發率（只看特徵分布，不看標記）
    from pathlib import Path
    base = Path(__file__).resolve().parent
    data = pd.read_parquet(base / "data" / "dataset.parquet")
    rounds = {1: [2013], 2: [2013, 2014], 3: [2013, 2014, 2015]}
    rows = []
    for r, years in rounds.items():
        train = data[data["特徵年"].isin(years)]
        th = compute_thresholds(train)
        res = [run_all_rules(f, th) for _, f in train.iterrows()]
        rate = lambda k: (sum(x[k]["triggered"] is True for x in res) / len(res))
        na = lambda k: (sum(x[k]["triggered"] is None for x in res) / len(res))
        rows.append({"輪次": r, "訓練特徵年": "、".join(map(str, years)), "訓練列數": len(train),
                     f"{ACCRUAL} 第 90 百分位": round(th[f"{ACCRUAL} 第 90 百分位"], 3),
                     f"{BM} 第 10 百分位": round(th[f"{BM} 第 10 百分位"], 3),
                     f"{PE} 第 90 百分位": round(th[f"{PE} 第 90 百分位"], 1),
                     "盈餘品質紅旗觸發率": round(rate(0), 3), "盈餘品質不適用率": round(na(0), 3),
                     "估值過高觸發率": round(rate(1), 3), "估值過高不適用率": round(na(1), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(base / "results" / "rule_thresholds.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
