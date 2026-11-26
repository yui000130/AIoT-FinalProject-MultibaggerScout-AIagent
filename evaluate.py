"""
evaluate.py －回測與 Baseline 比較

執行 DESIGN.md 第 11.5 節的實驗：
  E1：預測主要標記 y_main 的排序品質（2026-11-18 完成）
      方法：LightGBM、邏輯迴歸、隨機（1,000 次）、低 P/B、高 ROE、Yartseva 啟發規則
      指標：各輪 Precision@20/50、Recall@50、PR-AUC；基準為同輪可評估母體的實際正例率
  E2：每輪前 50 檔等權組合 vs 0050 買進持有（2026-11-18 完成）
      選股母體：測試年全部評分樣本（含 UNKNOWN，選股時不可能知道標記）；同分依代號
      進場：特徵日 t 的下一交易日收盤；出場：第 y + 5 年 4 月最後交易日收盤；0050 同規則
      進場等權、買進持有不再平衡；股利再投入同一檔；期間下市以最後交易價變現後持有現金
      對照：邏輯迴歸前 50、隨機 50 檔（種子 42，1,000 次）；指標為毛報酬（未計交易成本）與最大回撤
  E3：特徵關聯（2026-11-18 完成）：2013–2020 y_main 可判定樣本
      五等分正例率：10 項特徵各自在每個特徵年內分五等分、缺值另成一組；8 年合併正例率與「Q5 高於 Q1 的年數」
      三組中位數：主要正例、高報酬但營收未過門檻、其餘可判定，再依期間下市分層；附 0050 背景
      SHAP：LightGBM 內建 pred_contrib（TreeSHAP），三輪測試年的平均 |SHAP| 排名與方向（Spearman）
  E5（延伸）：敏感度分析（2026-11-26 完成；方法見 DESIGN 11.5 E5 方法定義）
      4 種標記：y_raw、營收門檻 5% / 10%（＝ y_main）/ 15%；每種各自依三輪切分重訓 LightGBM（參數同 train.py）
      指標：該標記可判定母體上的 Precision@20/50、Recall@50、PR-AUC 與各自的隨機基準（種子 42，1,000 次）
      執行：python -c "import evaluate; evaluate.run_e5()"（需先執行 python train.py 與 E1）
      輸出：results/e5_sensitivity.csv
（E4 為 Agent 報告評估，見 agent.py）

E1 方法定義（DESIGN 11.5，執行前固定）：
  - 可評估母體：測試年 y_main 可判定的列；所有方法對同一母體排序，同分依代號
  - 單一因子缺值者排最後；Yartseva 規則分數只平均有值的項目，篩選條件缺值視為未通過
  - Yartseva 啟發規則：帳面／市值 > 0.40 且 ROE > 0 者在前；分數 ＝ 當年百分位排名等權平均
    （市值小、FCF 殖利率高、帳面／市值高、過去 12 個月含息報酬低），見 knowledge/lit_yartseva.md
  - 百分位排名只用測試年同一特徵日的全部樣本特徵（特徵日已公開），不用標記

輸入：data/dataset.parquet、models/scores_round{n}.parquet（train.py）
  E4：報告品質（Agent vs 直接 LLM，2026-11-24 實作；方法見 DESIGN 11.5 E4 方法定義）
      三輪測試年各抽 30 家（種子 42，只看代號）× 2 組 × 3 次；同一個 Gemini 模型、temperature 0、同一種 JSON 格式
      Agent：agent.run_analysis（工具＋核對迴圈）；直接 LLM：取得原始欄位與計算口徑，自行計算，無工具與核對
      指標：key_figures 11 項的數字正確率（四捨五入到答案表顯示位數後相等，且公司與期間相符）、
            必要欄位覆蓋率、證據引用率、規則判定正確率、模板退回率；以公司為單位 bootstrap 95% 信賴區間
      執行：python -c "import evaluate; evaluate.run_e4()"（需 GEMINI_API_KEY；額度用完自動停止，重跑從快取接續）
輸出：results/e1_ranking.csv（各輪各方法的指標；隨機為 1,000 次平均）
      results/e2_portfolio.csv（各輪組合的 5 年累積含息報酬、最大回撤、下市與異常筆數）
      results/e3_quintile.csv、e3_groups.csv、e3_shap.csv（E3 彙總數字）
      results/e1_random.csv（隨機基準 1,000 次的分布：平均、第 5／50／95 百分位）
      results/e4_answer_key.csv（E4 答案表：90 家的 11 項數字（顯示位數）與規則判定）
      results/e4_reports.csv（每家 × 組 × 次的評分）、results/e4_summary.csv（兩組比較與信賴區間）

執行方式：python evaluate.py（需先執行 python train.py）
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import demo_pipeline as dp
import train

BASE = Path(__file__).resolve().parent
E1_CSV = BASE / "results" / "e1_ranking.csv"
E1_RANDOM_CSV = BASE / "results" / "e1_random.csv"
E2_CSV = BASE / "results" / "e2_portfolio.csv"
E3_QUINTILE_CSV = BASE / "results" / "e3_quintile.csv"
E3_GROUPS_CSV = BASE / "results" / "e3_groups.csv"
E3_SHAP_CSV = BASE / "results" / "e3_shap.csv"
E5_CSV = BASE / "results" / "e5_sensitivity.csv"
E5_THRESHOLDS = (0.05, 0.10, 0.15)   # 營收 5 年 CAGR 門檻（DESIGN 11.5 E5；10% 即 y_main）

TARGET = "y_main"
K_LIST = (20, 50)
N_RANDOM = 1000
SEED = 42

# Yartseva 啟發規則（DESIGN 11.5；Yartseva [1] p.15、25–27、29）
YARTSEVA_BM_MIN = 0.40
YARTSEVA_SCORE = {            # 特徵 → True 表示「值愈高愈好」
    "市值（億元）": False,
    "FCF 殖利率": True,
    "帳面/市值": True,
    "過去 12 個月含息報酬": False,
}


def precision_at_k(y_ordered, k):
    """前 k 名中真正暴漲股的比例（y_ordered：依排名排好的 0/1 標記）。"""
    return float(np.mean(y_ordered[:k]))


def recall_at_k(y_ordered, k):
    """全部暴漲股中，被排進前 k 名的比例。"""
    total = y_ordered.sum()
    return float(y_ordered[:k].sum() / total) if total else float("nan")


def pr_auc(y_ordered):
    """以完整排名計算 average precision（排名第 1 的分數最高）。"""
    n = len(y_ordered)
    return float(average_precision_score(y_ordered, np.arange(n, 0, -1)))


def metrics(y_ordered):
    out = {f"Precision@{k}": precision_at_k(y_ordered, k) for k in K_LIST}
    out["Recall@50"] = recall_at_k(y_ordered, 50)
    out["PR-AUC"] = pr_auc(y_ordered)
    return out


def order_by(df, keys, ascending):
    """依 keys 排序（缺值排最後），最後以代號排序打破同分；回傳排好的 y 陣列。"""
    ordered = df.sort_values(keys + ["代號"], ascending=ascending + [True],
                             na_position="last", kind="mergesort")
    return ordered[TARGET].astype(int).to_numpy()


def yartseva_frame(test_all):
    """Yartseva 啟發規則的篩選旗標與分數（排名用測試年全部樣本，只用特徵）。"""
    pct = test_all[list(YARTSEVA_SCORE)].rank(pct=True, method="average")
    for col, higher_better in YARTSEVA_SCORE.items():
        if not higher_better:
            pct[col] = 1 - pct[col] + 1 / len(test_all)   # 反向後仍在 (0, 1]
    score = pct.mean(axis=1, skipna=True)                   # 四項全缺 → NaN（排最後）
    passed = (test_all["帳面/市值"] > YARTSEVA_BM_MIN) & (test_all["ROE"] > 0)
    return pd.DataFrame({"yartseva_pass": passed.astype(int), "yartseva_score": score},
                        index=test_all.index)


def run_e1():
    """E1：模型與 Baseline 的排序指標比較。"""
    df = train.load_dataset()
    rows, random_rows = [], []
    for round_no, fold in enumerate(train.FOLDS, start=1):
        path = train.MODEL_DIR / f"scores_round{round_no}.parquet"
        if not path.exists():
            sys.exit(f"找不到 {path}，請先執行 python train.py")
        scores = pd.read_parquet(path)

        test_all = df[df["特徵年"] == fold["test"]].copy()
        test_all = test_all.join(yartseva_frame(test_all))
        test_all = test_all.merge(scores[["代號", "score_lgbm", "score_logit"]], on="代號",
                                  how="left", validate="one_to_one")
        assert test_all[["score_lgbm", "score_logit"]].notna().all().all(), "有測試列沒有模型分數"
        ev = test_all[test_all[TARGET].notna()].copy()   # 可評估母體（UNKNOWN 不當成負例）
        n, pos = len(ev), int(ev[TARGET].sum())
        base = {"輪次": round_no, "測試特徵年": fold["test"], "可評估筆數": n,
                "正例": pos, "基準正例率": pos / n}

        methods = {
            "LightGBM": (["score_lgbm"], [False], 0),
            "邏輯迴歸": (["score_logit"], [False], 0),
            "低 P/B": (["帳面/市值"], [False], int(ev["帳面/市值"].isna().sum())),
            "高 ROE": (["ROE"], [False], int(ev["ROE"].isna().sum())),
            "Yartseva 啟發規則": (["yartseva_pass", "yartseva_score"], [False, False],
                                int(ev["yartseva_score"].isna().sum())),
        }
        for name, (keys, asc, unscored) in methods.items():
            m = metrics(order_by(ev, keys, asc))
            extra = {"Yartseva 通過篩選": int(ev["yartseva_pass"].sum())} if name.startswith("Yartseva") else {}
            rows.append({**base, "方法": name, **m, "無法評分筆數": unscored, **extra})

        # 隨機：1,000 次隨機排列（固定種子）
        rng = np.random.default_rng(SEED)
        y = ev[TARGET].astype(int).to_numpy()
        sims = pd.DataFrame([metrics(y[rng.permutation(n)]) for _ in range(N_RANDOM)])
        rows.append({**base, "方法": "隨機（1,000 次平均）", **sims.mean().to_dict(), "無法評分筆數": 0})
        for col in sims.columns:
            q = sims[col].quantile([0.05, 0.5, 0.95])
            random_rows.append({"輪次": round_no, "測試特徵年": fold["test"], "指標": col,
                                "平均": sims[col].mean(), "第 5 百分位": q[0.05],
                                "中位數": q[0.5], "第 95 百分位": q[0.95]})

    out = pd.DataFrame(rows)
    cols = ["輪次", "測試特徵年", "方法", "可評估筆數", "正例", "基準正例率",
            "Precision@20", "Precision@50", "Recall@50", "PR-AUC", "無法評分筆數", "Yartseva 通過篩選"]
    out = out[cols]
    E1_CSV.parent.mkdir(exist_ok=True)
    out.round(4).to_csv(E1_CSV, index=False, encoding="utf-8-sig")
    rand = pd.DataFrame(random_rows).round(4)
    rand.to_csv(E1_RANDOM_CSV, index=False, encoding="utf-8-sig")
    return out, rand


def stock_value_path(stock_id, entry, exit_, calendar):
    """單一股票在 entry → exit_ 每個交易日的價值（進場時 = 1，含息、股利再投入同一檔股票）。

    進場日沒有成交：以之後第一個有成交日的收盤價進場，之前以現金持有（= 1）。
    期間下市或停止交易：最後一筆價值之後維持不變（以最後交易價變現、持有現金，報酬 0）。
    回傳 (價值序列, 資訊)；整段期間都沒有成交時價值序列為 None。
    """
    ws = dp.wealth_series(stock_id, entry, exit_)
    info = {"延後進場": False, "期間下市": False, "價格異常": False}
    if ws.empty:
        return None, info
    info["延後進場"] = ws.index[0] > entry
    info["期間下市"] = (exit_ - ws.index[-1]).days > dp.MAX_STALE_DAYS
    info["價格異常"] = bool(dp.price_jumps(ws, stock_id))
    v = ws["wealth"] / ws["wealth"].iloc[0]
    v = v.reindex(calendar).ffill().fillna(1.0)
    return v, info


def portfolio_stats(values):
    """等權買進持有：組合價值 ＝ 各檔價值平均（進場時等權、之後不再平衡）。"""
    port = values.mean(axis=1)
    drawdown = 1 - port / port.cummax()
    return float(port.iloc[-1] - 1), float(drawdown.max())


def run_e2():
    """E2：每輪前 50 檔等權組合 vs 0050 的 5 年累積含息報酬與最大回撤（毛報酬，未計交易成本）。"""
    df = train.load_dataset()
    cal_all = dp.trading_calendar()
    rows, check = [], []
    for round_no, fold in enumerate(train.FOLDS, start=1):
        year = fold["test"]
        scores = pd.read_parquet(train.MODEL_DIR / f"scores_round{round_no}.parquet")
        t = pd.Timestamp(scores["特徵日"].iloc[0])
        entry = cal_all[cal_all > t][0]            # t 收盤後評分，下一交易日收盤進場（DESIGN 5.5 第 9 條）
        exit_ = dp.horizon_end(year)               # 5 年後對應交易日（第 y + 5 年 4 月最後交易日）收盤
        assert entry > t, "進場日必須在特徵日之後"
        calendar = cal_all[(cal_all >= entry) & (cal_all <= exit_)]

        # 測試年全部評分樣本（不排除 UNKNOWN：選股時不可能知道標記）
        paths, infos = {}, {}
        for sid in scores["代號"]:
            v, info = stock_value_path(sid, entry, exit_, calendar)
            if v is not None:
                paths[sid], infos[sid] = v, info
        values = pd.DataFrame(paths)
        no_trade = sorted(set(scores["代號"]) - set(paths))   # 整段期間都沒有成交

        def summarize(name, ids):
            held = [s for s in ids if s in paths]
            ret, mdd = portfolio_stats(values[held].reindex(columns=held).join(
                pd.DataFrame(1.0, index=calendar, columns=[s for s in ids if s not in paths])))
            return {"輪次": round_no, "測試特徵年": year, "進場日": entry.date(), "出場日": exit_.date(),
                    "方法": name, "檔數": len(ids), "5 年累積含息報酬": ret, "最大回撤": mdd,
                    "延後進場": sum(infos[s]["延後進場"] for s in held),
                    "期間下市": sum(infos[s]["期間下市"] for s in held),
                    "價格異常": sum(infos[s]["價格異常"] for s in held),
                    "全期無成交": len(ids) - len(held)}

        for name, col in (("LightGBM 前 50", "score_lgbm"), ("邏輯迴歸 前 50", "score_logit")):
            top = scores.sort_values([col, "代號"], ascending=[False, True], kind="mergesort")
            rows.append(summarize(name, top["代號"].head(50).tolist()))

        # 隨機：固定種子，1,000 次從測試年全部樣本抽 50 檔
        rng = np.random.default_rng(SEED)
        ids_all = scores["代號"].to_numpy()
        sims = []
        for _ in range(N_RANDOM):
            pick = rng.choice(ids_all, size=50, replace=False)
            held = [s for s in pick if s in paths]
            cash = len(pick) - len(held)
            port = (values[held].sum(axis=1) + cash) / len(pick)
            sims.append((port.iloc[-1] - 1, (1 - port / port.cummax()).max()))
        sims = pd.DataFrame(sims, columns=["ret", "mdd"])
        rows.append({"輪次": round_no, "測試特徵年": year, "進場日": entry.date(), "出場日": exit_.date(),
                     "方法": "隨機 50 檔（1,000 次平均）", "檔數": 50,
                     "5 年累積含息報酬": sims["ret"].mean(), "最大回撤": sims["mdd"].mean(),
                     "報酬第 5 百分位": sims["ret"].quantile(0.05), "報酬第 95 百分位": sims["ret"].quantile(0.95),
                     "回撤第 5 百分位": sims["mdd"].quantile(0.05), "回撤第 95 百分位": sims["mdd"].quantile(0.95)})

        # 0050：同一套進出場規則
        v, info = stock_value_path(dp.BENCHMARK_ID, entry, exit_, calendar)
        ret, mdd = portfolio_stats(v.to_frame())
        rows.append({"輪次": round_no, "測試特徵年": year, "進場日": entry.date(), "出場日": exit_.date(),
                     "方法": "0050 買進持有", "檔數": 1, "5 年累積含息報酬": ret, "最大回撤": mdd,
                     "延後進場": int(info["延後進場"]), "期間下市": 0, "價格異常": int(info["價格異常"]),
                     "全期無成交": 0})
        # 核對：與資料集的 0050 同期含息報酬（特徵日進場、20 日平均終點）比較
        ds_0050 = df.loc[df["特徵年"] == year, "0050 同期含息報酬"].dropna().unique()
        check.append({"輪次": round_no, "E2 0050": ret, "資料集 0050": ds_0050[0] if len(ds_0050) else np.nan,
                      "全期無成交（測試樣本）": len(no_trade)})

    out = pd.DataFrame(rows)
    E2_CSV.parent.mkdir(exist_ok=True)
    out.round(4).to_csv(E2_CSV, index=False, encoding="utf-8-sig")
    return out, pd.DataFrame(check)


GROUPS = [("主要正例", lambda d: d["y_main"] == 1),
          ("高報酬但營收未過門檻", lambda d: (d["y_raw"] == 1) & (d["y_main"] == 0)),
          ("其餘可判定", lambda d: d["y_raw"] == 0)]


def quintile_table(ev):
    """10 項數值特徵：每個特徵年內分五等分（缺值另成一組）的 y_main 正例率與穩定性年數。"""
    base = ev[TARGET].mean()
    rows = []
    for feat in train.NUMERIC_FEATURES:
        pct = ev.groupby("特徵年")[feat].rank(pct=True, method="average")
        grp = np.ceil(pct * 5).clip(1, 5)
        label = grp.map(lambda q: f"Q{int(q)}" if pd.notna(q) else "缺值")
        # 穩定性：8 年中 Q5 正例率高於 Q1 的年數
        by_year = ev.assign(組=label).groupby(["特徵年", "組"])[TARGET].mean().unstack()
        both = by_year[["Q1", "Q5"]].dropna()
        higher, ties = int((both["Q5"] > both["Q1"]).sum()), int((both["Q5"] == both["Q1"]).sum())
        for g in ["Q1", "Q2", "Q3", "Q4", "Q5", "缺值"]:
            sub = ev[label == g]
            n, pos = len(sub), int(sub[TARGET].sum())
            rows.append({"特徵": feat, "組": g, "筆數": n, "正例": pos,
                         "正例率": pos / n if n else np.nan, "基準正例率": base,
                         "正例率 ÷ 基準": (pos / n) / base if n else np.nan,
                         "Q5 高於 Q1 年數": higher, "Q5 等於 Q1 年數": ties, "可比較年數": len(both)})
    return pd.DataFrame(rows)


def group_table(df):
    """三組（再依期間下市分層）的特徵中位數；附 UNKNOWN 筆數與各年 0050 同期含息報酬。"""
    ev = df[df[TARGET].notna()]
    rows = []
    for name, cond in GROUPS:
        g = ev[cond(ev)]
        delisted = g["報酬狀態"] == "下市暫估"
        for layer, sub in (("全部", g), ("期間下市", g[delisted]), ("未下市", g[~delisted])):
            for feat in train.NUMERIC_FEATURES:
                rows.append({"分析": "三組中位數", "組別": name, "分層": layer, "筆數": len(sub),
                             "特徵": feat, "數值": sub[feat].median(), "有值筆數": int(sub[feat].notna().sum())})
    unknown = df[df[TARGET].isna()]
    rows.append({"分析": "UNKNOWN", "組別": "y_main UNKNOWN（不納入三組）", "分層": "全部",
                 "筆數": len(unknown), "特徵": "", "數值": np.nan, "有值筆數": np.nan})
    for year, v in df.groupby("特徵年")["0050 同期含息報酬"].first().items():
        rows.append({"分析": "0050 背景", "組別": str(year), "分層": "", "筆數": np.nan,
                     "特徵": "0050 同期含息報酬", "數值": v, "有值筆數": np.nan})
    return pd.DataFrame(rows)


def shap_table(df):
    """LightGBM 內建 pred_contrib（TreeSHAP）：三輪測試年全部評分樣本的平均 |SHAP| 排名與方向。"""
    import joblib
    names = {v: k for k, v in {**train.NUMERIC_FEATURES, **train.CATEGORY_FEATURE}.items()}
    rows = []
    for round_no, fold in enumerate(train.FOLDS, start=1):
        tr, test, _ = train.split(df, fold)
        X = train.to_model_frame(test, sorted(tr["產業"].dropna().unique()))
        model = joblib.load(train.MODEL_DIR / f"lgbm_round{round_no}.joblib")
        contrib = model.predict(X, pred_contrib=True)[:, :-1]      # 最後一欄是基準值
        assert contrib.shape[1] == len(train.FEATURE_COLS)
        sh = pd.DataFrame(contrib, columns=train.FEATURE_COLS, index=X.index)
        imp = sh.abs().mean().sort_values(ascending=False)
        for rank, (col, val) in enumerate(imp.items(), start=1):
            rho = (X[col].corr(sh[col], method="spearman") if col != "industry" else np.nan)
            rows.append({"輪次": round_no, "測試特徵年": fold["test"], "特徵": names[col],
                         "平均 |SHAP|": val, "排名": rank, "Spearman（特徵值, SHAP）": rho})
    return pd.DataFrame(rows)


def run_e3():
    """E3：五等分正例率與穩定性、三組特徵中位數（依下市分層）、跨輪 SHAP。"""
    df = train.load_dataset()
    ev = df[df[TARGET].notna()].copy()
    ev[TARGET] = ev[TARGET].astype(int)
    q, g, sh = quintile_table(ev), group_table(df), shap_table(df)
    for table, path in ((q, E3_QUINTILE_CSV), (g, E3_GROUPS_CSV), (sh, E3_SHAP_CSV)):
        table.round(4).to_csv(path, index=False, encoding="utf-8-sig")
    return q, g, sh


# ───────────── E4：報告品質（Agent vs 直接 LLM，DESIGN 11.5 E4 方法定義） ─────────────
E4_PER_ROUND = 30
E4_REPS = 3
E4_N_BOOT = 1000
E4_KEY_CSV = BASE / "results" / "e4_answer_key.csv"
E4_REPORTS_CSV = BASE / "results" / "e4_reports.csv"
E4_SUMMARY_CSV = BASE / "results" / "e4_summary.csv"
E4_GROUPS = ("Agent", "直接 LLM")

# 答案表的顯示位數（建立答案表時固定，執行 E4 後不再更動）：欄位 → (倍率, 小數位數)
#   比率以百分比 1 位小數（帳面／市值例外，3 位小數）；市值以億元 1 位小數；倍數 1 位小數
def e4_display(field):
    import agent
    unit = agent.unit_of(field)
    if unit == "比率":
        return (1, 3) if "帳面/市值" in field else (100, 1)
    return (1, 1)


def e4_sample():
    """三輪測試年全部評分樣本中，種子 42 各抽 30 家；只用代號，不看標記與報酬。"""
    import agent
    df = agent.labeled()
    rows = []
    for n, f in enumerate(train.FOLDS, start=1):
        ids = df.loc[df["特徵年"] == f["test"], "代號"].sort_values().reset_index(drop=True)
        for sid in sorted(ids.sample(E4_PER_ROUND, random_state=SEED)):
            rows.append({"輪次": n, "特徵年": f["test"], "代號": sid})
    return pd.DataFrame(rows)


def e4_answer_key(sample):
    """程式建立的答案表：11 項數字（10 項特徵＋應計／平均總資產）與 2 條規則判定。"""
    import agent
    rows = []
    for _, s in sample.iterrows():
        log = agent.EvidenceLog()
        feats = agent.get_features(s["代號"], s["特徵年"], log)
        rl = agent.check_rules(s["代號"], s["特徵年"], log)
        row = {"輪次": s["輪次"], "代號": s["代號"], "名稱": feats["name"], "分析日": str(feats["as_of"])}
        for f in agent.KEY_FIELDS:
            row[f] = feats["features"][f]["value"]
        for r in rl["rules"]:
            row[f"規則：{r['rule']}"] = agent.STATE[r["triggered"]]
        rows.append(row)
    return pd.DataFrame(rows)


def e4_raw_inputs(stock_id, year):
    """直接 LLM 組的輸入：計算 11 項數字與 2 條規則所需的原始欄位（只用特徵日 t 以前已公開的資料）。"""
    import agent
    import rules as rl
    t = dp.feature_date(year)
    fy = year - 1
    meta = dp.stock_meta(stock_id)
    row = agent.find_row(stock_id, year)

    def bs(y, types):
        return dp.statement_value("TaiwanStockBalanceSheet", stock_id, pd.Timestamp(f"{y}-12-31"), types)[0]

    def inc(types):
        return dp.annual_income(stock_id, fy, types)[0]

    d1, p1 = dp.start_price(stock_id, t)
    d0, p0 = dp.start_price(stock_id, t - pd.DateOffset(years=1))
    p = dp.price_series(stock_id)
    p = p[(p.index > t - pd.DateOffset(years=1)) & (p.index <= t)]
    month_end = p.groupby(p.index.to_period("M")).last()
    events = []
    if d0 is not None:
        dv = dp.load("TaiwanStockDividendResult", stock_id)
        if not dv.empty:
            dv = dv[(dv["date"] > d0) & (dv["date"] <= t)]
            for _, r in dv.iterrows():
                after = r["after_price"] if r["after_price"] > 0 else r["reference_price"]
                events.append({"日期": f"{r['date']:%Y-%m-%d}", "事件": "除權息", "事件前價格": float(r["before_price"]),
                               "事件後參考價": float(after)})
        cr = dp.capital_reductions(stock_id)
        for _, r in cr[(cr["date"] > d0) & (cr["date"] <= t)].iterrows() if len(cr) else []:
            events.append({"日期": f"{r['date']:%Y-%m-%d}", "事件": "減資", "事件前價格": float(r["p_old"]),
                           "事件後參考價": float(r["p_ref"])})
        sp = dp.load("TaiwanStockSplitPrice", stock_id)
        if not sp.empty:
            for _, r in sp[(sp["date"] > d0) & (sp["date"] <= t)].iterrows():
                events.append({"日期": f"{r['date']:%Y-%m-%d}", "事件": "分割／面額變更",
                               "事件前價格": float(r["before_price"]), "事件後參考價": float(r["after_price"])})
    d_sh, issued = dp.issued_shares(stock_id, t)
    per = dp.load("TaiwanStockPER", stock_id)
    per = per[(per["date"] <= t) & (per["date"] > t - pd.DateOffset(days=dp.MAX_STALE_DAYS))] if len(per) else per
    rev = dp.load("TaiwanStockMonthRevenue", stock_id)
    rev = rev.drop_duplicates(["revenue_year", "revenue_month"], keep="last") if len(rev) else rev
    rev = rev[rev["revenue_year"].isin([fy - 1, fy])].sort_values(["revenue_year", "revenue_month"]) if len(rev) else rev
    years = train.train_years_for(year)
    lab = agent.labeled()
    th_rows = lab[lab["特徵年"].isin(years)] if years else agent.all_features()[agent.all_features()["特徵年"] == year]
    th = rl.compute_thresholds(th_rows)
    yuan = lambda v: None if v is None else float(v)
    return {
        "公司": {"代號": stock_id, "名稱": row["名稱"], "產業": row["產業"], "分析日（特徵日）": f"{t:%Y-%m-%d}",
               "前一年度": fy},
        "股價（元）": {"特徵日收盤": {"日期": f"{d1:%Y-%m-%d}" if d1 is not None else None, "收盤": p1},
                   "一年前收盤": {"日期": f"{d0:%Y-%m-%d}" if d0 is not None else None, "收盤": p0},
                   "近 12 個月月底收盤": {str(k): float(v) for k, v in month_end.items()},
                   "期間公司行動（除權息、減資、分割）": events},
        "股數": ({"特徵日發行股數（股）": issued, "日期": f"{d_sh:%Y-%m-%d}"} if issued is not None else
               {f"{fy} 年底普通股股本（元）": yuan(bs(fy, ["OrdinaryShare", "CapitalStock"])), "面額（元）": dp.FACE_VALUE}),
        "本益比（交易所公布，特徵日以前最近一筆）": (float(per.sort_values("date")["PER"].iloc[-1]) if len(per) else None),
        f"{fy} 年損益表（四季合計，元）": {"營收": yuan(inc(["Revenue"])), "毛利": yuan(inc(["GrossProfit"])),
                                  "歸屬母公司淨利": yuan(inc(dp.NI_PARENT_TYPES)), "合併淨利": yuan(inc(dp.NI_TOTAL_TYPES))},
        f"{fy} 年底資產負債表（元）": {"歸屬母公司權益": yuan(bs(fy, dp.EQUITY_TYPES)), "總負債": yuan(bs(fy, ["Liabilities"])),
                                 "總資產": yuan(bs(fy, ["TotalAssets"]))},
        f"{fy - 1} 年底資產負債表（元）": {"歸屬母公司權益": yuan(bs(fy - 1, dp.EQUITY_TYPES)),
                                     "總資產": yuan(bs(fy - 1, ["TotalAssets"]))},
        f"{fy} 年現金流量表（全年，元）": {
            "營業現金流": yuan(dp.statement_value("TaiwanStockCashFlowsStatement", stock_id, pd.Timestamp(f"{fy}-12-31"),
                                             dp.CFO_TYPES)[0]),
            "取得不動產、廠房及設備": yuan(dp.statement_value("TaiwanStockCashFlowsStatement", stock_id,
                                                  pd.Timestamp(f"{fy}-12-31"), ["PropertyAndPlantAndEquipment"])[0])},
        "月營收（元）": {f"{int(r['revenue_year'])}-{int(r['revenue_month']):02d}": float(r["revenue"])
                     for _, r in rev.iterrows()} if len(rev) else {},
        "領域規則門檻（訓練期特徵分布）": {k: v for k, v in th.items() if k not in ("樣本數",)},
    }


E4_DEFINITIONS = """計算口徑（與系統相同）：
- 市值（億元）＝ 特徵日收盤 × 特徵日發行股數 ÷ 1 億（沒有發行股數時用 股本 ÷ 面額 當股數）
- FCF 殖利率 ＝（營業現金流 − |取得不動產、廠房及設備|）÷ 市值
- 帳面/市值 ＝ 前一年底歸屬母公司權益 ÷ 市值（權益 ≤ 0 為缺值）
- 本益比 ＝ 提供的本益比（≤ 0 或沒有資料為缺值）
- ROE ＝ 歸屬母公司淨利 ÷ 平均權益（前一年底與前兩年底平均；缺前期時用期末）
- 毛利率 ＝ 毛利 ÷ 營收（損益表）
- 年營收成長率 ＝ 前一年 12 個月營收合計 ÷ 前兩年 12 個月營收合計 − 1（月份不齊為缺值）
- 營業現金流/淨利 ＝ 營業現金流 ÷ 合併淨利（淨利 ≤ 0 為缺值）
- 負債比 ＝ 總負債 ÷ 總資產
- 過去 12 個月含息報酬 ＝ 特徵日收盤 × Π（事件前價格 ÷ 事件後參考價）÷ 一年前收盤 − 1
- 應計/平均總資產 ＝（合併淨利 − 營業現金流）÷ 平均總資產（前一年底與前兩年底平均；金融保險業不適用）
- 盈餘品質紅旗：應計/平均總資產 ≥ 門檻（第 90 百分位）時觸發；缺值為不適用
- 估值過高：帳面/市值 ≤ 門檻（第 10 百分位），或本益比（> 0）≥ 門檻（第 90 百分位）時觸發；兩者皆缺為不適用"""

E4_SYSTEM_DIRECT = """你是台股研究助理。根據使用者提供的原始資料，自行計算需要的數字並撰寫結構化分析報告。規則：
1. 只能使用提供的資料，不可提到資料以外的公司資訊、新聞或分析日之後的事。
2. 照提供的計算口徑計算；key_figures 的 value 依下列格式填寫（與系統答案表相同）：
   市值（億元）：「123.4 億元」；本益比、營業現金流/淨利：「1.2 倍」；帳面/市值：小數 3 位，例如「0.642」；
   FCF 殖利率、ROE、毛利率、年營收成長率、負債比、過去 12 個月含息報酬、應計/平均總資產：百分比 1 位小數，例如「12.3%」。
3. 只輸出 JSON，格式：
{"stock_id": "代號", "as_of": "YYYY-MM-DD", "level": null,
 "summary": "2～3 句總結",
 "supporting": [{"text": "支持證據", "source": "用到的原始資料名稱"}],
 "opposing": [{"text": "反對證據或注意事項", "source": "用到的原始資料名稱"}],
 "rules": [{"rule": "盈餘品質紅旗 或 估值過高", "triggered": "觸發/未觸發/不適用", "text": "說明", "source": "..."}],
 "key_figures": [{"field": "特徵名稱", "value": "數值與單位（缺值填 null）", "source": "用到的原始資料名稱"}],
 "uncertainty": ["不確定性說明"]}
4. key_figures 必須列出以下 11 項：市值（億元）、FCF 殖利率、帳面/市值、本益比、ROE、毛利率、年營收成長率、營業現金流/淨利、負債比、過去 12 個月含息報酬、應計/平均總資產。"""


def _names(obj, out=None):
    """原始資料中所有欄位名稱（直接 LLM 組的引用來源比對用）。"""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out.add(re.sub(r"（.*?）", "", str(k)).strip())   # 「2017 年損益表（四季合計，元）」也接受「2017 年損益表」
            _names(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _names(v, out)
    return out


def e4_parse_value(field, val):
    """把報告中的數值字串換成與答案表相同的單位；缺值回傳 None，無法解析回傳 nan。"""
    import agent
    if val is None or str(val).strip() in [w for w in agent.MISSING_WORDS if w]:
        return None
    s = str(val).replace(",", "").replace("−", "-")
    m = re.search(r"([+\-]?\d+(?:\.\d+)?)\s*(%|％|兆元|億元|萬元|元|倍)?", s)
    if not m:
        return np.nan
    x, u = float(m.group(1)), m.group(2) or ""
    unit = agent.unit_of(field)
    if unit == "比率":
        return x / 100 if u in ("%", "％") else x
    if unit == "億元":
        return {"兆元": x * 1e4, "萬元": x / 1e4, "元": x / 1e8}.get(u, x)
    return x / 100 if u in ("%", "％") else x     # 倍數寫成百分比時換算（數值相同，只是寫法不同）


def _round_half_up(x, d, scale=1):
    """一般的四捨五入（先轉十進位再乘倍率）；與 agent 的顯示、核對共用同一個函式。"""
    import agent
    return agent.round_half_up(x, d, scale)


def e4_grade(report, key_row, group, evidence=None, raw_names=None):
    """數字正確率、必要欄位覆蓋率、證據引用率、規則判定正確率（DESIGN 11.5 E4 方法定義）。"""
    import agent
    out = {"可解析": isinstance(report, dict)}
    report = report if isinstance(report, dict) else {}
    entity_ok = str(report.get("stock_id")) == key_row["代號"] and str(report.get("as_of")) == key_row["分析日"]
    kf = {k.get("field"): k for k in report.get("key_figures") or [] if isinstance(k, dict)}
    given = correct = cited = covered = 0
    for f in agent.KEY_FIELDS:
        truth = key_row[f]
        truth = None if pd.isna(truth) else float(truth)
        if f not in kf:
            continue
        covered += 1
        x = e4_parse_value(f, kf[f].get("value"))
        if x is None:
            continue                                  # 回報缺值：不算數字；答案也是缺值時視為正確覆蓋
        given += 1
        scale, d = e4_display(f)
        ok = (entity_ok and truth is not None and not pd.isna(x)
              and _round_half_up(x, d, scale) == _round_half_up(truth, d, scale))
        correct += ok
        if group == "Agent":
            eid = kf[f].get("evidence_id")
            cited += bool(evidence and eid in evidence and evidence[eid]["field"] == f)
        else:
            src = str(kf[f].get("source") or "")
            cited += bool(src and raw_names and any(n in src for n in raw_names if len(n) >= 2))
    rules_ok = 0
    rmap = {r.get("rule"): r for r in report.get("rules") or [] if isinstance(r, dict)}
    for rname in ("盈餘品質紅旗", "估值過高"):
        rules_ok += int(entity_ok and rname in rmap and rmap[rname].get("triggered") == key_row[f"規則：{rname}"])
    n = len(agent.KEY_FIELDS)
    out.update({"公司與期間相符": entity_ok, "回報數字": given, "正確數字": correct,
                "數字正確率": correct / given if given else np.nan, "必要欄位覆蓋率": covered / n,
                "證據引用率": cited / given if given else np.nan, "規則判定正確率": rules_ok / 2})
    return out


def _bootstrap_ci(diff, seed=SEED, n_boot=E4_N_BOOT):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _n_samples(df):
    """樣本數：以（輪次, 代號）計，同一家在不同輪被抽到算不同樣本。"""
    return len(df[["輪次", "代號"]].drop_duplicates())


def e4_summary(rep_df):
    """以公司為單位：先平均 3 次重複，再算兩組差與 bootstrap 95% 信賴區間；逐輪與合計。"""
    metrics_ = ["數字正確率", "必要欄位覆蓋率", "證據引用率", "規則判定正確率"]
    done = rep_df.groupby(["代號", "輪次"])["組別"].count()
    full = done[done == 2 * E4_REPS].index
    df = rep_df.set_index(["代號", "輪次"]).loc[full].reset_index()
    comp = df.groupby(["輪次", "代號", "組別"])[metrics_].mean().reset_index()
    rows = []
    for scope, part in [(f"第 {r} 輪", comp[comp["輪次"] == r]) for r in sorted(comp["輪次"].unique())] + [("合計", comp)]:
        for m in metrics_:
            # 同一家可能在不同輪被抽到，樣本以（輪次, 代號）識別
            wide = part.pivot(index=["輪次", "代號"], columns="組別", values=m)
            if not set(E4_GROUPS) <= set(wide.columns):
                continue
            mean, lo, hi = _bootstrap_ci(wide["Agent"] - wide["直接 LLM"])
            rows.append({"範圍": scope, "指標": m, "公司數": len(wide), "Agent": wide["Agent"].mean(),
                         "直接 LLM": wide["直接 LLM"].mean(), "差（Agent − 直接）": mean,
                         "95% 信賴區間下限": lo, "95% 信賴區間上限": hi})
        ag = df[(df["組別"] == "Agent") & df["輪次"].isin(part["輪次"].unique())]
        if len(ag):
            nt = ag[ag["報告狀態"] == "ok"]
            rows.append({"範圍": scope, "指標": "模板退回率（Agent）", "公司數": _n_samples(ag),
                         "Agent": float((ag["報告狀態"] == "template").mean())})
            rows.append({"範圍": scope, "指標": "數字正確率（Agent，不含模板報告）", "公司數": _n_samples(nt),
                         "Agent": float(nt["數字正確率"].mean()) if len(nt) else np.nan})
            rows.append({"範圍": scope, "指標": "平均核對輪數（Agent）", "公司數": _n_samples(ag),
                         "Agent": float(ag["核對輪數"].mean())})
    return pd.DataFrame(rows)


def run_e4(limit=None, reps=E4_REPS):
    """E4：90 家 × 2 組 × 3 次。LLM 回應全部快取，額度用完時停止，重跑即從中斷處接續。

    limit：只跑前 limit 家（試跑用）；試跑不寫 e4_summary.csv。
    執行方式：python -c "import evaluate; evaluate.run_e4()"
    """
    import agent
    sample = e4_sample()
    key = e4_answer_key(sample)
    if limit is None:
        out = key.copy()
        for f in agent.KEY_FIELDS:
            scale, d = e4_display(f)
            out[f] = [_round_half_up(v, d, scale) if pd.notna(v) else np.nan for v in out[f]]
        out.rename(columns={f: (f"{f}（%）" if e4_display(f)[0] == 100 else f) for f in agent.KEY_FIELDS}) \
            .to_csv(E4_KEY_CSV, index=False, encoding="utf-8-sig")
    keymap = key.set_index(["輪次", "代號"], drop=False)   # 同一家可能在不同輪被抽到
    todo = sample if limit is None else sample.groupby("輪次").head(math.ceil(limit / 3)).head(limit)
    records, stopped = [], None
    try:
        for rep in range(reps):
            for _, s in todo.iterrows():
                sid, year, k = s["代號"], s["特徵年"], keymap.loc[(s["輪次"], s["代號"])]
                res = agent.run_analysis(sid, year, rep=rep)
                if res.get("llm_error"):
                    raise agent.LLMError(res["llm_error"])
                g = e4_grade(res["report"], k, "Agent", evidence=res["ctx"]["evidence"])
                records.append({"輪次": s["輪次"], "代號": sid, "重複": rep, "組別": "Agent",
                                "報告狀態": res["status"], "核對輪數": len(res["attempts"]), **g})
                raw = e4_raw_inputs(sid, year)
                prompt = json.dumps({"原始資料": raw, "計算口徑": E4_DEFINITIONS}, ensure_ascii=False, default=str)
                try:
                    rep_json = agent.llm_call(E4_SYSTEM_DIRECT, prompt, rep=rep)
                except ValueError:
                    rep_json = None
                g = e4_grade(rep_json, k, "直接 LLM", raw_names=_names(raw))
                records.append({"輪次": s["輪次"], "代號": sid, "重複": rep, "組別": "直接 LLM",
                                "報告狀態": "ok" if rep_json is not None else "JSON 錯誤", "核對輪數": np.nan, **g})
                print(f"第 {rep + 1} 次 {sid}：Agent {records[-2]['數字正確率']:.2f}（{res['status']}，"
                      f"{len(res['attempts'])} 輪）｜直接 LLM {records[-1]['數字正確率']:.2f}", flush=True)
    except agent.LLMError as e:
        stopped = str(e)
        print(f"停止：{e}（已完成 {len(records) // 2} 筆；重跑會從快取接續）")
    rep_df = pd.DataFrame(records)
    total = len(todo) * reps
    print(f"完成 {len(rep_df) // 2} / {total}")
    if limit is None:
        rep_df.to_csv(E4_REPORTS_CSV, index=False, encoding="utf-8-sig")
        if stopped is None and len(rep_df) == 2 * total:
            summ = e4_summary(rep_df)
            summ.round(4).to_csv(E4_SUMMARY_CSV, index=False, encoding="utf-8-sig")
            return rep_df, summ
    return rep_df, None


def e5_labels(df):
    """E5 的 4 種標記（DESIGN 11.5 E5 方法定義）：y_raw 與營收門檻 5% / 10% / 15%。

    門檻版：y_raw = 1 且 CAGR 可判定 → CAGR ≥ 門檻；y_raw = 1 但 CAGR 無法判定 → UNKNOWN；
    y_raw = 0 → 0；y_raw UNKNOWN → UNKNOWN（同 build_dataset.py 的 y_main 規則）。
    """
    y_raw = df["y_raw"]
    cagr_ok = df["CAGR 狀態"] == "OK"
    labels = {"y_raw（報酬前 5%）": y_raw.astype("Int64")}
    for g in E5_THRESHOLDS:
        lab = pd.Series(pd.NA, index=df.index, dtype="Int64")
        lab[y_raw == 0] = 0
        top = (y_raw == 1) & cagr_ok
        lab[top] = (df.loc[top, "營收 CAGR"] >= g).astype(int)
        labels[f"營收門檻 {round(g * 100)}%"] = lab
    # 自我檢查：10% 版本必須與資料集的 y_main 完全相同
    main = labels["營收門檻 10%"]
    assert main.isna().equals(df["y_main"].isna()), "10% 門檻重算的 UNKNOWN 與 y_main 不同"
    assert (main.dropna() == df["y_main"].dropna().astype(int)).all(), "10% 門檻重算結果與 y_main 不同"
    return labels


def run_e5():
    """（延伸）E5：主要／對照標記與營收門檻的敏感度分析（DESIGN 11.5 E5 方法定義）。

    每種標記各自依三輪切分重訓 LightGBM（參數、特徵、產業類別處理同 train.py），
    在該標記可判定的測試母體上算指標與隨機基準（種子 42，1,000 次）。
    自我檢查：營收門檻 10%（＝ y_main）重訓的分數與 train.py 相同、指標與 results/e1_ranking.csv 相同。
    """
    df = train.load_dataset()
    labels = e5_labels(df)
    e1 = pd.read_csv(E1_CSV, encoding="utf-8-sig")
    rows = []
    for name, lab in labels.items():
        d = df.copy()
        d[train.TARGET] = lab            # train.split 以 y_main 欄判定可訓練列，這裡換成該標記
        for round_no, fold in enumerate(train.FOLDS, start=1):
            tr, test, info = train.split(d, fold)
            cats = train.industry_categories(tr)
            y_tr = tr[train.TARGET].astype(int).to_numpy()
            model = train.train_lightgbm(train.to_model_frame(tr, cats), y_tr)
            test = test.assign(score=model.predict_proba(train.to_model_frame(test, cats))[:, 1])

            ev = test[test[train.TARGET].notna()]
            ordered = ev.sort_values(["score", "代號"], ascending=[False, True], kind="mergesort")
            y_ord = ordered[train.TARGET].astype(int).to_numpy()
            m = metrics(y_ord)

            rng = np.random.default_rng(SEED)
            y = ev[train.TARGET].astype(int).to_numpy()
            sims = pd.DataFrame([metrics(y[rng.permutation(len(y))]) for _ in range(N_RANDOM)])

            if name == "營收門檻 10%":   # 與 train.py、E1 一致性檢查
                saved = pd.read_parquet(train.MODEL_DIR / f"scores_round{round_no}.parquet")
                chk = test[["代號", "score"]].merge(saved[["代號", "score_lgbm"]], on="代號", validate="one_to_one")
                assert len(chk) == len(test) and np.allclose(chk["score"], chk["score_lgbm"], rtol=0, atol=1e-12), \
                    f"第 {round_no} 輪 y_main 重訓分數與 train.py 不同"
                ref = e1[(e1["輪次"] == round_no) & (e1["方法"] == "LightGBM")].iloc[0]
                for k, v in m.items():
                    assert round(v, 4) == round(float(ref[k]), 4), f"第 {round_no} 輪 {k} 與 E1 不同"

            base = {"標記": name, "輪次": round_no, "測試特徵年": fold["test"],
                    "訓練筆數": len(tr), "訓練正例": int(y_tr.sum()),
                    "訓練 UNKNOWN 排除": info["訓練 UNKNOWN 排除"],
                    "可評估筆數": len(ev), "測試正例": int(y.sum()), "基準正例率": y.mean()}
            for k in ["Precision@20", "Precision@50", "Recall@50", "PR-AUC"]:
                q = sims[k].quantile([0.05, 0.95])
                rows.append({**base, "指標": k, "LightGBM": m[k], "隨機平均": sims[k].mean(),
                             "隨機第 5 百分位": q[0.05], "隨機第 95 百分位": q[0.95],
                             "高於隨機第 95 百分位": bool(m[k] > q[0.95])})
        print(f"{name}：三輪完成")
    out = pd.DataFrame(rows)
    out.round(4).to_csv(E5_CSV, index=False, encoding="utf-8-sig")
    print(f"已輸出 {E5_CSV.relative_to(BASE)}；10% 門檻與 y_main、train.py 分數、E1 指標一致性檢查通過")
    return out


def main():
    """依序執行 E1、E2、E3；（延伸）E5 尚未實作。"""
    out, rand = run_e1()
    pd.set_option("display.width", 200)
    print(f"已輸出 {E1_CSV.relative_to(BASE)}、{E1_RANDOM_CSV.relative_to(BASE)}\n")
    print(out.drop(columns=["測試特徵年"]).round(3).to_string(index=False))
    print("\n隨機基準分布（1,000 次）")
    print(rand.drop(columns=["測試特徵年"]).round(3).to_string(index=False))
    e2, check = run_e2()
    print(f"\n已輸出 {E2_CSV.relative_to(BASE)}（毛報酬，未計交易成本）\n")
    print(e2.drop(columns=["測試特徵年"]).round(3).to_string(index=False))
    print("\n0050 核對（E2：下一交易日進場、單日收盤出場；資料集：特徵日進場、20 日平均終點）")
    print(check.round(4).to_string(index=False))
    q, g, sh = run_e3()
    print(f"\n已輸出 {E3_QUINTILE_CSV.relative_to(BASE)}、{E3_GROUPS_CSV.name}、{E3_SHAP_CSV.name}")
    print("\nE3 五等分正例率（8 年合併）")
    wide = q.pivot(index="特徵", columns="組", values="正例率")[["Q1", "Q2", "Q3", "Q4", "Q5", "缺值"]]
    stab = q.groupby("特徵")[["Q5 高於 Q1 年數", "Q5 等於 Q1 年數", "可比較年數"]].first()
    print(wide.join(stab).round(3).to_string())
    print(f"基準正例率 {q['基準正例率'].iloc[0]:.4f}")
    print("\nE3 三組中位數（全部）")
    med = g[(g["分析"] == "三組中位數") & (g["分層"] == "全部")].pivot(index="特徵", columns="組別", values="數值")
    print(med.round(3).to_string())
    print(g[g["分析"] != "三組中位數"].to_string(index=False))
    print("\nE3 SHAP 排名（平均 |SHAP|）")
    print(sh.pivot(index="特徵", columns="輪次", values="排名").join(
        sh.pivot(index="特徵", columns="輪次", values="Spearman（特徵值, SHAP）").add_prefix("ρ 第")).round(2).to_string())


if __name__ == "__main__":
    main()
