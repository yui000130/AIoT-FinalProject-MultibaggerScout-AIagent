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
  E5（延伸）：敏感度分析（待完成）
（E4 為 Agent 報告評估，見 agent.py）

E1 方法定義（DESIGN 11.5，執行前固定）：
  - 可評估母體：測試年 y_main 可判定的列；所有方法對同一母體排序，同分依代號
  - 單一因子缺值者排最後；Yartseva 規則分數只平均有值的項目，篩選條件缺值視為未通過
  - Yartseva 啟發規則：帳面／市值 > 0.40 且 ROE > 0 者在前；分數 ＝ 當年百分位排名等權平均
    （市值小、FCF 殖利率高、帳面／市值高、過去 12 個月含息報酬低），見 knowledge/lit_yartseva.md
  - 百分位排名只用測試年同一特徵日的全部樣本特徵（特徵日已公開），不用標記

輸入：data/dataset.parquet、models/scores_round{n}.parquet（train.py）
輸出：results/e1_ranking.csv（各輪各方法的指標；隨機為 1,000 次平均）
      results/e2_portfolio.csv（各輪組合的 5 年累積含息報酬、最大回撤、下市與異常筆數）
      results/e3_quintile.csv、e3_groups.csv、e3_shap.csv（E3 彙總數字）
      results/e1_random.csv（隨機基準 1,000 次的分布：平均、第 5／50／95 百分位）

執行方式：python evaluate.py（需先執行 python train.py）
"""

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


def run_e5():
    """（延伸）E5：觀察期、標記定義、營收門檻的敏感度分析。"""
    # TODO
    raise NotImplementedError


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
