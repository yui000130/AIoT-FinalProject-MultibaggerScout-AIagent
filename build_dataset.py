"""
build_dataset.py －Point-in-Time「公司 × 年份」資料集建構

對 2013–2020 年每年 4 月底的特徵日 t（DESIGN.md 5.2–5.6、第 8 節 Algorithm 2）：
  1. 母體：證交所上市普通股（含之後下市者），特徵日時已上市滿一年、不在興櫃
       現在上市：證交所上市日期 ≤ t − 1 年
       已終止上市：下市日 > t；沒有官方上市日期，以第一筆股價 ≤ t − 1 年且近一年無興櫃價推估
  2. 流動性過濾：t 之前一年（不含 t 之後）的日均成交值，由低到高排序（同值依代號），
     排除最低 floor(10% × N) 檔；沒有成交的交易日以 0 計入平均
  3. 10 項特徵＋產業類別、t → t + 5 年含息報酬、營收 5 年 CAGR：沿用 demo_pipeline.py 的計算
  4. 標記（每個特徵年分開）：
       可排名 ＝ 報酬狀態為 OK 或「下市暫估」（價格異常不參與，記為 UNKNOWN）
       y_raw ：可排名母體 N 檔中，報酬前 ceil(5% × N) 檔 = 1，其餘 0；同分依代號
       y_main：y_raw = 1 且營收 CAGR ≥ 10% → 1；y_raw = 1 但營收 UNKNOWN → UNKNOWN；y_raw = 0 → 0
       （延伸）y_main_eps：前 5% 且營收或 EPS 任一通過 → 1；兩者都不通過 → 0；其餘 UNKNOWN
         EPS：起點 > 0 時 5 年 CAGR ≥ 10%，起點 ≤ 0 時終點 > 0（由虧轉盈）
     UNKNOWN 以缺值（pd.NA）儲存，原因另存欄位，不當成負例
  5. 標記揭曉日 ＝ 觀察期終點（t + 5 年 4 月最後交易日）；訓練時只能用揭曉日 ≤ 決策日的列

輸入：data/raw/（collect_data.py），經 demo_pipeline.py 讀取
輸出：data/dataset.parquet（每列 = 公司 × 特徵年；原始資料衍生，只在本機、不上傳）
      results/dataset_summary.csv（各特徵年的母體、過濾、標記筆數；只有統計數字）
"""

import math
import sys

import numpy as np
import pandas as pd

import demo_pipeline as dp

FEATURE_YEARS = dp.FEATURE_YEARS
HORIZON_YEARS = dp.HORIZON_YEARS
TOP_PCT = 0.05                 # 報酬排名前 5%
LIQUIDITY_DROP_PCT = 0.10      # 排除日均成交值最低 10%
END_PRICE_DAYS = dp.END_PRICE_DAYS   # 結束價取最後 20 個交易日（約一個月）平均
GROWTH_CAGR = dp.GROWTH_CAGR   # 營收 5 年年複合成長率 ≥ 10%（事先固定，不依測試資料調整）
RANKABLE_STATUS = {"OK", "下市暫估"}

OUT_FILE = dp.BASE / "data" / "dataset.parquet"
SUMMARY_FILE = dp.BASE / "results" / "dataset_summary.csv"
# 0050 2025-06 分割暫停交易 5 天，交易日曆少這幾天；檢查不影響任何特徵日與終點
CALENDAR_GAP = (pd.Timestamp("2025-06-11"), pd.Timestamp("2025-06-17"))


def check_calendar():
    """確認特徵日與觀察期終點不受 0050 日曆缺口影響（終點 20 日平均用的是個股自己的成交日）。"""
    for y in FEATURE_YEARS:
        for d in (dp.feature_date(y), dp.horizon_end(y)):
            assert not (CALENDAR_GAP[0] <= d <= CALENDAR_GAP[1]), f"{y} 年的日期 {d:%Y-%m-%d} 落在日曆缺口"
            assert d.month == 4 and d.year in (y, y + HORIZON_YEARS), d


def load_universe():
    uni = pd.read_csv(dp.UNIVERSE_FILE, dtype={"stock_id": str}, encoding="utf-8-sig",
                      parse_dates=["listed_date", "delisted_date"])
    # 防呆：只留 4 位數、非 0 開頭、非 91 開頭、名稱不含 DR 的代號（證交所清單本來就只有普通股）
    ok = (uni["stock_id"].str.fullmatch(r"[1-9]\d{3}") & ~uni["stock_id"].str.startswith("91")
          & ~uni["stock_name"].astype(str).str.contains("DR"))
    return uni[ok].set_index("stock_id")


def eligibility(uni, stock_id, t):
    """特徵日 t 時是否納入母體；回傳 (是否納入, 原因)。"""
    r = uni.loc[stock_id]
    if pd.notna(r["delisted_date"]) and r["delisted_date"] <= t:
        return False, "特徵日前已下市"
    p = dp.price_series(stock_id)
    if p.empty or dp.start_price(stock_id, t)[0] is None:
        return False, "特徵日前 30 天內無成交"
    if dp.on_emerging(stock_id, t):
        return False, "特徵日時仍在興櫃"
    one_year_ago = t - pd.DateOffset(years=1)
    if pd.notna(r["listed_date"]):
        if r["listed_date"] > one_year_ago:
            return False, "上市未滿一年"
    elif p.index[0] > one_year_ago or dp.emerging_count(stock_id, one_year_ago, t) >= dp.EMERGING_MIN_DAYS:
        return False, "上市未滿一年（已下市公司，以股價推估）"
    return True, ""


def avg_trading_value(stock_id, t):
    """t 之前一年（含 t）的日均成交值（元）；沒有成交的交易日以 0 計入。"""
    cal = dp.trading_calendar()
    days = cal[(cal > t - pd.DateOffset(years=1)) & (cal <= t)]
    p = dp.load("TaiwanStockPrice", stock_id)
    p = p[(p["date"] > t - pd.DateOffset(years=1)) & (p["date"] <= t)]
    return float(p["Trading_money"].sum()) / max(len(days), 1)


def liquidity_keep(values, drop_pct=LIQUIDITY_DROP_PCT):
    """values：Series（index = 代號）。回傳保留的代號集合：由低到高排序（同值依代號），排除最低 floor(drop_pct × N) 檔。"""
    order = values.rename("v").reset_index().sort_values(["v", "stock_id"])
    n_drop = math.floor(drop_pct * len(order))
    return set(order["stock_id"].iloc[n_drop:])


def make_labels(df, pct=TOP_PCT):
    """同一特徵年內產生 y_raw、y_main 與原因（df 為該年流動性過濾後的列）。"""
    df = df.copy()
    rankable = df["報酬狀態"].isin(RANKABLE_STATUS) & df["5 年含息報酬"].notna()
    r = df[rankable].sort_values(["5 年含息報酬", "代號"], ascending=[False, True])
    n = len(r)
    k = math.ceil(pct * n)
    df["報酬排名"] = pd.Series(range(1, n + 1), index=r.index).reindex(df.index).astype("Int64")
    df["可排名母體"] = n
    winners = set(r.index[:k])

    y_raw = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y_main = pd.Series(pd.NA, index=df.index, dtype="Int64")
    why = pd.Series("", index=df.index, dtype=object)
    for i in df.index:
        if not rankable[i]:
            why[i] = f"報酬{df.at[i, '報酬狀態']}，y_raw、y_main 皆 UNKNOWN"
        elif i not in winners:
            y_raw[i], y_main[i] = 0, 0
        else:
            y_raw[i] = 1
            if df.at[i, "CAGR 狀態"] != "OK":
                why[i] = f"報酬前 5% 但營收 CAGR 無法判定（{df.at[i, 'CAGR 說明']}），y_main UNKNOWN"
            else:
                y_main[i] = int(df.at[i, "營收 CAGR"] >= GROWTH_CAGR)
    df["y_raw"], df["y_main"], df["標記說明"] = y_raw, y_main, why

    # （延伸）基本面確認加入 EPS：營收或 EPS 任一通過即可（DESIGN 5.4）
    y_eps = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for i in df.index:
        if pd.isna(y_raw[i]):
            continue
        if y_raw[i] == 0:
            y_eps[i] = 0
            continue
        rev = "UNKNOWN" if pd.isna(y_main[i]) else ("PASS" if y_main[i] == 1 else "FAIL")
        eps = df.at[i, "EPS 判定"]
        if "PASS" in (rev, eps):
            y_eps[i] = 1
        elif rev == "FAIL" and eps == "FAIL":
            y_eps[i] = 0
    df["y_main_eps"] = y_eps
    return df


def build_year(uni, year, rows_cache):
    t = dp.feature_date(year)
    stats = {"特徵年": year, "特徵日": t.date()}
    # 1. 母體
    eligible, reasons = [], {}
    for sid in uni.index:
        if not (dp.PART_DIR / "TaiwanStockPrice" / f"{sid}.parquet").exists():
            reasons["無股價資料"] = reasons.get("無股價資料", 0) + 1
            continue
        ok, why = eligibility(uni, sid, t)
        if ok:
            eligible.append(sid)
        elif why != "特徵日前已下市" and why != "特徵日前 30 天內無成交":
            reasons[why] = reasons.get(why, 0) + 1
    stats["母體（上市滿一年）"] = len(eligible)
    stats["排除：上市未滿一年"] = sum(v for k, v in reasons.items() if k.startswith("上市未滿一年"))
    stats["排除：仍在興櫃"] = reasons.get("特徵日時仍在興櫃", 0)
    # 2. 流動性過濾
    tv = pd.Series({sid: avg_trading_value(sid, t) for sid in eligible}, name="v")
    tv.index.name = "stock_id"
    keep = liquidity_keep(tv)
    stats["排除：流動性最低 10%"] = len(eligible) - len(keep)
    stats["流動性門檻（日均成交值，萬元）"] = round(tv[sorted(keep)].min() / 1e4, 1) if keep else np.nan
    # 3. 特徵、報酬、營收 CAGR（沿用 demo 總表的計算）
    rows = []
    for sid in sorted(keep):
        row = rows_cache(sid, year)
        row["日均成交值（萬元）"] = tv[sid] / 1e4
        rows.append(row)
    df = pd.DataFrame(rows)
    stats["樣本"] = len(df)
    # 4. 標記
    df = make_labels(df)
    df["標記揭曉日"] = dp.horizon_end(year).date()
    stats["可排名"] = int(df["可排名母體"].iloc[0])
    stats["報酬 UNKNOWN（價格異常）"] = int((df["報酬狀態"] == "價格異常").sum())
    stats["下市暫估（仍參與排名）"] = int((df["報酬狀態"] == "下市暫估").sum())
    stats["y_raw = 1"] = int((df["y_raw"] == 1).sum())
    stats["y_main = 1"] = int((df["y_main"] == 1).sum())
    stats["y_main = 0（前 5% 但營收未達 10%）"] = int(((df["y_raw"] == 1) & (df["y_main"] == 0)).sum())
    stats["y_main UNKNOWN（前 5% 但營收無法判定）"] = int(((df["y_raw"] == 1) & df["y_main"].isna()).sum())
    stats["y_main 可判定"] = int(df["y_main"].notna().sum())
    stats["y_main 正例率"] = round(stats["y_main = 1"] / stats["y_main 可判定"], 4) if stats["y_main 可判定"] else np.nan
    stats["（延伸）y_main_eps = 1（營收或 EPS）"] = int((df["y_main_eps"] == 1).sum())
    stats["（延伸）y_main_eps UNKNOWN"] = int(((df["y_raw"] == 1) & df["y_main_eps"].isna()).sum())
    stats["10 項特徵齊全"] = int((df["缺值數"] == 0).sum())
    return df, stats


def eps_check(stock_id, year):
    """（延伸）年度 EPS 條件：第 y − 1 年 → 第 y + 4 年（損益表單季 EPS 四季加總）。

    起點 > 0：5 年年複合成長率 ≥ 10% 為 PASS；起點 ≤ 0：終點 > 0（由虧轉盈）為 PASS。
    期間下市或任一端缺資料為 UNKNOWN。回傳 (判定, 起點, 終點, 原因)。
    """
    delisted = dp.stock_meta(stock_id)["delisted"]
    if delisted is not None and delisted <= dp.horizon_end(year):
        return "UNKNOWN", np.nan, np.nan, f"期間下市（{delisted:%Y-%m-%d}）"
    e0, _, w0 = dp.annual_income(stock_id, year - 1, ["EPS"])
    e1, _, w1 = dp.annual_income(stock_id, year + HORIZON_YEARS - 1, ["EPS"])
    if e0 is None or e1 is None:
        return "UNKNOWN", e0 if e0 is not None else np.nan, e1 if e1 is not None else np.nan, w0 or w1
    if e0 > 0:
        cagr = (e1 / e0) ** (1 / HORIZON_YEARS) - 1 if e1 > 0 else -1.0
        return ("PASS" if cagr >= GROWTH_CAGR else "FAIL"), e0, e1, ""
    return ("PASS" if e1 > 0 else "FAIL"), e0, e1, "起點 EPS ≤ 0，改看是否由虧轉盈"


def accrual_ratio(stock_id, year, industry):
    """規則輸入（不是模型特徵）：應計 ÷ 平均總資產，口徑依 Sloan (1996)，用現金流量表版本。

    應計 ＝ 第 y − 1 年合併淨利 − 營業現金流；平均總資產 ＝（第 y − 1、y − 2 年底總資產）÷ 2，
    缺前一期時改用期末（2013 年缺 2011 年底）。金融保險業不適用（原文排除銀行與保險）。
    回傳 (值, 缺值原因)。
    """
    if industry == "金融保險":
        return np.nan, "金融保險業不適用（Sloan 1996 排除銀行與保險）"
    fy = year - 1
    ni, _, why = dp.annual_income(stock_id, fy, dp.NI_TOTAL_TYPES)
    cfo, _ = dp.statement_value("TaiwanStockCashFlowsStatement", stock_id, pd.Timestamp(f"{fy}-12-31"), dp.CFO_TYPES)
    ta1, _ = dp.statement_value("TaiwanStockBalanceSheet", stock_id, pd.Timestamp(f"{fy}-12-31"), ["TotalAssets"])
    ta0, _ = dp.statement_value("TaiwanStockBalanceSheet", stock_id, pd.Timestamp(f"{fy - 1}-12-31"), ["TotalAssets"])
    if ni is None:
        return np.nan, why
    if cfo is None:
        return np.nan, f"{fy} 年底現金流量表無營業現金流"
    if ta1 is None or ta1 <= 0:
        return np.nan, f"{fy} 年底無總資產"
    avg = (ta1 + ta0) / 2 if ta0 and ta0 > 0 else ta1
    return (ni - cfo) / avg, ""


def analyze_row(sid, year):
    """把 demo_pipeline.analyze 的結果攤平成一列（欄名與 demo 總表一致）。"""
    a = dp.analyze(sid, year)
    row = {"代號": sid, "名稱": a["meta"]["name"], "產業": a["meta"]["industry"] or "（無資料）",
           "特徵年": year, "特徵日": a["t"].date()}
    missing = []
    for k, f in a["features"].items():
        row[k] = f["value"]
        if f["missing"]:
            missing.append(f"{k}：{f['missing']}")
    row.update({
        "缺值數": len(missing), "缺值原因": "；".join(missing),
        "5 年含息報酬": a["ret"]["total"], "報酬狀態": a["ret"]["status"], "報酬說明": a["ret"]["note"],
        "0050 同期含息報酬": a["bm"]["total"],
        "營收 CAGR": a["cagr"], "CAGR 狀態": a["cagr_status"], "CAGR 說明": a["cagr_why"],
    })
    acc, awhy = accrual_ratio(sid, year, row["產業"])
    row.update({"應計/平均總資產": acc, "應計說明": awhy})
    eps, e0, e1, ewhy = eps_check(sid, year)
    row.update({"EPS 起點": e0, "EPS 終點": e1, "EPS 判定": eps, "EPS 說明": ewhy})
    return row


def main():
    if not dp.FULL_MARKET:
        sys.exit("找不到全市場資料（data/raw/parts/），請先執行 collect_data.py")
    check_calendar()
    uni = load_universe()
    frames, summary = [], []
    for y in FEATURE_YEARS:
        df, stats = build_year(uni, y, analyze_row)
        frames.append(df)
        summary.append(stats)
        print(f"{y}：母體 {stats['母體（上市滿一年）']}、樣本 {stats['樣本']}、"
              f"y_raw=1 {stats['y_raw = 1']}、y_main=1 {stats['y_main = 1']}、"
              f"y_main UNKNOWN {stats['y_main UNKNOWN（前 5% 但營收無法判定）']}", flush=True)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(OUT_FILE, index=False)
    s = pd.DataFrame(summary)
    s.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    print(f"共 {len(data)} 列 → {OUT_FILE}；摘要 → {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
