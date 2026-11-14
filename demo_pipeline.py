"""
demo_pipeline.py －Demo 用的 Point-in-Time 計算（只讀本機資料，不呼叫 API）

資料來源（自動選擇，原始資料只在本機、不上傳）：
  - 有 data/raw/parts/（collect_data.py 抓的全市場資料）→ 用全市場
  - 否則用 cache/（verify_data.py 抓的 30 檔抽樣）
照 DESIGN.md（v0.6）計算：
  1. 特徵日 t ＝ 第 y 年 4 月最後交易日（交易日曆取 0050 的股價日期）
  2. 10 項特徵（DESIGN 5.3）：財報與月營收只用第 y − 1 年，股價只用 t（含）以前
     零分母、負權益、不適用時記為缺值並保留原因，不硬算
  3. t → t + 5 年含息報酬（DESIGN 5.4）：每日股價＋公司行動因子（除權息、減資、分割）自行計算，
     終點取最後 20 個交易日（約一個月）含息財富序列的平均；同期 0050 一起算作對照
     市值的股數 ＝ 特徵日的發行股數（TaiwanStockShareholding，2026-11-14 起）；
     特徵日前 30 天內沒有發行股數時，退回「年底股本 ÷ 10，再修正年底後的減資與分割」
  4. 營收 CAGR：第 y − 1 年 → 第 y + 4 年全年營收；期初須為正、5 年資料齊全，
     期間下市不滿 5 年記為 UNKNOWN
  5. 樣本範圍＝上市：特徵日時仍在興櫃（價格不在升降單位上）的公司 × 年不納入（上櫃轉上市無法用價格區分，列為限制）
  6. DESIGN 沒寫、看資料後加的兩個保守檢查（都只標記，不硬修正）：
     - 含息財富單日變動超過漲跌幅限制（興櫃期間除外）→ 報酬標為不可靠
       （減資、分割已處理；剩下的多為長期停牌重整或資料錯誤）
     - 營收比較跨過 2012 → 2013 且相差超過 3 倍 → 疑似 IFRS 口徑改變，記為缺值／UNKNOWN

與正式資料集（build_dataset.py）的差異：
  - 不做「前 5%」排名，排名欄一律寫「待正式資料集」（排名需要上市滿一年與流動性過濾後的母體）
  - 不做流動性過濾（需要全市場成交值分布）
  - 不訓練模型、不接 LLM

build_dataset.py 之後可直接沿用本檔的 feature_date / compute_features /
five_year_return / revenue_cagr 等函式。

輸入：data/raw/parts/{dataset}/{stock_id}.parquet（collect_data.py）或 cache/*.csv（verify_data.py）
輸出：build_table() 回傳的 DataFrame（每列 = 公司 × 特徵年），供 app.py 顯示；
      全市場時另存 data/demo_table.parquet（本機快取，不上傳），資料沒變就不重算
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ───────────── 設定 ─────────────
BASE = Path(__file__).resolve().parent
CACHE_DIR = BASE / "cache"
PART_DIR = BASE / "data" / "raw" / "parts"          # collect_data.py 的逐檔資料
UNIVERSE_FILE = BASE / "data" / "raw" / "universe.csv"
TABLE_CACHE = BASE / "data" / "demo_table.parquet"   # 全市場總表的本機快取
FULL_MARKET = (PART_DIR / "TaiwanStockPrice" / "0050.parquet").exists() and UNIVERSE_FILE.exists()
SOURCE_LABEL = "全市場（data/raw，collect_data.py）" if FULL_MARKET else "30 檔抽樣（cache/，verify_data.py）"
DATA_ATTRIBUTION = "資料來源：FinMind（https://finmindtrade.com/），原始資料來自臺灣證券交易所；僅供非商業研究與教學使用。"
FEATURE_YEARS = list(range(2013, 2021))
HORIZON_YEARS = 5
END_PRICE_DAYS = 20            # 終點取最後 20 個交易日（約一個月）平均；DESIGN v0.6（2026-11-10）由 60 改為 20
GROWTH_CAGR = 0.10             # 營收 5 年年複合成長門檻（事先固定）
FACE_VALUE = 10                # 股數 ＝ 股本 ÷ 面額；面額一律假設 10 元
BENCHMARK_ID = "0050"
MAX_STALE_DAYS = 30            # 特徵日前 30 天內沒有成交，視為當時未在交易
IFRS_FIRST_YEAR = 2013         # 台灣上市櫃公司 IFRS 首年；跨過這年的營收比較要檢查口徑
IFRS_JUMP = 3                  # 2012 → 2013 營收相差超過 3 倍視為口徑改變
# 漲跌幅限制：2015-06-01 前 7%、之後 10%；含息財富（已扣除除權息、減資、分割）單日超過即價格異常
LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
JUMP_TOL_OLD, JUMP_TOL_NEW = 0.075, 0.105
# 樣本範圍＝上市；特徵日前 90 天內有 ≥ 5 天興櫃價（不在升降單位上）→ 當時尚未上市，不納入
EMERGING_LOOKBACK_DAYS, EMERGING_MIN_DAYS = 90, 5

FEATURES = [
    "市值（億元）", "FCF 殖利率", "帳面/市值", "本益比", "ROE",
    "毛利率", "年營收成長率", "營業現金流/淨利", "負債比", "過去 12 個月含息報酬",
]

# 淨利欄位：前者歸屬母公司（ROE 用），後者合併淨利（營業現金流/淨利用，與合併現金流口徑一致）
NI_PARENT_TYPES = ["EquityAttributableToOwnersOfParent", "IncomeAfterTaxes", "NetIncome", "IncomeAfterTax"]
NI_TOTAL_TYPES = ["IncomeAfterTaxes", "NetIncome", "IncomeAfterTax", "EquityAttributableToOwnersOfParent"]
EQUITY_TYPES = ["EquityAttributableToOwnersOfParent", "Equity"]
CFO_TYPES = ["CashFlowsFromOperatingActivities", "NetCashInflowFromOperatingActivities"]


# ───────────── 讀快取 ─────────────
@lru_cache(maxsize=None)
def load(dataset, data_id="ALL"):
    """讀本機資料；檔案不存在或是空檔回傳空表。不會呼叫 API。

    個股資料：全市場模式讀 data/raw/parts/{dataset}/{data_id}.parquet，否則讀 cache/{dataset}_{data_id}.csv。
    股票清單與下市名單（data_id ＝ "ALL"）一律讀 cache/。
    """
    part = PART_DIR / dataset / f"{data_id}.parquet"
    if FULL_MARKET and data_id != "ALL":
        if not part.exists():
            return pd.DataFrame()
        df = pd.read_parquet(part)
    else:
        path = CACHE_DIR / f"{dataset}_{data_id}.csv"
        if not path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    if "stock_id" in df.columns:
        df["stock_id"] = df["stock_id"].astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


FETCH_BENCHMARK_CMD = ('python -c "import verify_data as v; '
                       '[v.fetch(d, \'0050\', v.START_DATE) for d in (\'TaiwanStockPrice\', \'TaiwanStockDividendResult\')]"')


def missing_cache():
    """回傳 Demo 需要但本機沒有的檔案（verify_data.py 不會抓 0050，需另外補抓）。"""
    if FULL_MARKET:
        need = ["TaiwanStockInfo_ALL.csv", "TaiwanStockDelisting_ALL.csv"]
        return [n for n in need if not (CACHE_DIR / n).exists()]
    need = [f"TaiwanStockPrice_{BENCHMARK_ID}.csv", f"TaiwanStockDividendResult_{BENCHMARK_ID}.csv",
            "TaiwanStockInfo_ALL.csv", "TaiwanStockDelisting_ALL.csv"]
    missing = [n for n in need if not (CACHE_DIR / n).exists()]
    if not any(CACHE_DIR.glob("TaiwanStockPrice_*.csv")):
        missing.append("30 檔抽樣股價")
    return missing


def demo_stock_ids():
    """有股價資料的股票（不含 0050）：全市場模式取研究範圍清單，否則取 cache/ 的 30 檔抽樣。"""
    if FULL_MARKET:
        uni = pd.read_csv(UNIVERSE_FILE, dtype={"stock_id": str}, encoding="utf-8-sig")
        return sorted(s for s in uni["stock_id"] if (PART_DIR / "TaiwanStockPrice" / f"{s}.parquet").exists())
    ids = {p.stem.split("_", 1)[1] for p in CACHE_DIR.glob("TaiwanStockPrice_*.csv")}
    return sorted(ids - {BENCHMARK_ID})


@lru_cache(maxsize=None)
def stock_meta(stock_id):
    """名稱、市場、產業（產業類別取 TaiwanStockInfo 現況，已下市者無資料）、下市日。"""
    info = load("TaiwanStockInfo")
    dl = load("TaiwanStockDelisting")
    row = info[info["stock_id"] == stock_id].sort_values("date") if len(info) else info
    row = row.iloc[[-1]] if len(row) else row   # 同一檔可能有興櫃、上市櫃多筆，取最新一筆
    drow = dl[dl["stock_id"] == stock_id] if len(dl) else dl
    delisted = drow["date"].iloc[0] if len(drow) else None
    if len(row):
        name, industry, market = row["stock_name"].iloc[0], row["industry_category"].iloc[0], row["type"].iloc[0]
    else:
        name = drow["stock_name"].iloc[0] if len(drow) else stock_id
        industry, market = None, None
    market = {"twse": "上市", "tpex": "上櫃"}.get(market, "不明（已下市）")
    return {"name": name, "industry": industry, "market": market, "delisted": delisted}


# ───────────── 交易日曆與特徵日 ─────────────
@lru_cache(maxsize=None)
def trading_calendar():
    """以 0050 的成交日期當全市場交易日曆。"""
    p = load("TaiwanStockPrice", BENCHMARK_ID)
    return pd.DatetimeIndex(sorted(p.loc[p["close"] > 0, "date"].unique()))


def feature_date(year):
    """第 year 年 4 月的最後一個交易日。"""
    cal = trading_calendar()
    april = cal[(cal.year == year) & (cal.month == 4)]
    return april.max()


def horizon_end(year):
    """觀察期終點：第 year + 5 年 4 月的最後一個交易日。"""
    return feature_date(year + HORIZON_YEARS)


# ───────────── 股價、除權息與含息財富序列 ─────────────
@lru_cache(maxsize=None)
def price_series(stock_id):
    """每日收盤價（只保留 > 0），index 為日期。"""
    p = load("TaiwanStockPrice", stock_id)
    if p.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    p = p[p["close"] > 0].drop_duplicates("date", keep="last").sort_values("date")
    return p.set_index("date")["close"].astype(float)


def _ratio_series(df, before_col, after_col):
    """before ÷ after，同日多筆相乘；回傳以日期為 index 的 Series。"""
    if df.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    ok = (df[before_col] > 0) & (df[after_col] > 0)
    f = (df.loc[ok, before_col] / df.loc[ok, after_col]).groupby(df.loc[ok, "date"]).prod()
    f.index = pd.DatetimeIndex(f.index)
    return f


@lru_cache(maxsize=None)
def capital_reductions(stock_id):
    """減資紀錄（date ＝ 減資後恢復買賣日），附「股數比 k」＝ 減資後股數 ÷ 減資前股數。

    參考價公式（交易所）：
      彌補虧損：P_ref ＝ P_old ÷ k                    → k ＝ P_old ÷ P_ref
      現金減資：P_ref ＝ (P_old − 退還現金) ÷ k，每股退 10 ×（1 − k）元
                                                     → k ＝ (P_old − 10) ÷ (P_ref − 10)
    現金減資的 k 假設按面額 10 元退款；算不出合理值（不在 0～1）時退回 P_old ÷ P_ref。
    """
    c = load("TaiwanStockCapitalReductionReferencePrice", stock_id)
    if c.empty:
        return pd.DataFrame(columns=["date", "p_old", "p_ref", "cash", "k"])
    c = c.rename(columns={"ClosingPriceonTheLastTradingDay": "p_old", "PostReductionReferencePrice": "p_ref"})
    c = c[(c["p_old"] > 0) & (c["p_ref"] > 0)].copy()
    reason = c["ReasonforCapitalReduction"].astype(str)
    c["cash"] = reason.str.contains("Cash|現金")
    k_loss = c["p_old"] / c["p_ref"]
    with np.errstate(divide="ignore", invalid="ignore"):
        k_cash = (c["p_old"] - FACE_VALUE) / (c["p_ref"] - FACE_VALUE)
    k_cash = k_cash.where((k_cash > 0) & (k_cash <= 1), k_loss)
    c["k"] = np.where(c["cash"], k_cash, k_loss)
    return c[["date", "p_old", "p_ref", "cash", "k"]].sort_values("date").reset_index(drop=True)


@lru_cache(maxsize=None)
def splits(stock_id):
    """股票分割／面額變更（date ＝ 恢復買賣日），k ＝ 分割後股數 ÷ 分割前股數 ＝ 前價 ÷ 後價。"""
    s = load("TaiwanStockSplitPrice", stock_id)
    if s.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return _ratio_series(s, "before_price", "after_price").sort_index()


def _tick(p):
    """台股升降單位（2005 年以來不變）：<10 元 0.01、<50 0.05、<100 0.1、<500 0.5、<1000 1、其餘 5。"""
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1], 5)


@lru_cache(maxsize=None)
def emerging_days(stock_id):
    """疑似興櫃交易的日期：開高低收任一不在升降單位上。

    FinMind 的日股價把興櫃與上市櫃期間接在同一條序列；興櫃以議價成交、報的是均價（例如 180.61），
    不受升降單位與漲跌幅限制。上市櫃股票的成交價一定落在升降單位上，藉此區分。
    """
    p = load("TaiwanStockPrice", stock_id)
    if p.empty or stock_id == BENCHMARK_ID:   # ETF 升降單位不同，0050 不做此檢查
        return pd.DatetimeIndex([])
    p = p[p["close"] > 0]
    bad =np.zeros(len(p), dtype=bool)
    for col in ["open", "max", "min", "close"]:
        v = p[col].to_numpy(dtype=float)
        q = v / _tick(v)
        bad |= (v > 0) & (np.abs(q - np.round(q)) > 1e-6)
    return pd.DatetimeIndex(p.loc[bad, "date"]).unique().sort_values()


def emerging_count(stock_id, start, end):
    """(start, end] 之間疑似興櫃交易的天數。"""
    e = emerging_days(stock_id)
    return int(((e > start) & (e <= end)).sum())


def on_emerging(stock_id, t):
    """特徵日 t 時是否仍在興櫃（尚未上市櫃）：t 前 EMERGING_LOOKBACK_DAYS 天內有 ≥ EMERGING_MIN_DAYS 天疑似興櫃價。"""
    return emerging_count(stock_id, t - pd.Timedelta(days=EMERGING_LOOKBACK_DAYS), t) >= EMERGING_MIN_DAYS


@lru_cache(maxsize=None)
def dividend_factors(stock_id):
    """所有公司行動的報酬因子（同日多筆相乘），用於含息財富序列：

      除權息：除權息前價格 ÷ 除權息後價格（參考價）
      減資  ：減資前最後交易日收盤價 ÷ 減資後參考價（現金減資等於把退回的現金再投入）
      分割  ：分割前價格 ÷ 分割後價格（股票分割、面額變更）
    """
    parts = []
    d = load("TaiwanStockDividendResult", stock_id)
    if not d.empty:
        d = d.assign(after=d["after_price"].where(d["after_price"] > 0, d["reference_price"]))
        parts.append(_ratio_series(d, "before_price", "after"))
    c = capital_reductions(stock_id)
    if len(c):
        parts.append(_ratio_series(c, "p_old", "p_ref"))
    s = splits(stock_id)
    if len(s):
        parts.append(s)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    f = pd.concat(parts)
    return f.groupby(level=0).prod().sort_index()


def share_multiplier(stock_id, after, upto):
    """股本日期 after（不含）→ 特徵日 upto（含）之間的減資與分割，對「股本 ÷ 面額」股數的修正倍數。

    分割／面額變更：股本不變、股數變多，任何時點之後都要乘上（面額不是 10 元的公司也靠這個修正）。
    減資：年底股本已反映年底前的減資，只修正 after 之後才恢復買賣的減資。
    """
    s = splits(stock_id)
    m_split = float(s[s.index <= upto].prod()) if len(s) else 1.0
    c = capital_reductions(stock_id)
    c = c[(c["date"] > after) & (c["date"] <= upto)] if len(c) else c
    m_red = float(c["k"].prod()) if len(c) else 1.0
    notes = []
    if m_split != 1.0:
        notes.append(f"分割／面額變更，股數 × {m_split:.4g}")
    for _, r in c.iterrows():
        notes.append(f"{r['date']:%Y-%m-%d} {'現金減資' if r['cash'] else '減資'}，股數 × {r['k']:.4g}")
    return m_split * m_red, "；".join(notes)


def wealth_series(stock_id, start, end):
    """start（含）→ end（含）的含息財富序列：收盤價 × 期間內已發生的除權息因子累乘。

    回傳 DataFrame：close（只看股價）、wealth（含息財富，起點與股價相同）、factor（累積因子）。
    """
    p = price_series(stock_id)
    p = p[(p.index >= start) & (p.index <= end)]
    if p.empty:
        return pd.DataFrame(columns=["close", "wealth", "factor"])
    f = dividend_factors(stock_id)
    f = f[(f.index > p.index[0]) & (f.index <= end)]
    cum = f.cumprod()
    cum = cum.reindex(cum.index.union(p.index)).ffill().fillna(1.0).reindex(p.index)
    return pd.DataFrame({"close": p, "wealth": p * cum, "factor": cum})


def price_jumps(ws, stock_id=None):
    """含息財富單日變動超過當時漲跌幅限制的日期 → 疑似未處理的公司行動或資料錯誤。

    有 stock_id 時，當天或前一個交易日是興櫃價（沒有漲跌幅限制）的跳動不算。
    """
    if len(ws) < 2:
        return []
    r = ws["wealth"] / ws["wealth"].shift(1) - 1
    tol = np.where(r.index < LIMIT_CHANGE_DATE, JUMP_TOL_OLD, JUMP_TOL_NEW)
    flag = (r.abs() > tol) & r.notna()
    if stock_id is not None and flag.any():
        emer = pd.Series(ws.index.isin(emerging_days(stock_id)), index=ws.index)
        flag &= ~(emer | emer.shift(1, fill_value=False))
    bad = r[flag]
    return [d.strftime("%Y-%m-%d") for d in bad.index]


def issued_shares(stock_id, t):
    """特徵日 t（含）以前最後一筆發行股數（TaiwanStockShareholding，證交所每日公布）。

    超過 MAX_STALE_DAYS 天沒有更新（例如資料中斷多年）回傳 (None, None)，由呼叫端退回股本推估。
    發行股數已反映增資、減資、分割，不需要再修正；少數公司（例如 2002）含上市特別股，列為限制。
    """
    sh = load("TaiwanStockShareholding", stock_id)
    if sh.empty:
        return None, None
    sh = sh[(sh["date"] <= t) & (sh["NumberOfSharesIssued"] > 0)].sort_values("date")
    if sh.empty or (t - sh["date"].iloc[-1]).days > MAX_STALE_DAYS:
        return None, None
    return sh["date"].iloc[-1], float(sh["NumberOfSharesIssued"].iloc[-1])


def start_price(stock_id, t):
    """特徵日 t（含）以前最後一筆收盤價；若距 t 超過 MAX_STALE_DAYS 天，視為當時未在交易。"""
    p = price_series(stock_id)
    p = p[p.index <= t]
    if p.empty or (t - p.index[-1]).days > MAX_STALE_DAYS:
        return None, None
    return p.index[-1], float(p.iloc[-1])


def five_year_return(stock_id, year):
    """t → t + 5 年的含息報酬與只看股價的報酬；終點取最後 20 個交易日平均。

    status：
      OK          ＝ 可靠
      下市暫估     ＝ 期間停止交易，以最後交易日價格暫估並標記（仍參與報酬排名，DESIGN 5.4、第 12 節）
      價格異常     ＝ 含息財富單日變動超過漲跌幅限制（長期停牌重整、資料錯誤等），報酬不可靠
      未在交易     ＝ 特徵日時沒有成交資料
    """
    t, end = feature_date(year), horizon_end(year)
    d0, p0 = start_price(stock_id, t)
    out = {"start_date": d0, "start_price": p0, "end_date": end, "total": np.nan,
           "price_only": np.nan, "status": "未在交易", "note": "", "jumps": []}
    if d0 is None:
        return out
    ws = wealth_series(stock_id, d0, end)
    out["jumps"] = price_jumps(ws, stock_id)
    last = ws.index[-1]
    if (end - last).days > MAX_STALE_DAYS:
        # 期間下市或長期停止交易：不平均，直接用最後交易日價格暫估
        out.update(total=ws["wealth"].iloc[-1] / p0 - 1, price_only=ws["close"].iloc[-1] / p0 - 1,
                   status="下市暫估", end_date=last,
                   note=f"{last:%Y-%m-%d} 後無成交，以最後交易價暫估（下市對價無法取得）")
        return out
    tail = ws.iloc[-END_PRICE_DAYS:]
    out.update(total=tail["wealth"].mean() / p0 - 1, price_only=tail["close"].mean() / p0 - 1,
               status="OK", note=f"終點取 {tail.index[0]:%Y-%m-%d}～{tail.index[-1]:%Y-%m-%d} 共 {len(tail)} 日平均")
    if out["jumps"]:
        out["status"] = "價格異常"
        out["note"] = f"{', '.join(out['jumps'][:3])} 單日變動超過漲跌幅限制，可能是長期停牌重整或資料錯誤"
    return out


def trailing_12m_return(stock_id, t):
    """特徵：t 前 12 個月的含息報酬（只用 t 以前的股價），不做平均。"""
    d1, p1 = start_price(stock_id, t)
    d0, p0 = start_price(stock_id, t - pd.DateOffset(years=1))
    if d1 is None or d0 is None:
        return np.nan, "一年前尚無成交資料"
    ws = wealth_series(stock_id, d0, d1)
    if price_jumps(ws, stock_id):
        return np.nan, "期間單日變動超過漲跌幅限制（價格異常），報酬不可靠"
    return ws["wealth"].iloc[-1] / p0 - 1, ""


# ───────────── 月營收 ─────────────
@lru_cache(maxsize=None)
def annual_revenue(stock_id):
    """每年 1–12 月營收合計與月數（同一個月重複時取最後一筆）。"""
    r = load("TaiwanStockMonthRevenue", stock_id)
    if r.empty:
        return pd.DataFrame(columns=["revenue", "months"])
    r = r.drop_duplicates(["revenue_year", "revenue_month"], keep="last")
    g = r.groupby("revenue_year").agg(revenue=("revenue", "sum"), months=("revenue_month", "nunique"))
    g.index = g.index.astype(int)
    return g


def full_year_revenue(stock_id, year):
    """回傳 (第 year 年全年營收, 缺值原因)；12 個月不齊就不算。"""
    g = annual_revenue(stock_id)
    if year not in g.index:
        return None, f"{year} 年無月營收資料"
    if g.loc[year, "months"] < 12:
        return None, f"{year} 年月營收只有 {int(g.loc[year, 'months'])}/12 個月"
    return float(g.loc[year, "revenue"]), ""


def ifrs_break(stock_id, y0, y1):
    """比較區間跨過 2012 → 2013（IFRS 首年）且兩年營收相差超過 3 倍時回傳原因。

    2013 年起改採 IFRS，部分公司首次合併子公司，月營收口徑跟著改變（例如 2605 新興 2012 → 2013 營收跳 30 倍），
    跨口徑比較會把會計口徑變化誤當成營收成長。
    """
    if not (y0 <= IFRS_FIRST_YEAR - 1 < y1):
        return ""
    a, _ = full_year_revenue(stock_id, IFRS_FIRST_YEAR - 1)
    b, _ = full_year_revenue(stock_id, IFRS_FIRST_YEAR)
    if a and b and a > 0 and not (1 / IFRS_JUMP < b / a < IFRS_JUMP):
        return (f"{IFRS_FIRST_YEAR - 1}→{IFRS_FIRST_YEAR} 年營收相差 {b / a:.1f} 倍，"
                f"疑似 IFRS 上路後口徑改變（個體→合併），跨口徑比較不可靠")
    return ""


def revenue_cagr(stock_id, year):
    """第 y − 1 年 → 第 y + 4 年營收 5 年年複合成長率；回傳 (cagr, status, 原因)。"""
    delisted = stock_meta(stock_id)["delisted"]
    if delisted is not None and delisted <= horizon_end(year):
        return np.nan, "UNKNOWN", f"期間下市（{delisted:%Y-%m-%d}），不滿 5 年，不用較短期間硬套"
    vals = {}
    for yr in range(year - 1, year + HORIZON_YEARS):
        v, why = full_year_revenue(stock_id, yr)
        if v is None:
            return np.nan, "UNKNOWN", why
        vals[yr] = v
    start, end = vals[year - 1], vals[year + HORIZON_YEARS - 1]
    why = ifrs_break(stock_id, year - 1, year + HORIZON_YEARS - 1)
    if why:
        return np.nan, "UNKNOWN", why
    if start <= 0:
        return np.nan, "UNKNOWN", f"期初（{year - 1} 年）營收 ≤ 0，無法計算成長率"
    if end < 0:
        return np.nan, "UNKNOWN", f"期末（{year + 4} 年）營收為負，無法計算成長率"
    return (end / start) ** (1 / HORIZON_YEARS) - 1, "OK", ""


# ───────────── 財報 ─────────────
def statement_value(dataset, stock_id, date, types):
    """取某一期的欄位值；types 依序嘗試，回傳 (值, 使用的欄位)，都沒有則 (None, None)。"""
    df = load(dataset, stock_id)
    if df.empty:
        return None, None
    rows = df[df["date"] == pd.Timestamp(date)]
    for t in types:
        v = rows.loc[rows["type"] == t, "value"]
        if len(v):
            return float(v.iloc[0]), t
    return None, None


def annual_income(stock_id, year, types):
    """損益表是單季值：第 year 年四季加總；同一欄位四季都要有才算。回傳 (值, 欄位, 缺值原因)。"""
    df = load("TaiwanStockFinancialStatements", stock_id)
    if df.empty:
        return None, None, "損益表無資料"
    quarters = pd.to_datetime([f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"])
    rows = df[df["date"].isin(quarters)]
    for t in types:
        v = rows[rows["type"] == t].drop_duplicates("date")
        if len(v) == 4:
            return float(v["value"].sum()), t, ""
    n = rows["date"].nunique()
    return None, None, f"{year} 年損益表只有 {n}/4 季" if n < 4 else f"{year} 年損益表缺 {types[0]} 欄位"


# ───────────── 10 項特徵 ─────────────
def _feat(value, formula, source, available, reason=""):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        value = np.nan
    return {"value": value, "formula": formula, "source": source,
            "available": available, "missing": reason if pd.isna(value) else ""}


def compute_features(stock_id, year):
    """第 year 年特徵日的 10 項特徵；每項附計算口徑、來源資料、公告時間與缺值原因。"""
    t = feature_date(year)
    fy = year - 1
    fy_end = pd.Timestamp(f"{fy}-12-31")
    annual_deadline = f"年報法定期限 {year}-03-31"
    bs_src = f"資產負債表 {fy}-12-31"
    is_src = f"損益表 {fy} 年 Q1–Q4 單季加總"
    cf_src = f"現金流量表 {fy} 年全年累計（12-31）"
    feats = {}

    # 股價與股數 → 市值（股數優先用特徵日的發行股數，沒有才退回股本推估；DESIGN 5.3）
    d_t, p_t = start_price(stock_id, t)
    price_src = f"每日股價 {d_t:%Y-%m-%d} 收盤" if d_t is not None else "每日股價"
    d_sh, issued = issued_shares(stock_id, t)
    if p_t is None:
        mcap, why, formula, src = None, "特徵日前 30 天內無成交", "股價 × 發行股數", price_src
    elif issued is not None:
        mcap, why = p_t * issued, ""
        formula, src = "股價 × 特徵日的發行股數", f"{price_src}＋外資持股表 {d_sh:%Y-%m-%d} 發行股數"
    else:
        shares_cap, _ = statement_value("TaiwanStockBalanceSheet", stock_id, fy_end, ["OrdinaryShare", "CapitalStock"])
        formula = f"（無發行股數，退回推估）股價 × 股本 ÷ 面額 {FACE_VALUE} 元（年底後的減資、分割修正股數）"
        src = f"{price_src}＋{bs_src} 股本"
        if shares_cap is None or shares_cap <= 0:
            mcap, why = None, f"特徵日前 {MAX_STALE_DAYS} 天內無發行股數，{fy} 年底資產負債表也無股本"
        else:
            mult, adj_note = share_multiplier(stock_id, fy_end, t)
            mcap, why = p_t * shares_cap / FACE_VALUE * mult, ""
            if adj_note:
                src += f"（{adj_note}）"
    feats["市值（億元）"] = _feat(mcap / 1e8 if mcap else None, formula, src, "股價、發行股數：特徵日", why)

    # FCF 殖利率
    cfo, _ = statement_value("TaiwanStockCashFlowsStatement", stock_id, fy_end, CFO_TYPES)
    capex, _ = statement_value("TaiwanStockCashFlowsStatement", stock_id, fy_end, ["PropertyAndPlantAndEquipment"])
    if cfo is None:
        v, why = None, f"{fy} 年底現金流量表無營業現金流"
    elif capex is None:
        v, why = None, f"{fy} 年底現金流量表無「取得不動產、廠房及設備」欄位"
    elif not mcap:
        v, why = None, "市值缺值"
    else:
        v, why = (cfo - abs(capex)) / mcap, ""   # 資本支出在現金流量表為負數，統一取絕對值再相減
    feats["FCF 殖利率"] = _feat(v, "（營業現金流 − |資本支出|）÷ 市值", f"{cf_src}＋市值", annual_deadline, why)

    # 帳面/市值
    eq, _ = statement_value("TaiwanStockBalanceSheet", stock_id, fy_end, EQUITY_TYPES)
    if eq is None:
        v, why = None, f"{fy} 年底資產負債表無權益"
    elif eq <= 0:
        v, why = None, "負權益（或為 0），比率無意義"
    elif not mcap:
        v, why = None, "市值缺值"
    else:
        v, why = eq / mcap, ""
    feats["帳面/市值"] = _feat(v, "歸屬母公司權益 ÷ 市值", f"{bs_src}＋市值", annual_deadline, why)

    # 本益比（TaiwanStockPER 特徵日當天或之前最近一筆）
    per = load("TaiwanStockPER", stock_id)
    per = per[(per["date"] <= t) & (per["date"] > t - pd.DateOffset(days=MAX_STALE_DAYS))] if len(per) else per
    if per.empty:
        v, why, src = None, "特徵日前 30 天內無本益比資料", "TaiwanStockPER"
    else:
        last = per.sort_values("date").iloc[-1]
        src = f"TaiwanStockPER {last['date']:%Y-%m-%d}"
        v, why = (float(last["PER"]), "") if last["PER"] > 0 else (None, "本益比為 0 或負（通常是虧損），不適用")
    feats["本益比"] = _feat(v, "TaiwanStockPER 當日值", src, "特徵日當天（含）以前", why)

    # ROE
    ni_p, _, ni_why = annual_income(stock_id, fy, NI_PARENT_TYPES)
    eq_prev, _ = statement_value("TaiwanStockBalanceSheet", stock_id, pd.Timestamp(f"{fy - 1}-12-31"), EQUITY_TYPES)
    if ni_p is None:
        v, why, formula = None, ni_why, "淨利 ÷ 平均權益"
    elif eq is None:
        v, why, formula = None, f"{fy} 年底資產負債表無權益", "淨利 ÷ 平均權益"
    else:
        if eq_prev is None:
            avg_eq = eq
            formula = "淨利 ÷ 期末權益（" + ("2013 年缺前一期權益" if year == 2013 else f"缺 {fy - 1} 年底權益") + "，改用期末）"
        else:
            avg_eq, formula = (eq + eq_prev) / 2, "淨利 ÷ 平均權益（期初＋期末）÷ 2"
        v, why = (ni_p / avg_eq, "") if avg_eq > 0 else (None, "權益 ≤ 0，比率無意義")
    feats["ROE"] = _feat(v, formula, f"{is_src}（歸屬母公司淨利）＋{bs_src}", annual_deadline, why)

    # 毛利率
    rev_is, _, rev_why = annual_income(stock_id, fy, ["Revenue"])
    gp, _, gp_why = annual_income(stock_id, fy, ["GrossProfit"])
    if rev_is is None:
        v, why = None, rev_why
    elif gp is None:
        v, why = None, "無毛利欄位（如金融業）或季度不齊，不適用" if "欄位" in gp_why else gp_why
    elif rev_is <= 0:
        v, why = None, "營收 ≤ 0，零分母"
    else:
        v, why = gp / rev_is, ""
    feats["毛利率"] = _feat(v, "毛利 ÷ 營收", is_src, annual_deadline, why)

    # 年營收成長率（月營收）
    r1, w1 = full_year_revenue(stock_id, fy)
    r0, w0 = full_year_revenue(stock_id, fy - 1)
    if r1 is None or r0 is None:
        v, why = None, w1 or w0
    elif ifrs_break(stock_id, fy - 1, fy):
        v, why = None, ifrs_break(stock_id, fy - 1, fy)
    elif r0 <= 0:
        v, why = None, f"{fy - 1} 年營收 ≤ 0，零分母"
    else:
        v, why = r1 / r0 - 1, ""
    feats["年營收成長率"] = _feat(v, f"{fy} 年營收 ÷ {fy - 1} 年營收 − 1", f"月營收 {fy - 1}/01–{fy}/12",
                              f"12 月營收法定期限 {year}-01-10", why)

    # 營業現金流/淨利
    ni_t, _, nt_why = annual_income(stock_id, fy, NI_TOTAL_TYPES)
    if cfo is None:
        v, why = None, f"{fy} 年底現金流量表無營業現金流"
    elif ni_t is None:
        v, why = None, nt_why
    elif ni_t <= 0:
        v, why = None, "淨利 ≤ 0，比率方向會反轉，不計算"
    else:
        v, why = cfo / ni_t, ""
    feats["營業現金流/淨利"] = _feat(v, "營業現金流 ÷ 合併淨利", f"{cf_src}＋{is_src}", annual_deadline, why)

    # 負債比
    liab, _ = statement_value("TaiwanStockBalanceSheet", stock_id, fy_end, ["Liabilities"])
    assets, _ = statement_value("TaiwanStockBalanceSheet", stock_id, fy_end, ["TotalAssets"])
    if liab is None or assets is None:
        v, why = None, f"{fy} 年底資產負債表無負債或資產總額"
    elif assets <= 0:
        v, why = None, "資產總額 ≤ 0，零分母"
    else:
        v, why = liab / assets, ""
    feats["負債比"] = _feat(v, "總負債 ÷ 總資產", bs_src, annual_deadline, why)

    # 過去 12 個月含息報酬
    v, why = trailing_12m_return(stock_id, t)
    feats["過去 12 個月含息報酬"] = _feat(v, "（t 收盤 × 除權息因子累乘）÷ 一年前收盤 − 1",
                                    f"每日股價＋除權息結果 {(t - pd.DateOffset(years=1)):%Y-%m-%d}～{t:%Y-%m-%d}",
                                    "特徵日當天（含）以前", why)
    return feats


# ───────────── 判定與總表 ─────────────
def growth_verdict(cagr_status, cagr):
    """只判定營收成長條件；30 檔是抽樣，報酬排名留給全量母體。"""
    if cagr_status != "OK":
        return "無法判定"
    return "符合成長條件" if cagr >= GROWTH_CAGR else "不符合成長條件"


@lru_cache(maxsize=None)
def benchmark_return(year):
    return five_year_return(BENCHMARK_ID, year)


def analyze(stock_id, year):
    """單一公司 × 特徵年的完整結果（app 的時光機用）。

    特徵日時未在交易，或仍在興櫃（尚未上市）則回傳 None：樣本範圍是上市公司，
    興櫃公司等真正上市之後的特徵年才納入。
    """
    t = feature_date(year)
    d_t, _ = start_price(stock_id, t)
    if d_t is None or (stock_id != BENCHMARK_ID and on_emerging(stock_id, t)):
        return None
    meta = stock_meta(stock_id)
    first = price_series(stock_id).index[0]
    one_year_ago = t - pd.DateOffset(years=1)
    # 上市滿一年：一年前就有成交，且這一年內沒有興櫃價
    listed_1y = first <= one_year_ago and emerging_count(stock_id, one_year_ago, t) < EMERGING_MIN_DAYS
    feats = compute_features(stock_id, year)
    ret = five_year_return(stock_id, year)
    bm = benchmark_return(year)
    cagr, cagr_status, cagr_why = revenue_cagr(stock_id, year)
    return {"stock_id": stock_id, "year": year, "t": t, "meta": meta, "listed_1y": listed_1y,
            "features": feats, "ret": ret, "bm": bm, "cagr": cagr, "cagr_status": cagr_status,
            "cagr_why": cagr_why, "verdict": growth_verdict(cagr_status, cagr)}


def build_table(progress=None):
    """每檔每年一列：10 項特徵、5 年含息報酬、0050 同期、營收 CAGR、判定與缺值原因。

    全市場模式會把結果存到 data/demo_table.parquet；資料沒更新就直接讀，不重算（全市場重算約需數分鐘）。
    progress：可選的回呼函式 progress(已完成檔數, 總檔數)，給網頁顯示進度條。
    """
    if FULL_MARKET and TABLE_CACHE.exists() and TABLE_CACHE.stat().st_mtime >= _data_mtime():
        return pd.read_parquet(TABLE_CACHE)
    rows = []
    ids = demo_stock_ids()
    for i, sid in enumerate(ids, 1):
        if progress:
            progress(i, len(ids))
        for y in FEATURE_YEARS:
            a = analyze(sid, y)
            if a is None:
                continue
            row = {"代號": sid, "名稱": a["meta"]["name"], "市場": a["meta"]["market"],
                   "產業": a["meta"]["industry"] or "（無資料）", "特徵年": y, "特徵日": a["t"].date()}
            missing = []
            for k, f in a["features"].items():
                row[k] = f["value"]
                if f["missing"]:
                    missing.append(f"{k}：{f['missing']}")
            row.update({
                "缺值數": len(missing),
                "5 年含息報酬": a["ret"]["total"],
                "只看股價報酬": a["ret"]["price_only"],
                "0050 同期含息報酬": a["bm"]["total"],
                "超越 0050": a["ret"]["total"] - a["bm"]["total"],
                "報酬狀態": a["ret"]["status"],
                "營收 CAGR": a["cagr"],
                "CAGR 狀態": a["cagr_status"],
                "判定": a["verdict"],
                "排名": "待正式資料集",
                "樣本備註": "" if a["listed_1y"] else "上市未滿一年（正式樣本不納入）",
                "缺值原因": "；".join(missing),
                "報酬說明": a["ret"]["note"],
                "CAGR 說明": a["cagr_why"],
            })
            rows.append(row)
    table = pd.DataFrame(rows)
    if FULL_MARKET:
        table.to_parquet(TABLE_CACHE, index=False)
    return table


def _data_mtime():
    """全市場資料或計算程式的最後更新時間（資料夾修改時間＋本檔修改時間；改了計算方式也要重算）。"""
    times = [d.stat().st_mtime for d in PART_DIR.iterdir() if d.is_dir()]
    return max(times + [UNIVERSE_FILE.stat().st_mtime, Path(__file__).stat().st_mtime])


def revenue_bars(stock_id, year):
    """第 y − 2 年 → 第 y + 4 年的全年營收（不足 12 個月的年份標示月數）。"""
    g = annual_revenue(stock_id)
    out = []
    for yr in range(year - 2, year + HORIZON_YEARS):
        if yr in g.index:
            out.append({"年度": yr, "營收（億元）": g.loc[yr, "revenue"] / 1e8, "月數": int(g.loc[yr, "months"])})
        else:
            out.append({"年度": yr, "營收（億元）": np.nan, "月數": 0})
    return pd.DataFrame(out)


# ───────────── 例子挑選（「我們要找什麼樣的股票」區塊） ─────────────
def pick_examples(table):
    """從總表自動挑三個例子；同一檔股票不重複使用。

    A：報酬可靠、營收 CAGR ≥ 10%，超越 0050 最多者
    B：報酬可靠、營收 CAGR 可判定且 < 10%、報酬高於 0050，報酬最高者
    C：期間下市或資料不足（報酬不可靠或 CAGR UNKNOWN）；優先選特徵最齊、有下市紀錄者
    """
    ok = table[(table["報酬狀態"] == "OK") & table["樣本備註"].eq("")]
    used = set()

    a = ok[(ok["CAGR 狀態"] == "OK") & (ok["營收 CAGR"] >= GROWTH_CAGR)].sort_values("超越 0050", ascending=False)
    ex_a = a.iloc[0] if len(a) else None
    if ex_a is not None:
        used.add(ex_a["代號"])

    b = ok[(ok["CAGR 狀態"] == "OK") & (ok["營收 CAGR"] < GROWTH_CAGR) & (ok["超越 0050"] > 0)
           & ~ok["代號"].isin(used)].sort_values("5 年含息報酬", ascending=False)
    ex_b = b.iloc[0] if len(b) else None
    if ex_b is not None:
        used.add(ex_b["代號"])

    c = table[((table["報酬狀態"] != "OK") | (table["CAGR 狀態"] != "OK")) & ~table["代號"].isin(used)].copy()
    c["_delisted"] = c["報酬狀態"].eq("下市暫估")
    c = c.sort_values(["_delisted", "缺值數", "特徵年"], ascending=[False, True, False])
    ex_c = c.iloc[0].drop("_delisted") if len(c) else None
    return ex_a, ex_b, ex_c


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    tbl = build_table()
    print(tbl[["代號", "名稱", "特徵年", "缺值數", "5 年含息報酬", "0050 同期含息報酬", "報酬狀態",
               "營收 CAGR", "CAGR 狀態", "判定"]].to_string(index=False))
