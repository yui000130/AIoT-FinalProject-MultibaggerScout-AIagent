"""
agent.py －LLM Agent（個股分析與追問）

核心原則：數字一律由程式與模型計算，LLM 只負責規劃與解釋；
報告中的數字會自動與工具輸出核對，不一致就重寫（最多 3 次，失敗則改用模板報告）。
範圍：只做個股分析與追問；全市場篩選為延伸，不實作（DESIGN 2.3）。

流程見 DESIGN.md 第 8 節 Algorithm 1、工具規格見 3.1。實驗 E4（Agent vs 直接 LLM）在 evaluate.py 的 run_e4。

已完成（2026-11-24，不使用 LLM 的工具；每個數字都記一筆 evidence）：
  resolve_feature_year：請求日期 → 分析日 t（最近一個已過的 4 月底特徵日）
  get_features ：10 項特徵＋產業、規則輸入，附計算口徑、來源與缺值原因
  score_model  ：模型分數、當年全市場名次與潛力等級、SHAP、訓練期基準率（DESIGN 5.6、第 6 節）
  find_cases   ：kNN 相似案例，只用特徵年 ≤ t − 5 的已揭曉樣本（DESIGN 第 6 節、5.5 第 6、10 條）
  check_rules  ：2 條領域規則，門檻只用訓練特徵年（rules.py、DESIGN 第 7 節）
LLM 部分（2026-11-24，模型 gemini-3.5-flash-lite、temperature 0，DESIGN 第 6 節 LLM 設定）：
  plan          ：LLM 只判斷意圖（analyze／follow_up／clarify）與白名單內的工具；代號、日期由程式解析
  run_analysis  ：Algorithm 1 第 05～16 行；工具依固定順序執行，最多 12 次工具呼叫
  verify_report ：核對代號、分析日、潛力等級、必要欄位、每個數字（須引用證據且在顯示位數下相等）、日期與年份
  template_report：LLM 三輪未通過或無法使用時由程式產生；t < 2018 時輸出不評分的限制版報告
  follow_up     ：只用上一份報告的證據回答，同樣核對，最多 3 輪
  LLM 回應快取在 cache/llm/；金鑰只從環境變數或 .env 讀取，不印出、不寫入

輸入：股票代號、請求日期；data/dataset.parquet、data/live_features.parquet、models/（train.py）
輸出：各工具回傳 dict；EvidenceLog 收集 evidence_id、stock_id、as_of、field、value、unit、source
執行方式：由 app.py 呼叫（analyze(查詢, request_date, session)）；
          自我檢查 python agent.py（2018–2020 分數與 train.py 一致、kNN 時間限制、模板報告與核對器；不呼叫 LLM）
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import requests

import demo_pipeline as dp
import rules
import train

DATASET = train.DATASET
LIVE_DATASET = train.LIVE_DATASET
FIRST_YEAR, LAST_YEAR = 2013, max(train.LIVE_YEARS)
HIGH_PCT, MID_PCT = 0.05, 0.20        # 潛力等級切點：當年全市場前 5% 高、5%～20% 中（DESIGN 第 6 節）
K_CASES = 3                           # 成功、失敗案例各取 3 筆（DESIGN 第 6 節）
RANK_FILL = 0.5                       # kNN 缺值填排名中位數
MAX_TOOL_CALLS = 12
MAX_REWRITE = 3

LLM_MODEL = "gemini-3.5-flash-lite"   # 固定版本（DESIGN 第 6 節 LLM 設定），temperature = 0
PROMPT_VERSION = "2026-11-24 v3"   # v3：key_figures、規則 triggered、案例距離列為證據
LLM_CACHE = dp.BASE / "cache" / "llm"
LLM_MAX_WAITS = 10                    # 429 最多等待重試次數
LLM_BACKOFF = (10, 30, 60)            # 5xx／連線錯誤的等待秒數
REQUIRED_FIELDS = ("summary", "rules", "uncertainty", "key_figures")
KEY_FIELDS = list(train.NUMERIC_FEATURES) + [rules.ACCRUAL]     # key_figures 必填的 11 項（E4 答案表同欄位）
STATE = {True: "觸發", False: "未觸發", None: "不適用"}
MISSING_WORDS = (None, "", "缺值", "不適用", "null", "無")

NUMERIC = list(train.NUMERIC_FEATURES)          # 10 項數值特徵（中文欄名）
ACCRUAL = rules.ACCRUAL
UNITS = {"市值（億元）": "億元", "本益比": "倍", "營業現金流/淨利": "倍"}   # 其餘為比率（小數）


# ───────────── 證據紀錄 ─────────────
class EvidenceLog:
    """收集工具輸出的每個數字，供報告引用與核對（DESIGN 3.1）。"""

    def __init__(self):
        self.items = []

    def add(self, stock_id, as_of, field, value, unit, source):
        eid = f"E{len(self.items) + 1:03d}"
        v = None if value is None or (isinstance(value, float) and math.isnan(value)) else value
        self.items.append({"evidence_id": eid, "stock_id": stock_id, "as_of": str(as_of),
                           "field": field, "value": v, "unit": unit, "source": source})
        return eid

    def frame(self):
        return pd.DataFrame(self.items)


def unit_of(field):
    return UNITS.get(field, "比率")


# ───────────── 資料 ─────────────
@lru_cache(maxsize=None)
def labeled():
    """2013–2020 正式資料集（含標記）。"""
    if not DATASET.exists():
        sys.exit(f"找不到 {DATASET}，請先執行 python build_dataset.py")
    return pd.read_parquet(DATASET)


@lru_cache(maxsize=None)
def all_features():
    """2013–2026 每年評分樣本的特徵（2013–2020 來自正式資料集，2021–2026 來自 live_features）。"""
    if not LIVE_DATASET.exists():
        sys.exit(f"找不到 {LIVE_DATASET}，請先執行 python build_dataset.py")
    live = pd.read_parquet(LIVE_DATASET)
    cols = list(live.columns)
    df = pd.concat([labeled()[cols], live], ignore_index=True)
    assert not df.duplicated(["代號", "特徵年"]).any(), "同一公司同一特徵年重複"
    return df


@lru_cache(maxsize=None)
def year_ranks():
    """各特徵年內的百分位排名（同 train.rank_by_year；只用同年特徵，不用標記）。"""
    df = all_features()
    return train.rank_by_year(df).set_axis(NUMERIC, axis=1).assign(代號=df["代號"], 特徵年=df["特徵年"])


def resolve_feature_year(request_date):
    """請求日期 → 分析日 t：最近一個已過（含當天）的 4 月底特徵日（Algorithm 1 第 02 行）。

    回傳 (特徵年, 特徵日)；早於 2013 年特徵日或資料範圍外回傳 (None, 原因)。
    """
    d = pd.Timestamp(request_date)
    if d.year > LAST_YEAR:
        year = LAST_YEAR
    elif d.year < FIRST_YEAR:
        year = d.year
    else:
        year = d.year if d >= dp.feature_date(d.year) else d.year - 1
    if year < FIRST_YEAR:
        return None, f"最早的分析日是 {dp.feature_date(FIRST_YEAR):%Y-%m-%d}"
    return year, dp.feature_date(year)


def find_row(stock_id, year):
    df = all_features()
    r = df[(df["代號"] == stock_id) & (df["特徵年"] == year)]
    return r.iloc[0] if len(r) else None


# ───────────── 工具 ─────────────
def get_features(stock_id, year, log):
    """10 項特徵＋產業與規則輸入；不在當年評分樣本中（未上市滿一年、興櫃、流動性最低 10%、已下市）時回傳 ok=False。"""
    row = find_row(stock_id, year)
    t = dp.feature_date(year)
    if row is None:
        return {"ok": False, "stock_id": stock_id, "year": year, "as_of": t.date(),
                "reason": f"{stock_id} 不在 {year} 年的評分樣本中（未上市滿一年、仍在興櫃、流動性最低 10% 或已下市）"}
    detail = dp.compute_features(stock_id, year)       # 計算口徑與來源；值必須與資料集一致
    feats = {}
    for k in NUMERIC:
        v = None if pd.isna(row[k]) else float(row[k])
        d = detail[k]
        assert (v is None) == (d["value"] is None or pd.isna(d["value"])), f"{stock_id} {year} {k} 缺值狀態不一致"
        assert v is None or np.isclose(v, d["value"], rtol=1e-9), f"{stock_id} {year} {k} 與資料集不一致"
        src = f"{d['source']}；{d['formula']}"
        feats[k] = {"value": v, "unit": unit_of(k), "source": src, "missing": d["missing"],
                    "evidence_id": log.add(stock_id, t.date(), k, v, unit_of(k), src)}
    acc = None if pd.isna(row[ACCRUAL]) else float(row[ACCRUAL])
    acc_src = f"前一年度（{year - 1}）合併淨利 − 營業現金流，÷ 平均總資產（Sloan 1996 口徑）"
    feats[ACCRUAL] = {"value": acc, "unit": "比率", "source": acc_src, "missing": row["應計說明"] or "",
                      "evidence_id": log.add(stock_id, t.date(), ACCRUAL, acc, "比率", acc_src)}
    return {"ok": True, "stock_id": stock_id, "name": row["名稱"], "industry": row["產業"], "year": year,
            "as_of": t.date(), "features": feats, "missing_count": int(row["缺值數"])}


def model_info(year):
    """分析日 t 使用的模型（DESIGN 5.6）：t < 2018 不評分；2018–2020 用第 1～3 輪；2021–2026 用上線版。"""
    years = train.train_years_for(year)
    if not years:
        return None
    rounds = {f["test"]: n for n, f in enumerate(train.FOLDS, start=1)}
    if year in rounds:
        path, tested = train.MODEL_DIR / f"lgbm_round{rounds[year]}.joblib", "有 walk-forward 測試結果"
    else:
        path, tested = train.MODEL_DIR / f"lgbm_live{year}.joblib", "此模型尚無測試結果"
    tr = labeled()[labeled()["特徵年"].isin(years) & labeled()[train.TARGET].notna()]
    assert (pd.to_datetime(tr["標記揭曉日"]) <= dp.feature_date(year)).all(), "訓練標記在分析日之後才揭曉"
    return {"path": path, "train_years": years, "tested": tested, "categories": train.industry_categories(tr),
            "base_rate": float(tr[train.TARGET].astype(int).mean()), "train_n": len(tr)}


@lru_cache(maxsize=None)
def score_year(year):
    """同一模型對 t 當年全部評分樣本打分數，回傳 (分數表, SHAP 表, 模型資訊)；不評分時回傳 None。"""
    info = model_info(year)
    if info is None:
        return None
    model = joblib.load(info["path"])
    rows = all_features()[all_features()["特徵年"] == year]
    X = train.to_model_frame(rows, info["categories"])
    scores = rows[["代號", "名稱"]].copy()
    scores["score"] = model.predict_proba(X)[:, 1]
    order = scores.sort_values(["score", "代號"], ascending=[False, True])
    scores["名次"] = pd.Series(range(1, len(order) + 1), index=order.index)
    contrib = model.predict(X, pred_contrib=True)[:, :-1]    # 同 evaluate.py：最後一欄是基準值
    shap = pd.DataFrame(contrib, columns=train.FEATURE_COLS, index=X.index)
    return scores, shap, info


def potential_level(rank, n):
    """當年全市場名次 → 潛力等級（前 5% 高、5%～20% 中、其餘低）。"""
    pct = rank / n
    return "高" if pct <= HIGH_PCT else ("中" if pct <= MID_PCT else "低")


def score_model(stock_id, year, log):
    """模型分數、名次、潛力等級、SHAP 與基準率；t < 2018 或不在樣本中時 ok=False。"""
    t = dp.feature_date(year)
    res = score_year(year)
    if res is None:
        return {"ok": False, "reason": f"{year} 年沒有已揭曉的訓練標記（最早可評分的分析日為 2018 年 4 月底），不評分"}
    scores, shap, info = res
    hit = scores[scores["代號"] == stock_id]
    if hit.empty:
        return {"ok": False, "reason": f"{stock_id} 不在 {year} 年的評分樣本中"}
    i = hit.index[0]
    n, rank, score = len(scores), int(hit.at[i, "名次"]), float(hit.at[i, "score"])
    level = potential_level(rank, n)
    names = {v: k for k, v in {**train.NUMERIC_FEATURES, **train.CATEGORY_FEATURE}.items()}
    sh = shap.loc[i].rename(index=names).sort_values(key=lambda s: -s.abs())
    src_model = f"LightGBM（訓練特徵年 {info['train_years'][0]}–{info['train_years'][-1]}；{info['tested']}）"
    out = {"ok": True, "as_of": t.date(), "tested": info["tested"], "train_years": info["train_years"],
           "score": score, "rank": rank, "n": n, "level": level, "base_rate": info["base_rate"],
           "shap": [{"feature": f, "value": float(v),
                     "evidence_id": log.add(stock_id, t.date(), f"SHAP：{f}", float(v), "log-odds 貢獻",
                                            f"{src_model} pred_contrib")} for f, v in sh.items()]}
    out["evidence_id"] = {
        "score": log.add(stock_id, t.date(), "模型分數", score, "模型分數（未校準，不是機率）", src_model),
        "rank": log.add(stock_id, t.date(), "當年名次", rank, "名", f"{year} 年評分樣本 {n} 家依模型分數排序"),
        "n": log.add(stock_id, t.date(), "當年評分家數", n, "家", f"{year} 年評分樣本（上市滿一年、流動性過濾後）"),
        "level": log.add(stock_id, t.date(), "潛力等級", level, "等級", "當年名次前 5% 高、5%～20% 中、其餘低"),
        "base_rate": log.add(stock_id, t.date(), "基準率", info["base_rate"], "比率",
                             f"訓練特徵年主要標記正例率（{info['train_n']} 筆可判定）"),
    }
    return out


def find_cases(stock_id, year, log, k=K_CASES):
    """kNN：特徵年 ≤ t − 5、y_main 可判定、排除同公司；排名空間歐氏距離，成功與失敗各取 k 筆。"""
    t = dp.feature_date(year)
    ranks = year_ranks()
    me = ranks[(ranks["代號"] == stock_id) & (ranks["特徵年"] == year)]
    if me.empty:
        return {"ok": False, "reason": f"{stock_id} 不在 {year} 年的評分樣本中", "success": [], "failure": []}
    lab = labeled()
    pool = lab[(lab["特徵年"] <= year - train.HORIZON_YEARS) & lab[train.TARGET].notna() & (lab["代號"] != stock_id)]
    if pool.empty:
        return {"ok": True, "success": [], "failure": [], "note": "此時點尚無已揭曉的歷史案例"}
    assert (pd.to_datetime(pool["標記揭曉日"]) <= t).all(), "案例標記在分析日之後才揭曉"
    me_vec = me[NUMERIC].iloc[0]
    my_filled = int(me_vec.isna().sum())
    key = ranks.set_index(["代號", "特徵年"])[NUMERIC]
    pv = key.loc[list(zip(pool["代號"], pool["特徵年"]))]
    dist = np.sqrt(((pv.fillna(RANK_FILL).to_numpy() - me_vec.fillna(RANK_FILL).to_numpy()) ** 2).sum(axis=1))
    pool = pool.assign(距離=dist, 填補數=pv.isna().sum(axis=1).to_numpy())

    def pick(label):
        g = pool[pool[train.TARGET] == label].sort_values(["距離", "代號"]).head(k)
        cases = []
        for _, r in g.iterrows():
            ret = None if pd.isna(r["5 年含息報酬"]) else float(r["5 年含息報酬"])
            src = f"{r['代號']} {int(r['特徵年'])} 年特徵日 → 5 年後（標記揭曉日 {r['標記揭曉日']}）"
            cases.append({"stock_id": r["代號"], "name": r["名稱"], "industry": r["產業"], "year": int(r["特徵年"]),
                          "distance": float(r["距離"]), "filled": int(r["填補數"]), "return_5y": ret,
                          "return_status": r["報酬狀態"], "y_main": int(r[train.TARGET]),
                          "evidence_id": log.add(r["代號"], r["特徵日"], "5 年含息報酬", ret, "比率", src),
                          "distance_evidence_id": log.add(r["代號"], r["特徵日"], "相似度距離", float(r["距離"]), "距離",
                                                          "kNN：10 項特徵當年百分位排名的歐氏距離")})
        return cases

    return {"ok": True, "success": pick(1), "failure": pick(0), "my_filled": my_filled,
            "pool_years": sorted(int(y) for y in pool["特徵年"].unique()), "pool_n": len(pool)}


def check_rules(stock_id, year, log, features=None):
    """2 條領域規則；門檻只用訓練特徵年（2013 ～ min(t − 5, 2020)），沒有時改用當年全市場特徵分布。"""
    t = dp.feature_date(year)
    row = find_row(stock_id, year)
    if row is None:
        return {"ok": False, "reason": f"{stock_id} 不在 {year} 年的評分樣本中"}
    years = train.train_years_for(year)
    if years:
        th_rows, th_src = labeled()[labeled()["特徵年"].isin(years)], f"訓練特徵年 {years[0]}–{years[-1]} 的特徵分布"
    else:
        th_rows, th_src = all_features()[all_features()["特徵年"] == year], f"{year} 年當年全市場特徵分布（尚無已揭曉樣本）"
    th = rules.compute_thresholds(th_rows)
    results = rules.run_all_rules(row, th)
    for r in results:
        r["threshold_source"] = th_src
        r["evidence_id"] = {k: log.add(stock_id, t.date(), f"{r['rule']}：{k}", v,
                                       "倍" if "本益比" in k else "比率", f"rules.py；門檻來源：{th_src}")
                            for k, v in r["evidence"].items()}
    return {"ok": True, "rules": results, "threshold_source": th_src}


# ───────────── LLM 呼叫（Gemini REST，快取） ─────────────
class LLMError(RuntimeError):
    """LLM 無法使用（金鑰錯誤、每日額度用完、連續失敗）；呼叫端改用模板報告或停止。"""


def _api_key():
    """只從環境變數或 .env 讀取金鑰；不印出、不寫入任何檔案或 log。"""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env = dp.BASE / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise LLMError("找不到 GEMINI_API_KEY（請在 .env 設定）")
    return key


def _retry_seconds(err_json, default=60):
    for d in err_json.get("error", {}).get("details", []) or []:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(d.get("retryDelay", "")))
        if m:
            return math.ceil(float(m.group(1))) + 1
    return default


def llm_call(system, prompt, rep=0):
    """呼叫 Gemini 並回傳解析後的 JSON；同一 (模型, 系統提示, 提示, 重複編號) 只呼叫一次（cache/llm/）。

    429：依伺服器建議時間等待後重試（最多 LLM_MAX_WAITS 次）；每日額度用完、400／401／403 立即停止；
    5xx 或連線錯誤：等待 10、30、60 秒重試，仍失敗則停止。
    """
    payload = {"model": LLM_MODEL, "system": system, "prompt": prompt, "rep": rep}
    h = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    path = LLM_CACHE / f"{h}.json"
    if path.exists():
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
        return json.loads(text)
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    waits, errors = 0, 0
    while True:
        try:
            r = requests.post(url, json=body, headers={"x-goog-api-key": _api_key()}, timeout=120)
        except requests.RequestException as e:
            r, exc = None, e
        if r is not None and r.ok:
            parts = r.json()["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            LLM_CACHE.mkdir(parents=True, exist_ok=True)   # 先存快取再解析：不合法 JSON 也不重複花額度
            path.write_text(json.dumps({"model": LLM_MODEL, "rep": rep, "text": text,
                                        "created": pd.Timestamp.now().isoformat()}, ensure_ascii=False),
                            encoding="utf-8")
            return json.loads(text)              # 不是合法 JSON 時由呼叫端當成一次失敗的生成
        if r is not None and r.status_code == 429:
            info = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if "PerDay" in json.dumps(info) or waits >= LLM_MAX_WAITS:
                raise LLMError("Gemini 額度已用完（每日額度或等待次數上限），請改天再試")
            waits += 1
            sec = _retry_seconds(info)
            print(f"Gemini 429：等待 {sec} 秒後重試（第 {waits} 次）", flush=True)
            time.sleep(sec)
            continue
        if r is not None and r.status_code in (400, 401, 403):
            status = r.json().get("error", {}).get("status", "") if r.content else ""
            raise LLMError(f"Gemini 拒絕請求（HTTP {r.status_code} {status}），請檢查金鑰或模型名稱；已停止")
        errors += 1
        if errors > len(LLM_BACKOFF):
            why = f"HTTP {r.status_code}" if r is not None else type(exc).__name__
            raise LLMError(f"Gemini 連續失敗（{why}），已停止")
        time.sleep(LLM_BACKOFF[errors - 1])


# ───────────── 數字格式與報告核對 ─────────────
def round_half_up(x, d, scale=1):
    """一般的四捨五入（以十進位字串計算；Python round 與格式化會受浮點誤差與銀行家捨入影響，例如 34.55 → 34.5）。
    顯示、核對（_matches）與 E4 評分都用這個函式，口徑一致。"""
    q = (Decimal(repr(float(x))) * scale).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)   # 先轉十進位再乘倍率
    return float(q)


def fmt(value, unit, field=""):
    """程式產生的顯示字串（報告引用的數字以此為準；核對時也接受其他合理寫法）。"""
    if value is None:
        return "缺值"
    if isinstance(value, str):
        return value
    r = round_half_up
    if unit == "比率":
        return f"{r(value, 3):.3f}" if "帳面/市值" in field else f"{r(value, 1, 100):.1f}%"
    if unit == "億元":
        return f"{r(value, 1):,.1f} 億元"
    if unit == "倍":
        return f"{r(value, 1):.1f} 倍"
    if unit in ("名", "家"):
        return f"{int(value)} {unit}"
    if unit == "距離":
        return f"{r(value, 3):.3f}"
    if unit == "log-odds 貢獻":
        return f"{r(value, 2):+.2f}"
    return f"{r(value, 3):.3f}"


CITE = re.compile(r"\[(E\d{3}(?:\s*[,，、]\s*E\d{3})*)\]")   # [E001] 或 [E001, E002]
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CN_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?|(\d{4})/(\d{1,2})(?:/(\d{1,2}))?")
NUM = re.compile(r"(?<![A-Za-z0-9_.])([+\-−]?)(\d[\d,]*(?:\.\d+)?)\s*(%|％|個百分點|倍|兆元|億元|名|家|年|筆|個月|個)?")
ALLOWED_CONSTANTS = {3, 5, 10, 12, 20, 90}   # 規格常數（5 年觀察期、前 5%／20%、第 10／90 百分位、各 3 筆、12 個月）


def _matches(token, sign, unit, ev):
    """token（字串）是否等於證據值 ev 在顯示位數下的某種寫法。"""
    v = ev["value"]
    if v is None or isinstance(v, str):
        return False
    x = float(token.replace(",", ""))
    if sign in ("-", "−"):
        x = -x
    d = len(token.split(".")[1]) if "." in token else 0
    if unit in ("%", "％", "個百分點"):
        cands = [(v, 100)] if ev["unit"] == "比率" else [(v, 1)]
    elif unit == "兆元":
        cands = [(v, Decimal("0.0001"))] if ev["unit"] == "億元" else []
    else:
        cands = [(v, 1), (v, 100)] if ev["unit"] == "比率" else [(v, 1)]
    for c, sc in cands:
        if math.isclose(round_half_up(c, d, sc), x, abs_tol=1e-9):
            return True
        if not sign and math.isclose(round_half_up(abs(c), d, sc), x, abs_tol=1e-9):   # 方向用文字表達（例如「下降 7.7%」）
            return True
    return False


def check_text(text, cited, ctx):
    """核對一段文字：日期、年份、代號、數字都必須對應到證據；回傳錯誤清單。"""
    errs = []
    cited = set(cited) | {c for g in CITE.findall(text) for c in re.findall(r"E\d{3}", g)}
    unknown = [c for c in cited if c not in ctx["evidence"]]
    errs += [f"引用了不存在的證據 {c}" for c in unknown]
    body = CITE.sub(" ", text)
    for d in DATE.findall(body):
        if d not in ctx["dates"]:
            errs.append(f"日期 {d} 不在證據中")
    body = DATE.sub(" ", body)
    for m in CN_DATE.finditer(body):                 # 2019 年 4 月 30 日、2019/4/30、2019 年 4 月（底）
        y, mo, d = int(m.group(1) or m.group(4)), int(m.group(2) or m.group(5)), m.group(3) or m.group(6)
        ok = (f"{y}-{mo:02d}-{int(d):02d}" in ctx["dates"]) if d else any(x.startswith(f"{y}-{mo:02d}") for x in ctx["dates"])
        if not ok:
            errs.append(f"日期 {m.group(0).strip()} 不在證據中")
    body = CN_DATE.sub(" ", body)
    body = re.sub(r"(?<=\d)\s*[-–~～]\s*(?=\d)", " – ", body)        # 2013-2020 視為範圍，不是負號
    evs = [ctx["evidence"][c] for c in cited if c in ctx["evidence"]]
    for sign, token, unit in NUM.findall(body):
        plain = token.replace(",", "")
        if "." not in plain and not sign:
            n = int(plain)
            if plain in ctx["stock_ids"] or (1990 <= n <= 2100 and n in ctx["years"]) or n in ALLOWED_CONSTANTS:
                continue
            if 1990 <= n <= 2100 and unit in ("", "年"):
                errs.append(f"年份 {n} 不在證據中")
                continue
        if not evs:
            errs.append(f"數字 {sign}{token}{unit} 沒有引用證據")
        elif not any(_matches(token, sign, unit, e) for e in evs):
            errs.append(f"數字 {sign}{token}{unit} 與引用的證據 {sorted(cited)} 不符")
    return errs


def verify_report(report, ctx):
    """Algorithm 1 第 13 行：核對公司、期間、潛力等級、必要欄位與每個數字；回傳錯誤清單（空 ＝ 通過）。"""
    errs = []
    if not isinstance(report, dict):
        return ["報告不是 JSON 物件"]
    if str(report.get("stock_id")) != ctx["stock_id"]:
        errs.append(f"公司代號應為 {ctx['stock_id']}")
    if str(report.get("as_of")) != ctx["as_of"]:
        errs.append(f"分析日應為 {ctx['as_of']}")
    if ctx.get("level") and report.get("level") != ctx["level"]:
        errs.append(f"潛力等級應為「{ctx['level']}」")
    for k in REQUIRED_FIELDS:
        if not report.get(k):
            errs.append(f"缺少必要欄位 {k}")
    if not report.get("supporting") and not report.get("opposing"):
        errs.append("支持與反對證據都是空的")
    names = {r["rule"] for r in report.get("rules") or [] if isinstance(r, dict)}
    errs += [f"缺少規則「{n}」的說明" for n in ctx["rule_names"] if n not in names]
    for r in report.get("rules") or []:
        if isinstance(r, dict) and r.get("rule") in ctx["rule_state"] and r.get("triggered") != ctx["rule_state"][r["rule"]]:
            errs.append(f"規則「{r['rule']}」的 triggered 應為「{ctx['rule_state'][r['rule']]}」")
    kf = {k.get("field"): k for k in report.get("key_figures") or [] if isinstance(k, dict)}
    for f, eid in ctx["key_evidence"].items():
        if f not in kf:
            errs.append(f"key_figures 缺少「{f}」")
            continue
        val, cid = kf[f].get("value"), kf[f].get("evidence_id")
        truth = ctx["evidence"][eid]["value"]
        if truth is None:
            if val not in MISSING_WORDS:
                errs.append(f"key_figures「{f}」應為缺值")
        elif cid != eid:
            errs.append(f"key_figures「{f}」應引用 {eid}")
        else:
            errs += [f"key_figures「{f}」：{e}" for e in check_text(str(val), [cid], ctx)]
    if isinstance(report.get("summary"), str):
        errs += check_text(report["summary"], [], ctx)
    for k in ("supporting", "opposing", "cases", "rules"):
        for item in report.get(k) or []:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                errs.append(f"{k} 的項目格式錯誤")
                continue
            errs += [f"{k}：{e}" for e in check_text(item["text"], item.get("evidence_ids") or [], ctx)]
    for u in report.get("uncertainty") or []:
        errs += [f"uncertainty：{e}" for e in check_text(str(u), [], ctx)]
    return errs


# ───────────── 規則固定說明（knowledge/） ─────────────
def _section(md, heading):
    m = re.search(rf"^## {re.escape(heading)}[^\n]*\n(.*?)(?=^## |\Z)", md, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def explain_rules(rule_results):
    """讀 knowledge/ 中每條規則的固定說明：「這條規則在看什麼」給 LLM 參考，「注意事項」由程式原文附在報告後。

    文獻中的報酬數字不交給 LLM（不是本系統計算的證據，避免被當成本股的數字引用）。
    """
    out = {}
    for r in rule_results:
        md = (dp.BASE / r["doc"]).read_text(encoding="utf-8")
        out[r["rule"]] = {"doc": r["doc"], "what": _section(md, "這條規則在看什麼"),
                          "caveats": _section(md, "與原文的差異與注意事項"),
                          "citation": "Sloan (1996)" if "盈餘" in r["rule"] else "Lakonishok, Shleifer & Vishny (1994)"}
    return out


# ───────────── 證據包、模板報告與 LLM 報告 ─────────────
def evidence_bundle(stock_id, year, log, feats, score, cases, rules_out):
    """Algorithm 1 第 10 行：整理給 LLM 的證據包與核對用的 ctx。"""
    t = dp.feature_date(year)
    ev = {e["evidence_id"]: e for e in log.items}
    for e in ev.values():
        e["display"] = fmt(e["value"], e["unit"], e["field"])
    case_list = (cases.get("success") or []) + (cases.get("failure") or [])
    years = {year, year - 1, *(score.get("train_years") or [])} | {c["year"] for c in case_list} \
        | {c["year"] + train.HORIZON_YEARS for c in case_list}
    if score.get("ok"):
        years |= {score["train_years"][0], score["train_years"][-1]}
    dates = {str(e["as_of"]) for e in ev.values()} | {str(dp.horizon_end(c["year"]).date()) for c in case_list}
    ctx = {"stock_id": stock_id, "as_of": str(t.date()), "level": score.get("level") if score.get("ok") else None,
           "evidence": ev, "dates": dates, "years": years,
           "stock_ids": {stock_id} | {c["stock_id"] for c in case_list},
           "rule_names": [r["rule"] for r in rules_out.get("rules", [])],
           "rule_state": {r["rule"]: STATE[r["triggered"]] for r in rules_out.get("rules", [])},
           "key_evidence": {f: feats["features"][f]["evidence_id"] for f in KEY_FIELDS}}
    bundle = {
        "公司": {"代號": stock_id, "名稱": feats["name"], "產業": feats["industry"], "分析日": ctx["as_of"]},
        "模型": ({"潛力等級": score["level"], "模型說明": score["tested"],
                  "訓練特徵年": f"{score['train_years'][0]}–{score['train_years'][-1]}",
                  "證據": [ev[score["evidence_id"][k]] for k in ("score", "rank", "n", "level", "base_rate")],
                  "SHAP（前 5 項，正值推高分數）": [ev[s["evidence_id"]] for s in score["shap"][:5]]}
                 if score.get("ok") else {"說明": score.get("reason")}),
        "特徵": [ev[f["evidence_id"]] | {"缺值原因": f["missing"]} for f in feats["features"].values()],
        "相似案例": ({"說明": cases.get("note")} if not case_list else
                     [{"類別": "成功（主要標記 = 1）" if c["y_main"] else "失敗（主要標記 = 0）", "代號": c["stock_id"],
                       "名稱": c["name"], "特徵年": c["year"], "距離": ev[c["distance_evidence_id"]],
                       "5 年含息報酬": ev[c["evidence_id"]]} for c in case_list]),
        "領域規則": [{"規則": r["rule"], "triggered": STATE[r["triggered"]],
                      "程式判斷": r["reason"], "證據": [ev[i] for i in r["evidence_id"].values()]}
                     for r in rules_out.get("rules", [])],
    }
    return bundle, ctx


def _cite(ev):
    return f"{ev['display']} [{ev['evidence_id']}]"


def template_report(bundle, ctx, score, cases, rules_out):
    """Algorithm 1 第 16 行：LLM 三輪都沒通過核對或無法使用時，由程式依固定模板產生（數字全來自證據）。"""
    ev = ctx["evidence"]
    sid, name = ctx["stock_id"], bundle["公司"]["名稱"]
    rep = {"stock_id": sid, "as_of": ctx["as_of"], "level": ctx["level"]}
    if score.get("ok"):
        e = {k: ev[i] for k, i in score["evidence_id"].items()}
        rep["summary"] = (f"{name}（{sid}）在 {ctx['as_of']} 的模型分數為 {_cite(e['score'])}，"
                          f"當年 {_cite(e['n'])} 中排第 {_cite(e['rank'])}，潛力等級「{ctx['level']}」；"
                          f"訓練期基準率 {_cite(e['base_rate'])}。{score['tested']}。")
        pos = [s for s in score["shap"][:5] if s["value"] > 0]
        neg = [s for s in score["shap"][:5] if s["value"] < 0]
        rep["supporting"] = [{"text": f"{s['feature']} 推高模型分數（SHAP {_cite(ev[s['evidence_id']])}）",
                              "evidence_ids": [s["evidence_id"]]} for s in pos]
        rep["opposing"] = [{"text": f"{s['feature']} 拉低模型分數（SHAP {_cite(ev[s['evidence_id']])}）",
                            "evidence_ids": [s["evidence_id"]]} for s in neg]
    else:
        rep["summary"] = f"{name}（{sid}）在 {ctx['as_of']}：{score.get('reason')}。以下只列特徵與領域規則。"
        rep["supporting"], rep["opposing"] = [], []
    rep["cases"] = [{"text": f"{'成功' if c['y_main'] else '失敗'}案例：{c['name']}（{c['stock_id']}）{c['year']} 年，"
                             f"之後 5 年含息報酬 {_cite(ev[c['evidence_id']])}",
                     "evidence_ids": [c["evidence_id"]]}
                    for c in (cases.get("success") or []) + (cases.get("failure") or [])]
    rep["rules"] = []
    for r in rules_out.get("rules", []):
        ids = list(r["evidence_id"].values())
        state = {True: "觸發", False: "未觸發", None: "不適用"}[r["triggered"]]
        vals = "、".join(f"{k} {_cite(ev[i])}" for k, i in r["evidence_id"].items() if ev[i]["value"] is not None)
        rep["rules"].append({"rule": r["rule"], "triggered": state, "text": f"{r['rule']}：{state}。{vals}",
                             "evidence_ids": ids})
        if r["triggered"]:
            rep["opposing"].append({"text": f"領域規則「{r['rule']}」觸發", "evidence_ids": ids})
    if not rep["supporting"] and not rep["opposing"]:
        rep["opposing"] = [{"text": "沒有可用的模型證據", "evidence_ids": []}]
    rep["key_figures"] = [{"field": f, "value": ev[i]["display"] if ev[i]["value"] is not None else None,
                           "evidence_id": i} for f, i in ctx["key_evidence"].items()]
    rep["uncertainty"] = ["模型分數未經機率校準，只用來排序，不是上漲機率。",
                          "相似案例只用結果已揭曉的歷史樣本，不代表本股會有相同結果。"]
    if not cases.get("success") and not cases.get("failure"):
        rep["uncertainty"].append("此時點尚無已揭曉的歷史案例。")
    return rep


SYSTEM_REPORT = """你是台股研究助理，只能根據使用者提供的「證據包」撰寫結構化分析報告。規則：
1. 不可自行計算、估計或補充任何數字；所有數字必須逐字取自證據的 display 欄位，並在數字後標註來源，例如「35.4% [E004]」。
2. 不可提到證據包以外的公司資訊、新聞、股價走勢或未來表現；分析日之後發生的事一律不談。
3. 「模型分數」不是機率，不可寫成上漲機率；潛力等級照證據寫。
4. 只輸出 JSON，格式：
{"stock_id": "代號", "as_of": "YYYY-MM-DD", "level": "高/中/低（無模型時 null）",
 "summary": "2～3 句總結",
 "supporting": [{"text": "支持證據", "evidence_ids": ["E001"]}],
 "opposing": [{"text": "反對證據或注意事項", "evidence_ids": ["E002"]}],
 "cases": [{"text": "相似案例說明", "evidence_ids": ["E030"]}],
 "rules": [{"rule": "規則名稱", "triggered": "觸發/未觸發/不適用", "text": "規則結果說明", "evidence_ids": ["E020"]}],
 "key_figures": [{"field": "特徵名稱", "value": "照 display 填寫（缺值填 null）", "evidence_id": "E001"}],
 "uncertainty": ["不確定性說明"]}
5. 每一條領域規則都要在 rules 中出現一次（含未觸發與不適用）。
6. key_figures 必須列出證據包「特徵」中的每一項（共 11 項），value 照該證據的 display 填寫，evidence_id 填該證據編號。"""


def report_prompt(bundle, rule_docs, errors_so_far):
    p = {"證據包": bundle, "領域規則說明（固定文字）": {k: {"在看什麼": v["what"], "文獻": v["citation"]}
                                                    for k, v in rule_docs.items()}}
    text = json.dumps(p, ensure_ascii=False, default=str, indent=1)
    if errors_so_far:
        text += "\n\n上一版報告核對未通過，請修正以下問題後重寫：\n" + "\n".join(f"- {e}" for e in errors_so_far[:30])
    return text


# ───────────── Planner（意圖與工具選擇） ─────────────
TOOLS = {   # DESIGN 3.1 的白名單（verify_report 與 follow_up 由程式執行，不由 Planner 選）
    "get_features": "特徵、取得日期、來源與缺值",
    "score_model": "模型分數、名次、潛力等級、SHAP、基準率",
    "find_cases": "結果已揭曉的相似成功與失敗案例",
    "check_rules": "盈餘品質與估值兩條領域規則",
}
REQUIRED_TOOLS = list(TOOLS)   # 分析新股票時 Algorithm 1 一定執行的工具（依此順序）

SYSTEM_PLANNER = """你是台股個股分析系統的 Planner，只做兩件事：判斷意圖、從工具白名單中選工具。不回答問題、不產生任何數字。
意圖只有三種：analyze（分析一檔新的股票）、follow_up（針對上一份報告追問）、clarify（缺少股票或看不懂，需要請使用者補充）。
只輸出 JSON：{"intent": "analyze|follow_up|clarify", "tools": ["工具名稱"], "reason": "一句話"}"""


def parse_query(text):
    """程式解析代號（或名稱）與日期；不交給 LLM，避免代號、日期被改寫。回傳 (代號或 None, 日期或 None, 候選代號)。"""
    t = str(text)
    date = None
    m = re.search(r"(\d{4})[-/.年](\d{1,2})(?:[-/.月](\d{1,2}))?", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 28)
        try:
            date = pd.Timestamp(year=y, month=mo, day=d)
        except ValueError:
            date = None
        t = t.replace(m.group(0), " ")
    else:
        m = re.search(r"(\d{4})\s*年", t)
        if m:
            date, t = pd.Timestamp(year=int(m.group(1)), month=12, day=31), t.replace(m.group(0), " ")
    df = all_features()
    ids = set(df["代號"])
    found = [c for c in re.findall(r"(?<!\d)(\d{4})(?!\d)", t) if c in ids]
    if not found:
        names = df.drop_duplicates("代號", keep="last")[["代號", "名稱"]]
        hits = [(n, c) for c, n in zip(names["代號"], names["名稱"]) if isinstance(n, str) and len(n) >= 2 and n in t]
        hits.sort(key=lambda x: -len(x[0]))
        found = [hits[0][1]] if hits else []
    found = list(dict.fromkeys(found))
    return (found[0] if len(found) == 1 else None), date, found


def plan(query, session):
    """Algorithm 1 第 04 行：LLM 判斷意圖並在白名單內選工具；程式檢查白名單與上限，LLM 失敗時依規則判斷。"""
    state = {"上一份報告": ({"代號": session["stock_id"], "分析日": session["as_of"]} if session else None),
             "工具白名單": TOOLS}
    try:
        p = llm_call(SYSTEM_PLANNER, json.dumps({"使用者輸入": query, "目前狀態": state}, ensure_ascii=False))
        intent = p.get("intent") if p.get("intent") in ("analyze", "follow_up", "clarify") else None
        tools = [x for x in (p.get("tools") or []) if x in TOOLS]
        source = "LLM"
    except (LLMError, ValueError, KeyError, TypeError) as e:     # LLM 不可用時以規則判斷，並記錄
        intent, tools, source = None, [], f"規則（LLM 無法使用：{e}）"
    sid, _, cands = parse_query(query)
    if intent is None:
        intent = "analyze" if sid else ("follow_up" if session else "clarify")
    if intent == "follow_up" and not session:
        intent = "analyze" if sid else "clarify"
    if intent == "analyze" and not sid:
        intent = "follow_up" if session and not cands else "clarify"
    return {"intent": intent, "tools": tools[:MAX_TOOL_CALLS], "source": source}


# ───────────── 主流程 ─────────────
def run_analysis(stock_id, year, rep=0, use_llm=True):
    """Algorithm 1 第 05～16 行：執行工具 → 證據包 → LLM 報告 ⇄ 核對（最多 3 輪）→ 失敗改用模板報告。"""
    log, calls = EvidenceLog(), 0
    feats = get_features(stock_id, year, log); calls += 1
    if not feats["ok"]:
        return {"status": "clarify", "message": feats["reason"], "tool_calls": calls}
    score = score_model(stock_id, year, log); calls += 1
    cases = find_cases(stock_id, year, log); calls += 1
    rules_out = check_rules(stock_id, year, log); calls += 1
    assert calls <= MAX_TOOL_CALLS
    rule_docs = explain_rules(rules_out.get("rules", []))
    bundle, ctx = evidence_bundle(stock_id, year, log, feats, score, cases, rules_out)
    result = {"stock_id": stock_id, "as_of": ctx["as_of"], "name": feats["name"], "tool_calls": calls,
              "model": LLM_MODEL, "prompt_version": PROMPT_VERSION, "evidence": log.items, "bundle": bundle,
              "rule_docs": rule_docs, "score": score, "cases": cases, "rules": rules_out, "features": feats,
              "attempts": [], "ctx": ctx}
    if not score.get("ok"):                          # 第 06 行：不評分時輸出限制版報告（程式產生）
        result.update(status="limited", report=template_report(bundle, ctx, score, cases, rules_out))
        return result
    errors = []
    if use_llm:
        for attempt in range(1, MAX_REWRITE + 1):
            try:
                report = llm_call(SYSTEM_REPORT, report_prompt(bundle, rule_docs, errors), rep=rep)
            except ValueError:                       # 回應不是合法 JSON
                report, errors = None, ["回應不是合法 JSON"]
            except LLMError as e:
                result["llm_error"] = str(e)
                break
            else:
                errors = verify_report(report, ctx)
            result["attempts"].append({"attempt": attempt, "errors": errors})
            if not errors:
                result.update(status="ok", report=report)
                return result
    tpl = template_report(bundle, ctx, score, cases, rules_out)
    assert not verify_report(tpl, ctx), "模板報告未通過核對"
    result.update(status="template", report=tpl)
    return result


SYSTEM_FOLLOW = """你是台股研究助理，只能根據「證據包」與「上一份報告」回答使用者的追問。
不可自行計算或補充數字；數字逐字取自證據的 display 欄位並標註來源，例如「35.4% [E004]」。
證據不足時明說「目前的證據無法回答」，並說明缺什麼。不可談論分析日之後的事。
只輸出 JSON：{"answer": "回答", "evidence_ids": ["E001"], "enough_evidence": true}"""


def follow_up(question, session, rep=0):
    """追問：只用上一份報告的證據回答；同樣核對數字，最多 3 輪，失敗時回覆限制說明。"""
    ctx, errors, attempts = session["ctx"], [], []
    base = {"證據包": session["bundle"], "上一份報告": session["report"], "追問": question}
    for attempt in range(1, MAX_REWRITE + 1):
        prompt = json.dumps(base, ensure_ascii=False, default=str)
        if errors:
            prompt += "\n\n上一版回答核對未通過，請修正：\n" + "\n".join(f"- {e}" for e in errors[:30])
        try:
            ans = llm_call(SYSTEM_FOLLOW, prompt, rep=rep)
            errors = check_text(str(ans.get("answer", "")), ans.get("evidence_ids") or [], ctx) \
                if isinstance(ans, dict) and ans.get("answer") else ["沒有回答"]
        except ValueError:
            ans, errors = None, ["回應不是合法 JSON"]
        except LLMError as e:
            return {"status": "limited", "answer": f"目前無法使用 LLM（{e}），請參考上方報告與證據表。",
                    "attempts": attempts}
        attempts.append({"attempt": attempt, "errors": errors})
        if not errors:
            return {"status": "ok", "answer": ans["answer"], "evidence_ids": ans.get("evidence_ids") or [],
                    "attempts": attempts}
    return {"status": "limited", "answer": "這個問題無法用目前的證據可靠回答（回答三次都未通過數字核對）。",
            "attempts": attempts}


def analyze(query, request_date=None, session=None, rep=0):
    """Agent 主流程入口（Algorithm 1）。session：上一份分析結果（追問用）。"""
    p = plan(query, session)
    if p["intent"] == "clarify":
        _, _, cands = parse_query(query)
        msg = (f"找到多個股票代號 {cands}，請只輸入一檔。" if len(cands) > 1 else
               "請輸入要分析的股票代號或名稱（例如 2330 或 台積電），可加上日期（例如 2024-05-01）。")
        return {"status": "clarify", "message": msg, "plan": p}
    if p["intent"] == "follow_up":
        out = follow_up(query, session, rep=rep)
        out["plan"] = p
        return out
    sid, qdate, _ = parse_query(query)
    year, t = resolve_feature_year(qdate if qdate is not None else (request_date or pd.Timestamp.today()))
    if year is None:
        return {"status": "clarify", "message": t, "plan": p}
    out = run_analysis(sid, year, rep=rep)
    out["plan"] = p
    return out


# ───────────── 自我檢查 ─────────────
def self_check():
    """1. 2018–2020 的模型分數與 train.py 存下的測試年分數完全相同；2. kNN 案例都在 t − 5 年以前且已揭曉。"""
    for n, f in enumerate(train.FOLDS, start=1):
        scores, _, _ = score_year(f["test"])
        saved = pd.read_parquet(train.MODEL_DIR / f"scores_round{n}.parquet")
        m = scores.merge(saved[["代號", "score_lgbm"]], on="代號", how="outer", indicator=True)
        assert (m["_merge"] == "both").all(), f"第 {n} 輪評分樣本不一致"
        diff = float((m["score"] - m["score_lgbm"]).abs().max())
        assert diff == 0, f"第 {n} 輪分數不一致（最大差 {diff}）"
        print(f"第 {n} 輪（{f['test']}）：{len(m)} 家分數與 train.py 完全相同")
    for year in (2017, 2018, 2020, 2023, 2026):
        g = all_features()[all_features()["特徵年"] == year]
        for sid in g.sample(5, random_state=42)["代號"]:
            log = EvidenceLog()
            c = find_cases(sid, year, log)
            for case in c["success"] + c["failure"]:
                assert case["year"] <= year - 5 and case["stock_id"] != sid
            assert len(c["success"]) == (K_CASES if year >= 2018 else 0)
        print(f"{year}：kNN 抽 5 家，案例皆為特徵年 ≤ {year - 5}、不含同公司")
    # 3. 模板報告一定通過核對（run_analysis 內 assert）；核對器要抓到竄改
    r = run_analysis("2330", 2026, use_llm=False)
    rep, ctx = r["report"], r["ctx"]
    rank = r["score"]["rank"]
    bad = [dict(rep, summary=rep["summary"].replace(f"{rank} 名", f"{rank + 1} 名")),
           dict(rep, level="高" if rep["level"] != "高" else "低"),
           dict(rep, uncertainty=rep["uncertainty"] + ["預期 2028 年營收翻倍"]),
           dict(rep, supporting=rep["supporting"] + [{"text": "毛利率 12.3%", "evidence_ids": []}]),
           dict(rep, rules=rep["rules"][:1])]
    assert not verify_report(rep, ctx) and all(verify_report(b, ctx) for b in bad)
    print("模板報告通過核對；5 種竄改（名次、等級、未來年份、未引用數字、缺規則）都被抓到")


if __name__ == "__main__":
    self_check()
