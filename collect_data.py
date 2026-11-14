"""
collect_data.py －全市場資料收集

收集臺灣證券交易所上市普通股（包含 2012 年後下市的公司）的 11 種 FinMind 資料：
（研究範圍只有上市；上櫃與興櫃不納入，2026-11-10 決定）
損益表、資產負債表、現金流量表、月營收、每日股價、本益比/淨值比/殖利率、除權息結果、減資恢復買賣參考價、股票分割參考價、
外資持股（含每日發行股數）、三大法人買賣超。
另外抓 0050 的每日股價、除權息結果與分割參考價，作為實驗 E2 的買進持有基準（0050 是 ETF，不在普通股清單內）。

設計要求：
  - 自動等待：API 次數用完（402）時每 5 分鐘試一次，額度恢復後繼續
  - 可接續：每個（資料集, 股票）抓完立刻存成一個小檔，中斷後重跑只抓還沒抓過的
  - 省次數：verify_data.py 已抓過的 cache/ 檔案直接轉存，不再呼叫 API
  - 輸出 Parquet：全部抓完後，每個資料集合併成一個檔案

額度：2026-11-09 實測未註冊每小時約 2,200 次（以過去一小時滾動計算），比官方文件寫的 300 次多。
全市場約 1,320 檔 × 11 種 ≈ 14,500 次，約需 6～7 小時（可分多次跑；之後新增資料集只會抓新的部分）。
外資持股與法人買賣超是延伸用資料（DESIGN.md 2.3），核心特徵目前不使用。

執行：
  python collect_data.py              # 開始或接續收集，全部完成後自動合併
  python collect_data.py --limit 5    # 只跑前 5 檔，先試跑
  python collect_data.py --merge      # 不抓資料，只把已抓的小檔合併成 Parquet
  python collect_data.py --status     # 不抓資料，只看目前進度（收集中也可以另開視窗看）
  python collect_data.py --refresh-twse  # 重新下載證交所的上市與終止上市清單，再開始收集

研究範圍（2026-11-12 起改用證交所官網的兩份公開清單，見 DESIGN.md 5.1、5.2）：
  - 現在上市的普通股：證交所 ISIN 清單（含上市日期；上櫃轉上市記的是轉上市那天）
  - 2012 年後終止上市的普通股：證交所「終止上市公司」名單（含下市日期）
  FinMind 的 TaiwanStockDelisting 是「下市櫃」表，2012 年後那 93 家經比對全部不是終止上市的公司；
  TaiwanStockInfo 的紀錄只到 2020 年，無法判斷當年是否上市。兩者都不再用來決定範圍。

輸入：證交所兩份公開清單（data/raw/twse_listed.csv、twse_delisted.csv；沒有才下載，--refresh-twse 重抓）
輸出：data/raw/parts/{dataset}/{stock_id}.parquet（逐檔小檔，用於接續）
      data/raw/{dataset}.parquet（合併後的正式輸出，供 build_dataset.py 使用）
      data/raw/universe.csv（研究範圍股票清單，含上市日、下市日）
      data/raw/progress.txt、progress.csv、collect.log（進度摘要、每檔明細、完整紀錄）
"""

import argparse
import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import verify_data as vd   # 沿用 .env 讀取、API 網址、Token、普通股判斷與股票清單快取

# ───────────── 設定 ─────────────
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"
PART_DIR = RAW_DIR / "parts"
CACHE_DIR = BASE / "cache"
START_DATE = vd.START_DATE            # 與 verify_data.py 相同（2005-01-01），cache/ 才能直接沿用
DELISTED_SINCE = "2012-01-01"
BENCHMARK_ID = "0050"
SLEEP_SECONDS = 0.6                   # 每次請求間隔，避免太密集被擋
CALLS_PER_HOUR = 2200                # 預估剩餘時間用；2026-11-09 未註冊實測每小時約 2,200 次就達上限
QUOTA_RECHECK_MINUTES = 5             # 額度用完時，每隔幾分鐘試一次（FinMind 額度是滾動一小時）

DATASETS = [
    "TaiwanStockFinancialStatements",
    "TaiwanStockBalanceSheet",
    "TaiwanStockCashFlowsStatement",
    "TaiwanStockMonthRevenue",
    "TaiwanStockPrice",
    "TaiwanStockPER",
    "TaiwanStockDividendResult",
    "TaiwanStockCapitalReductionReferencePrice",   # 減資恢復買賣參考價（2026-11-10 加入，修正含息報酬）
    "TaiwanStockSplitPrice",                       # 股票分割／面額變更參考價（同上）
    "TaiwanStockShareholding",                     # 外資持股與每日發行股數（2026-11-12 加入，2005 起）
    "TaiwanStockInstitutionalInvestorsBuySell",    # 三大法人買賣超（2026-11-12 加入，2012-05 起）
]
BENCHMARK_DATASETS = ["TaiwanStockPrice", "TaiwanStockDividendResult", "TaiwanStockSplitPrice"]  # 0050 於 2025 年分割


DS_SHORT = {   # 進度顯示用的簡稱
    "TaiwanStockFinancialStatements": "損益",
    "TaiwanStockBalanceSheet": "資產負債",
    "TaiwanStockCashFlowsStatement": "現金流",
    "TaiwanStockMonthRevenue": "月營收",
    "TaiwanStockPrice": "股價",
    "TaiwanStockPER": "本益比",
    "TaiwanStockDividendResult": "除權息",
    "TaiwanStockCapitalReductionReferencePrice": "減資",
    "TaiwanStockSplitPrice": "分割",
    "TaiwanStockShareholding": "外資持股",
    "TaiwanStockInstitutionalInvestorsBuySell": "法人買賣",
}
# 證交所官網的公開清單（只用來判定研究範圍與上市、下市日期）
TWSE_LISTED_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
TWSE_DELISTED_URL = "https://www.twse.com.tw/rwd/zh/company/suspendListingCsvAndHtml?type=html&lang=zh"
TWSE_LISTED_FILE = RAW_DIR / "twse_listed.csv"
TWSE_DELISTED_FILE = RAW_DIR / "twse_delisted.csv"

LOG_FILE = RAW_DIR / "collect.log"
PROGRESS_FILE = RAW_DIR / "progress.csv"    # 每檔股票 × 資料集的結果，可用 Excel 開
SUMMARY_FILE = RAW_DIR / "progress.txt"     # 整體進度摘要，隨時打開看

# Windows 終端機若不是 UTF-8，遇到無法顯示的字改成「?」而不是讓程式當掉
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ───────────── 股票清單 ─────────────
def _download(url):
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def fetch_twse_lists(refresh=False):
    """下載證交所兩份公開清單並存成 CSV；檔案已存在且不 refresh 就直接讀。

    twse_listed.csv  ：現在上市的有價證券（ISIN 清單「股票」類），stock_id、stock_name、listed_date、market、industry、cfi
    twse_delisted.csv：終止上市公司，stock_id、stock_name、delisted_date（民國年轉西元）
    """
    if refresh or not TWSE_LISTED_FILE.exists():
        html = _download(TWSE_LISTED_URL).decode("big5hkscs", errors="replace")
        t = pd.read_html(io.StringIO(html), header=0)[0]
        t.columns = ["code_name", "isin", "listed_date", "market", "industry", "cfi", "note"]
        t = t[t["cfi"].astype(str).str.startswith("E")]          # 只留股票類（ES＝普通股，ED＝存託憑證等）
        parts = t["code_name"].str.split(r"\s+", n=1, expand=True)
        t = t.assign(stock_id=parts[0], stock_name=parts[1],
                     listed_date=pd.to_datetime(t["listed_date"], format="%Y/%m/%d", errors="coerce"))
        t = t[["stock_id", "stock_name", "listed_date", "market", "industry", "cfi"]]
        t.to_csv(TWSE_LISTED_FILE, index=False, encoding="utf-8-sig")
        log(f"下載證交所上市清單：{len(t):,} 筆 → {TWSE_LISTED_FILE.relative_to(BASE)}")
    if refresh or not TWSE_DELISTED_FILE.exists():
        html = _download(TWSE_DELISTED_URL).decode("utf-8")
        d = pd.read_html(io.StringIO(html))[0]
        d.columns = ["date", "stock_name", "stock_id"]
        d["stock_id"] = d["stock_id"].astype(str).str.zfill(4)
        ymd = d["date"].str.extract(r"(\d+)年(\d+)月(\d+)日").astype(int)
        d["delisted_date"] = pd.to_datetime(dict(year=ymd[0] + 1911, month=ymd[1], day=ymd[2]))
        d = d[["stock_id", "stock_name", "delisted_date"]]
        d.to_csv(TWSE_DELISTED_FILE, index=False, encoding="utf-8-sig")
        log(f"下載證交所終止上市名單：{len(d):,} 筆 → {TWSE_DELISTED_FILE.relative_to(BASE)}")
    listed = pd.read_csv(TWSE_LISTED_FILE, dtype={"stock_id": str}, parse_dates=["listed_date"], encoding="utf-8-sig")
    delisted = pd.read_csv(TWSE_DELISTED_FILE, dtype={"stock_id": str}, parse_dates=["delisted_date"], encoding="utf-8-sig")
    return listed, delisted


def is_depositary_receipt(stock_id, stock_name):
    """存託憑證：91 開頭或名稱含 DR（99xx 等 9 開頭的普通股不算）。"""
    return stock_id.startswith("91") or "DR" in str(stock_name).upper()


def get_universe(refresh_twse=False):
    """回傳研究範圍的股票清單（DataFrame：stock_id、stock_name、status、listed_date、delisted_date）：
    證交所現在上市的普通股＋2012 年後終止上市的普通股（排除存託憑證）。

    已下市公司的上市日期官方清單沒有，留空，由 build_dataset.py 以股價與興櫃判定推估。
    """
    listed, delisted = fetch_twse_lists(refresh_twse)
    cur = listed[(listed["cfi"].str.startswith("ES")) & listed["stock_id"].str.fullmatch(vd.COMMON_STOCK_ID)]
    gone = delisted[(delisted["delisted_date"] >= DELISTED_SINCE)
                    & delisted["stock_id"].str.fullmatch(vd.COMMON_STOCK_ID)
                    & ~delisted.apply(lambda r: is_depositary_receipt(r["stock_id"], r["stock_name"]), axis=1)
                    & ~delisted["stock_id"].isin(cur["stock_id"])]
    gone = gone.sort_values("delisted_date").drop_duplicates("stock_id", keep="last")
    uni = pd.concat([
        cur[["stock_id", "stock_name", "listed_date"]].assign(status="上市中", delisted_date=pd.NaT),
        gone[["stock_id", "stock_name", "delisted_date"]].assign(status="已下市", listed_date=pd.NaT),
    ])[["stock_id", "stock_name", "status", "listed_date", "delisted_date"]]
    return uni.sort_values("stock_id").reset_index(drop=True)


# ───────────── 抓資料（自動等待額度） ─────────────
def wait_for_quota(msg, first):
    """API 次數用完：等 QUOTA_RECHECK_MINUTES 分鐘再試。

    實測 FinMind 的額度是「過去一小時內」滾動計算（不是整點歸零），一小時前的呼叫過期後就有新額度，
    所以每隔幾分鐘試一次，比等到下一個整點快。每次試只花 1 次請求。
    """
    resume = datetime.now() + timedelta(minutes=QUOTA_RECHECK_MINUTES)
    if first:
        log(f"⏸ API 次數已用完（{msg}），每 {QUOTA_RECHECK_MINUTES} 分鐘試一次，額度恢復就自動繼續。"
            "可以按 Ctrl+C 中斷，之後重跑會接著抓。")
    write_summary(note=f"等待 API 額度恢復，{resume:%H:%M} 再試")
    time.sleep(QUOTA_RECHECK_MINUTES * 60)


def fetch_with_quota(dataset, data_id, start_date):
    """抓單一（資料集, 股票）；遇到 API 次數用完時自動等待後重試。

    回傳 DataFrame（可能是空表＝FinMind 沒有這檔的資料）；連線一直失敗或 API 回傳其他錯誤時回傳 None，
    這時不存檔，下次重跑會再試。
    """
    params = {"dataset": dataset, "data_id": data_id, "start_date": start_date}
    headers = {"Authorization": f"Bearer {vd.TOKEN}"} if vd.TOKEN else {}
    quota_waits, net_fails = 0, 0
    while True:
        try:
            resp = requests.get(vd.API_URL, params=params, headers=headers, timeout=60)
            body = resp.json()
        except (requests.RequestException, ValueError) as e:
            net_fails += 1
            if net_fails >= 5:
                log(f"  放棄 {dataset} {data_id}：連線連續失敗（{e}）")
                return None
            wait = 10 * net_fails
            log(f"  連線失敗（第 {net_fails} 次）：{e}，{wait} 秒後重試")
            time.sleep(wait)
            continue

        status = body.get("status", resp.status_code)
        if status == 402 or resp.status_code == 402:
            wait_for_quota(body.get("msg", ""), first=quota_waits == 0)
            quota_waits += 1
            continue
        if quota_waits:
            log(f"▶ 額度恢復，繼續抓（等了約 {quota_waits * QUOTA_RECHECK_MINUTES} 分鐘）")
        if status != 200:
            log(f"  API 回傳錯誤 {status}：{body.get('msg', '')}（{dataset} {data_id}），下次重跑再試")
            return None

        time.sleep(SLEEP_SECONDS)
        df = pd.DataFrame(body.get("data", []))
        if "stock_id" in df.columns:
            df["stock_id"] = df["stock_id"].astype(str)
        return df


# ───────────── 進度與存檔 ─────────────
def part_path(dataset, stock_id):
    return PART_DIR / dataset / f"{stock_id}.parquet"


def load_progress():
    """已完成的（資料集, 股票）→ 列數：有小檔就算完成（空表也存成 0 列的小檔，代表 FinMind 確實沒資料）。"""
    import pyarrow.parquet as pq
    return {(p.parent.name, p.stem): pq.read_metadata(p).num_rows for p in PART_DIR.glob("*/*.parquet")}


# 進度狀態：results[(股票, 資料集)] ＝ 列數（0 ＝ 沒資料），失敗記為 -1；沒有鍵 ＝ 還沒抓
STATE = {"results": {}, "stocks": [], "names": {}, "status": {}, "api": 0, "began": None}


def mark(result_rows):
    if result_rows is None:
        return "?"
    return "✗" if result_rows < 0 else ("○" if result_rows == 0 else "✓")


def eta_text(remaining_api):
    hours = remaining_api / CALLS_PER_HOUR
    return f"約 {hours:.1f} 小時" if hours >= 1 else f"約 {hours * 60:.0f} 分鐘"


def write_summary(note="收集中"):
    """寫 progress.txt（整體摘要）與 progress.csv（每檔明細），隨時可以打開看。"""
    if not STATE["stocks"]:
        return
    res, stocks = STATE["results"], STATE["stocks"]
    tasks = [(s, ds) for s in stocks for ds in (BENCHMARK_DATASETS if s == BENCHMARK_ID else DATASETS)]
    done = sum(1 for t in tasks if res.get(t, -1) >= 0)
    failed = [t for t in tasks if res.get(t) == -1]
    stocks_done = sum(1 for s in stocks if all(res.get((s, ds), -1) >= 0
                      for ds in (BENCHMARK_DATASETS if s == BENCHMARK_ID else DATASETS)))
    lines = [
        f"更新時間：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"狀態：{note}",
        f"整體進度：{done:,} / {len(tasks):,} 個任務（{done / len(tasks):.1%}），股票 {stocks_done:,} / {len(stocks):,} 檔完成",
        f"本次執行：呼叫 API {STATE['api']:,} 次，開始於 {STATE['began']:%m-%d %H:%M}",
        f"預估剩餘：{eta_text(len(tasks) - done)}",
        "",
        "各資料集（✓ 有資料　○ FinMind 沒有這檔的資料　✗ 失敗，重跑會再試）：",
    ]
    for ds in DATASETS:
        vals = [res.get((s, ds)) for s in stocks if s != BENCHMARK_ID]
        ok = sum(1 for v in vals if v is not None and v > 0)
        empty = sum(1 for v in vals if v == 0)
        bad = sum(1 for v in vals if v == -1)
        left = sum(1 for v in vals if v is None)
        lines.append(f"  {DS_SHORT[ds]:<6}✓ {ok:>5,}　○ {empty:>4,}　✗ {bad:>3,}　未抓 {left:>5,}")
    if failed:
        lines += ["", f"失敗 {len(failed)} 個（前 10 個）：" + "、".join(f"{s} {DS_SHORT[ds]}" for s, ds in failed[:10])]
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for s in stocks:
        row = {"股票代號": s, "名稱": STATE["names"].get(s, ""), "狀態": STATE["status"].get(s, "")}
        for ds in DATASETS:
            v = res.get((s, ds))
            row[DS_SHORT[ds]] = "" if v is None else ("失敗" if v < 0 else v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(PROGRESS_FILE, index=False, encoding="utf-8-sig")


def print_status():
    if SUMMARY_FILE.exists():
        print(SUMMARY_FILE.read_text(encoding="utf-8"))
    else:
        print("還沒有進度紀錄，請先執行 python collect_data.py")


def save_part(dataset, stock_id, df):
    path = part_path(dataset, stock_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)          # 先寫暫存檔再改名，中途被中斷也不會留下壞檔


def from_cache(dataset, stock_id):
    """verify_data.py 抓過的 cache/ 檔案；沒有就回傳 None。"""
    path = CACHE_DIR / f"{dataset}_{stock_id}.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_parts():
    """把每個資料集的小檔合併成 data/raw/{dataset}.parquet（只合併研究範圍內的股票與 0050）。"""
    keep = set(pd.read_csv(RAW_DIR / "universe.csv", dtype={"stock_id": str})["stock_id"]) | {BENCHMARK_ID}
    for ds in sorted({*DATASETS, *BENCHMARK_DATASETS}):
        files = sorted(f for f in (PART_DIR / ds).glob("*.parquet") if f.stem in keep)
        frames = [df for df in (pd.read_parquet(f) for f in files) if len(df)]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["stock_id"] = df["stock_id"].astype(str)
        out = RAW_DIR / f"{ds}.parquet"
        df.to_parquet(out, index=False)
        log(f"合併 {ds}：{len(files)} 檔 → {len(df):,} 列（{out.relative_to(BASE)}）")


# ───────────── 主程式 ─────────────
def main():
    ap = argparse.ArgumentParser(description="收集全市場 FinMind 資料（可中斷、可接續）")
    ap.add_argument("--limit", type=int, help="只處理前 N 檔股票（試跑用）")
    ap.add_argument("--merge", action="store_true", help="不抓資料，只合併已抓的小檔")
    ap.add_argument("--status", action="store_true", help="不抓資料，只顯示目前進度")
    ap.add_argument("--refresh-twse", action="store_true", help="重新下載證交所上市與終止上市清單")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status()
        return
    if args.merge:
        merge_parts()
        return

    uni = get_universe(args.refresh_twse)
    uni.to_csv(RAW_DIR / "universe.csv", index=False, encoding="utf-8-sig")
    stocks = list(uni["stock_id"])[: args.limit] if args.limit else list(uni["stock_id"])
    STATE.update(
        stocks=[BENCHMARK_ID] + stocks,
        names=dict(zip(uni["stock_id"], uni["stock_name"])) | {BENCHMARK_ID: "元大台灣50（基準）"},
        status=dict(zip(uni["stock_id"], uni["status"])) | {BENCHMARK_ID: "ETF"},
        results={(s, ds): n for (ds, s), n in load_progress().items()},
        began=datetime.now(),
    )
    res = STATE["results"]
    total = len(STATE["stocks"])
    log("=" * 70)
    log(f"模式：{'已設定 Token（每小時約 600 次）' if vd.TOKEN else '未註冊（每小時約 300 次）'}")
    log(f"股票 {len(stocks)} 檔（上市中 {(uni['status'] == '上市中').sum()}、已下市 {(uni['status'] == '已下市').sum()}）＋ 0050 基準")
    log(f"符號：✓ 有資料　○ FinMind 沒有這檔的資料　✗ 失敗（重跑會再試）")
    log(f"進度摘要隨時看：{SUMMARY_FILE.relative_to(BASE)}，或另開視窗執行 python collect_data.py --status")
    write_summary()

    for i, sid in enumerate(STATE["stocks"], 1):
        datasets = BENCHMARK_DATASETS if sid == BENCHMARK_ID else DATASETS
        todo = [ds for ds in datasets if res.get((sid, ds), -1) < 0]
        if not todo:
            continue                       # 之前已抓完
        for ds in todo:
            df = from_cache(ds, sid)
            if df is None:
                df = fetch_with_quota(ds, sid, START_DATE)
                STATE["api"] += 1
            if df is None:
                res[(sid, ds)] = -1
                continue
            save_part(ds, sid, df)
            res[(sid, ds)] = len(df)

        marks = "　".join(f"{DS_SHORT[ds]}{mark(res.get((sid, ds)))}" for ds in datasets)
        left = sum(1 for s in STATE["stocks"] for ds in (BENCHMARK_DATASETS if s == BENCHMARK_ID else DATASETS)
                   if res.get((s, ds), -1) < 0)
        log(f"[{i:>4}/{total} {i / total:5.1%}] {sid} {STATE['names'].get(sid, '')}（{STATE['status'].get(sid, '')}）"
            f"　{marks}　｜剩 {eta_text(left)}")
        write_summary()

    failed = [k for k, v in res.items() if v == -1]
    if failed:
        log(f"有 {len(failed)} 個任務失敗，重跑本程式會再試這些（明細見 {PROGRESS_FILE.relative_to(BASE)}）")
        write_summary(note=f"本輪結束，有 {len(failed)} 個失敗，請重跑")
    else:
        log("全部任務完成 🎉")
        write_summary(note="全部完成")
        if not args.limit:
            merge_parts()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n已中斷。抓好的資料都存在 data/raw/parts/，重跑會從斷點接著抓。")
