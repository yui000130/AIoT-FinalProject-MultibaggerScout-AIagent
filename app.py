"""
app.py －Streamlit 網站（目前為資料管線 Demo）

目前內容（只讀本機資料，計算見 demo_pipeline.py，不呼叫 API、不接 LLM）：
  有 data/raw/（collect_data.py 抓的全市場資料）就用全市場，否則用 cache/ 的 30 檔抽樣。
  原始資料只在本機，不上傳；頁面標示資料來源 FinMind（FinMind 授權要求）。
  1. 我們要找什麼樣的股票：白話說明主要標記，自動挑出三種例子
  2. 時光機：選股票與特徵年，看當時可用的 10 項特徵與 5 年後的結果
  3. 總表：每檔每年一列，可排序
  4. 個股分析（骨架，analysis_page.py）：F1–F4 版面；特徵、規則、基準率為真資料，模型分數等標示假資料

之後規劃加入的頁面（見 DESIGN.md）：
  - 個股分析：輸入股票代號與分析日期，顯示 Agent 報告、相似案例卡片，可追問
  - （延伸）全市場篩選、回測成果

執行：streamlit run app.py
頁首須顯示免責聲明：研究與教學用途，不構成投資建議。

輸入：使用者操作
輸出：網頁
"""

import altair as alt
import pandas as pd
import streamlit as st

import analysis_page
import demo_pipeline as dp

DISCLAIMER = ("資料管線雛形：示範特徵與 5 年結果的計算，非模型預測，也不是選股建議。"
              "研究與教學用途，不構成投資建議。")
PCT_FEATURES = {"FCF 殖利率", "ROE", "毛利率", "年營收成長率", "負債比", "過去 12 個月含息報酬"}
COLOR_PRICE, COLOR_WEALTH = "#9AA0A6", "#1A73E8"
COLOR_KEY, COLOR_OTHER = "#1A73E8", "#C6DAFC"


# ───────────── 資料（快取） ─────────────
@st.cache_data(show_spinner=False)
def get_table():
    bar = st.progress(0.0, text="計算特徵中…")
    table = dp.build_table(progress=lambda i, n: bar.progress(
        i / n, text=f"計算特徵：{i:,} / {n:,} 檔（全市場第一次約需數分鐘，之後直接讀本機快取）"))
    bar.empty()
    return table


@st.cache_data
def get_analysis(stock_id, year):
    return dp.analyze(stock_id, year)


@st.cache_data
def get_wealth(stock_id, start, end):
    return dp.wealth_series(stock_id, start, end)


# ───────────── 格式 ─────────────
def fmt_pct(v, sign=True):
    if pd.isna(v):
        return "—"
    return f"{v:+.1%}" if sign else f"{v:.1%}"


def fmt_feature(name, v):
    if pd.isna(v):
        return "缺值"
    if name in PCT_FEATURES:
        return fmt_pct(v, sign=name in {"年營收成長率", "過去 12 個月含息報酬", "FCF 殖利率", "ROE"})
    if name == "市值（億元）":
        return f"{v:,.1f} 億元"
    if name == "營業現金流/淨利":
        return f"{v:.2f} 倍"
    return f"{v:.2f}"


# ───────────── 圖 ─────────────
def wealth_chart(a, height=260):
    """股價與含息財富兩條線（起點相同 = 特徵日收盤價），標出終點平均的 20 個交易日。"""
    ret = a["ret"]
    ws = get_wealth(a["stock_id"], ret["start_date"], dp.horizon_end(a["year"]))
    df = ws.reset_index().rename(columns={"index": "日期", "date": "日期", "close": "只看股價", "wealth": "含息財富"})
    long = df.melt("日期", ["只看股價", "含息財富"], var_name="序列", value_name="元")
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("日期:T", title=None),
        y=alt.Y("元:Q", title="元（起點 = 特徵日收盤價）"),
        color=alt.Color("序列:N", scale=alt.Scale(domain=["只看股價", "含息財富"], range=[COLOR_PRICE, COLOR_WEALTH]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("日期:T"), "序列:N", alt.Tooltip("元:Q", format=",.2f")],
    )
    layers = [lines]
    if ret["status"] != "下市暫估" and len(df) >= dp.END_PRICE_DAYS:
        band = pd.DataFrame({"start": [df["日期"].iloc[-dp.END_PRICE_DAYS]], "end": [df["日期"].iloc[-1]]})
        layers.insert(0, alt.Chart(band).mark_rect(opacity=0.12, color=COLOR_WEALTH).encode(x="start:T", x2="end:T"))
    return alt.layer(*layers).properties(height=height, title="股價走勢與含息財富（陰影 = 終點取平均的最後 20 個交易日）")


def revenue_chart(a, height=260):
    """第 y − 2 年 → 第 y + 4 年全年營收；期初（y − 1）與期末（y + 4）上色，不足 12 個月的年份註明。"""
    y = a["year"]
    df = dp.revenue_bars(a["stock_id"], y)
    df["角色"] = df["年度"].map({y - 1: "CAGR 期初", y + 4: "CAGR 期末"}).fillna("其他年度")
    df["標註"] = df["月數"].map(lambda m: "" if m == 12 else ("無資料" if m == 0 else f"只有 {m} 個月"))
    df["營收（億元）"] = df["營收（億元）"].fillna(0)
    base = alt.Chart(df).encode(x=alt.X("年度:O", title=None))
    bars = base.mark_bar().encode(
        y=alt.Y("營收（億元）:Q", title="全年營收（億元）"),
        color=alt.Color("角色:N", scale=alt.Scale(domain=["CAGR 期初", "CAGR 期末", "其他年度"],
                                                range=[COLOR_KEY, COLOR_KEY, COLOR_OTHER]), legend=None),
        tooltip=["年度:O", alt.Tooltip("營收（億元）:Q", format=",.1f"), "月數:Q"],
    )
    text = base.mark_text(dy=-6, fontSize=11, color="#B3261E").encode(y="營收（億元）:Q", text="標註:N")
    return (bars + text).properties(height=height, title=f"營收逐年（深色 = {y - 1} 年期初、{y + 4} 年期末）")


# ───────────── 區塊 1：我們要找什麼樣的股票 ─────────────
def example_sentence(kind, row):
    name, y = f"{row['名稱']}（{row['代號']}）", int(row["特徵年"])
    r, b, g = row["5 年含息報酬"], row["0050 同期含息報酬"], row["營收 CAGR"]
    if kind == "A":
        return (f"{name} 若在 {y} 年 4 月底買進並持有 5 年，含息報酬 {fmt_pct(r)}，比同期 0050 的 {fmt_pct(b)} "
                f"多出 {(r - b) * 100:.0f} 個百分點；營收從 {y - 1} 年到 {y + 4} 年每年成長 {fmt_pct(g, False)}，"
                f"股價上漲有營收成長撐腰——這就是我們要找的類型。")
    if kind == "B":
        return (f"{name} 從 {y} 年 4 月底持有 5 年，含息報酬 {fmt_pct(r)}，也贏過 0050 的 {fmt_pct(b)}；"
                f"但營收 5 年年複合成長只有 {fmt_pct(g, False)}，未達 10%。漲幅沒有營收成長支撐，"
                f"可能來自估值變高或景氣循環，不是我們要找的類型。")
    reasons = "；".join(x for x in [row["報酬說明"] if row["報酬狀態"] != "OK" else "", row["CAGR 說明"]] if x)
    return (f"{name} 的 {y} 年視窗資料不完整（{reasons}）。這類公司標為「無法判定」，"
            f"不當成失敗、也不當成成功，避免把資料缺漏誤讀成結果。")


def show_example(kind, title, row):
    with st.container(border=True):
        if row is None:
            st.markdown(f"#### {title}")
            st.info("目前資料中找不到符合條件的例子。")
            return
        a = get_analysis(row["代號"], int(row["特徵年"]))
        st.markdown(f"#### {title}：{row['名稱']}（{row['代號']}）· {int(row['特徵年'])} 年 4 月底")
        c = st.columns(4)
        c[0].metric("5 年含息報酬" + ("" if row["報酬狀態"] == "OK" else f"（{row['報酬狀態']}）"),
                    fmt_pct(row["5 年含息報酬"]))
        c[1].metric("同期 0050 含息報酬", fmt_pct(row["0050 同期含息報酬"]))
        c[2].metric("營收 5 年年複合成長", fmt_pct(row["營收 CAGR"], False) if row["CAGR 狀態"] == "OK" else "UNKNOWN")
        c[3].metric("判定（成長條件）", row["判定"])
        left, right = st.columns(2)
        left.altair_chart(wealth_chart(a), use_container_width=True)
        right.altair_chart(revenue_chart(a), use_container_width=True)
        st.markdown(f"**一句話**：{example_sentence(kind, row)}")


def section_target(table):
    st.subheader("我們要找什麼樣的股票")
    st.markdown(
        """
回到某一年的 **4 月底**，只看當時已公開的資料，想找出接下來 **5 年**會大漲、**而且大漲是因為公司真的在成長**的股票。
具體來說，「主要標記」要同時滿足兩個條件：

1. **報酬夠高**：從 4 月底起持有 5 年的含息報酬（股利再投入），在全市場排名 **前 5%**。
   終點取最後 20 個交易日（約一個月）的平均，避免剛好碰上短線高點或急跌。
2. **有營收成長撐腰**：營收從前一年（第 y − 1 年）到第 y + 4 年，**每年複合成長 ≥ 10%**（5 年累計約 +61%）。

只有條件 1 的股票，漲幅可能來自炒作或本益比變高；加上條件 2，才是我們想從基本面事先辨識的類型。
期間下市或資料不足的公司標為 **無法判定**，不當成失敗。
"""
    )
    st.caption("正式資料集（上市滿一年、流動性過濾後的母體）還沒建立，這裡先不做「全市場前 5%」排名，"
               "下面的例子改用「報酬遠高於同期 0050」示意條件 1。")
    ex_a, ex_b, ex_c = dp.pick_examples(table)
    show_example("A", "例子 A　我們要找的類型", ex_a)
    show_example("B", "例子 B　報酬高，但沒有營收成長支撐", ex_b)
    show_example("C", "例子 C　期間下市或資料不足 → 無法判定", ex_c)


# ───────────── 區塊 2：時光機 ─────────────
def section_time_machine(table):
    st.subheader("時光機：回到當年 4 月底")
    st.caption("特徵只用第 y − 1 年的年報與月營收、特徵日（含）以前的股價；5 年後的結果在特徵日當時還看不到，只用來事後檢驗。")
    stocks = table.drop_duplicates("代號")
    labels = {r["代號"]: f"{r['代號']} {r['名稱']}" for _, r in stocks.iterrows()}
    c1, c2 = st.columns([2, 1])
    default = list(labels).index("2330") if "2330" in labels else 0
    sid = c1.selectbox("股票", list(labels), index=default, format_func=labels.get)
    years = sorted(table.loc[table["代號"] == sid, "特徵年"].tolist())
    y = c2.selectbox("特徵年", years, index=len(years) - 1)
    a = get_analysis(sid, y)
    meta = a["meta"]

    st.markdown(
        f"**{meta['name']}（{sid}）**　特徵日 **{a['t']:%Y-%m-%d}**　市場：{meta['market']}　"
        f"產業：{meta['industry'] or '（已下市，無資料）'}"
        + ("" if a["listed_1y"] else "　⚠️ 上市未滿一年，正式樣本不納入")
    )

    st.markdown("##### 當時可用的 10 項特徵")
    feats = pd.DataFrame([{"特徵": k, "數值": fmt_feature(k, f["value"]), "計算口徑": f["formula"],
                           "來源資料": f["source"], "公告／可得時間": f["available"], "缺值原因": f["missing"]}
                          for k, f in a["features"].items()])
    st.dataframe(feats, hide_index=True, use_container_width=True, column_config={
        "特徵": st.column_config.TextColumn(width="small"),
        "數值": st.column_config.TextColumn(width="small"),
        "來源資料": st.column_config.TextColumn(width="large"),
        "缺值原因": st.column_config.TextColumn(width="medium"),
    })
    st.caption(f"產業類別取 TaiwanStockInfo 現況分類（非當年分類）；股數取特徵日的發行股數，沒有資料時才以股本 ÷ 面額 {dp.FACE_VALUE} 元推估。")

    st.markdown(f"##### 5 年後的結果（{a['t']:%Y-%m-%d} → {dp.horizon_end(y):%Y-%m-%d}）")
    ret, bm = a["ret"], a["bm"]
    c = st.columns(4)
    c[0].metric("含息報酬" + ("" if ret["status"] == "OK" else f"（{ret['status']}）"), fmt_pct(ret["total"]))
    c[1].metric("只看股價", fmt_pct(ret["price_only"]))
    c[2].metric("同期 0050 含息報酬", fmt_pct(bm["total"]))
    c[3].metric("營收 5 年 CAGR", fmt_pct(a["cagr"], False) if a["cagr_status"] == "OK" else "UNKNOWN")
    c = st.columns(4)
    c[0].metric("判定（營收成長條件 ≥ 10%）", a["verdict"])
    c[1].metric("前 5% 報酬排名", "待全量母體")
    notes = [f"報酬：{ret['note']}" if ret["note"] else "", f"營收 CAGR：{a['cagr_why']}" if a["cagr_why"] else "",
             f"0050：{bm['note']}"]
    st.caption("　".join(n for n in notes if n))
    if ret["status"] in {"下市暫估", "價格異常"}:
        st.warning("這筆報酬不可靠（" + ret["status"] + "），僅供參考。")

    left, right = st.columns(2)
    left.altair_chart(wealth_chart(a), use_container_width=True)
    right.altair_chart(revenue_chart(a), use_container_width=True)


# ───────────── 區塊 3：總表 ─────────────
def section_table(table):
    st.subheader(f"總表（{table['代號'].nunique():,} 檔 × {len(dp.FEATURE_YEARS)} 年）")
    st.caption("每檔每年一列；點欄位標題可排序。報酬狀態不是 OK 的列（下市暫估、價格異常）報酬僅供參考。")
    c1, c2 = st.columns(2)
    years = c1.multiselect("特徵年", dp.FEATURE_YEARS, default=dp.FEATURE_YEARS)
    verdicts = c2.multiselect("判定", ["符合成長條件", "不符合成長條件", "無法判定"],
                              default=["符合成長條件", "不符合成長條件", "無法判定"])
    df = table[table["特徵年"].isin(years) & table["判定"].isin(verdicts)].copy()
    pct_cols = ["FCF 殖利率", "ROE", "毛利率", "年營收成長率", "負債比", "過去 12 個月含息報酬",
                "5 年含息報酬", "只看股價報酬", "0050 同期含息報酬", "超越 0050", "營收 CAGR"]
    df[pct_cols] = df[pct_cols] * 100   # 顯示成百分比，仍保留數值以便排序
    config = {k: st.column_config.NumberColumn(k, format="%.1f%%") for k in pct_cols}
    config.update({
        "代號": st.column_config.TextColumn("代號", pinned=True),
        "名稱": st.column_config.TextColumn("名稱", pinned=True),
        "市值（億元）": st.column_config.NumberColumn("市值（億元）", format="%.1f"),
        "帳面/市值": st.column_config.NumberColumn("帳面/市值", format="%.2f"),
        "本益比": st.column_config.NumberColumn("本益比", format="%.1f"),
        "營業現金流/淨利": st.column_config.NumberColumn("營業現金流/淨利", format="%.2f"),
        "特徵年": st.column_config.NumberColumn("特徵年", format="%d"),
    })
    first = ["代號", "名稱", "特徵年", "判定", "5 年含息報酬", "0050 同期含息報酬", "超越 0050", "報酬狀態",
             "營收 CAGR", "CAGR 狀態", "排名", "缺值數"]
    order = first + [c for c in df.columns if c not in first]
    st.dataframe(df, hide_index=True, use_container_width=True, height=620, column_config=config, column_order=order)
    st.caption(f"共 {len(df)} 列；空白（None）＝缺值，原因見最右側「缺值原因」欄。"
               "判定只看營收成長條件（≥ 10%），報酬排名欄一律為「待正式資料集」。")


# ───────────── 主程式 ─────────────
def main():
    st.set_page_config(page_title="Multibagger Scout｜資料管線 Demo", layout="wide")
    st.warning(DISCLAIMER)
    st.title("Multibagger Scout｜資料管線 Demo")
    st.caption(f"資料：{dp.SOURCE_LABEL}　｜　{dp.DATA_ATTRIBUTION}")
    missing = dp.missing_cache()
    if missing:
        st.error("cache/ 缺少 Demo 需要的資料：" + "、".join(missing) + "\n\n"
                 "請先執行 `python verify_data.py`，再補抓 0050（2 次 API）：\n\n"
                 f"`{dp.FETCH_BENCHMARK_CMD}`")
        st.stop()
    table = get_table()
    tab1, tab2, tab3, tab4 = st.tabs(["① 我們要找什麼樣的股票", "② 時光機", "③ 總表", "④ 個股分析（骨架）"])
    with tab1:
        section_target(table)
    with tab2:
        section_time_machine(table)
    with tab3:
        section_table(table)
    with tab4:
        analysis_page.section_analysis()
    st.divider()
    st.caption(dp.DATA_ATTRIBUTION + "原始資料只存在本機，不隨專案上傳。")


if __name__ == "__main__":
    main()
