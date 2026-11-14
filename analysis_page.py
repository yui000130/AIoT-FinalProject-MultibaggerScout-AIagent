"""
analysis_page.py －Streamlit 個股分析頁骨架（F1–F4，DESIGN.md 2.1）

app.py 的分頁「④ 個股分析（骨架）」。版面先定好，之後 train.py、agent.py 完成再換成真的：
  真資料：10 項特徵（data/dataset.parquet）、領域規則（rules.py）、基準率（已揭曉的訓練樣本正例率）
  假資料：模型分數、潛力等級、SHAP 支持／反對證據、相似案例、LLM 報告與追問回覆
          （全部標示「假資料」，由固定亂數種子產生，不是模型結果）

時間一致性：基準率、規則門檻只用「標記揭曉日 ≤ 分析日」的特徵年；沒有已揭曉樣本時，
規則門檻改用分析日當年的全市場特徵分布（特徵在分析日都已公開）。

"""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import demo_pipeline as dp
import rules

DATASET = dp.BASE / "data" / "dataset.parquet"
FAKE = "🧪 假資料"


@st.cache_data
def load_dataset():
    return pd.read_parquet(DATASET) if DATASET.exists() else None


def revealed_rows(data, t):
    """標記揭曉日 ≤ 分析日的列（DESIGN 5.5 第 10 條）。"""
    return data[pd.to_datetime(data["標記揭曉日"]) <= pd.Timestamp(t)]


def _rng(sid, year):
    seed = int(hashlib.md5(f"{sid}-{year}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def fake_model_output(row):
    """假的模型分數、SHAP 與相似案例（之後由 train.py / agent.py 取代）。"""
    rng = _rng(row["代號"], row["特徵年"])
    score = float(rng.beta(1.2, 20))
    feats = [f for f in dp.FEATURES if pd.notna(row[f])]
    contrib = pd.DataFrame({"特徵": feats, "數值": [row[f] for f in feats],
                            "貢獻（假）": rng.normal(0, 0.01, len(feats))})
    contrib = contrib.reindex(contrib["貢獻（假）"].abs().sort_values(ascending=False).index)
    return score, contrib


def potential_level(score, base):
    if base is None or base == 0:
        return "—"
    r = score / base
    return "高" if r >= 2 else ("中" if r >= 1 else "低")


def section_analysis():
    st.subheader("個股分析（骨架）")
    st.info("這一頁先把版面定好。**標示「🧪 假資料」的區塊不是模型結果**，等模型（`train.py`）與 Agent（`agent.py`）完成後替換；"
            "特徵、領域規則與基準率是真的。")
    data = load_dataset()
    if data is None:
        st.error("找不到 data/dataset.parquet，請先執行 `python build_dataset.py`。")
        return

    # ── F1 輸入 ──
    stocks = data.drop_duplicates("代號").set_index("代號")["名稱"]
    c1, c2 = st.columns([2, 1])
    default = list(stocks.index).index("2330") if "2330" in stocks.index else 0
    sid = c1.selectbox("股票代號", list(stocks.index), index=default,
                       format_func=lambda s: f"{s} {stocks[s]}", key="ap_sid")
    years = sorted(data.loc[data["代號"] == sid, "特徵年"].tolist())
    year = c2.selectbox("分析日期（當年 4 月底）", years, index=len(years) - 1,
                        format_func=lambda y: f"{dp.feature_date(y):%Y-%m-%d}", key="ap_year")
    row = data[(data["代號"] == sid) & (data["特徵年"] == year)].iloc[0]
    t = dp.feature_date(year)
    train = revealed_rows(data, t)

    # ── F1 模型分數與基準率 ──
    st.markdown(f"#### {row['名稱']}（{sid}）　分析日 {t:%Y-%m-%d}　產業：{row['產業']}")
    base = float((train["y_main"] == 1).sum() / train["y_main"].notna().sum()) if len(train) else None
    score, contrib = fake_model_output(row)
    c = st.columns(3)
    c[0].metric(f"模型分數 {FAKE}", f"{score:.3f}")
    c[1].metric("基準率（已揭曉訓練樣本的主要標記正例率）",
                f"{base:.1%}" if base is not None else "尚無已揭曉樣本")
    c[2].metric(f"潛力等級 {FAKE}", potential_level(score, base))
    st.caption("模型分數未經機率校準，只用來排序，不稱為機率。基準率只用標記揭曉日 ≤ 分析日的樣本"
               + ("" if len(train) else "；2018 年以前的分析日還沒有任何已揭曉的 5 年結果") + "。")

    # ── F2 支持／反對證據 ──
    st.markdown("#### 支持與反對證據")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**特徵貢獻（SHAP）{FAKE}**")
        show = contrib.head(6).copy()
        show["方向"] = np.where(show["貢獻（假）"] > 0, "支持", "反對")
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"數值": st.column_config.NumberColumn(format="%.3f"),
                                    "貢獻（假）": st.column_config.NumberColumn(format="%+.4f")})
    with right:
        st.markdown("**領域規則（真資料，`rules.py`）**")
        if len(train):
            th, th_src = rules.compute_thresholds(train), f"已揭曉訓練特徵年 {', '.join(map(str, sorted(train['特徵年'].unique())))}"
        else:
            th, th_src = rules.compute_thresholds(data[data["特徵年"] == year]), f"{year} 年當年全市場特徵分布（尚無已揭曉樣本）"
        for r in rules.run_all_rules(row, th):
            icon = {True: "🚩", False: "✅", None: "➖"}[r["triggered"]]
            st.markdown(f"{icon} **{r['rule']}**：{r['reason']}")
            st.caption(f"門檻來源：{th_src}；說明與文獻：`{r['doc']}`")

    # ── F3 相似案例 ──
    st.markdown(f"#### 歷史相似案例（kNN）{FAKE}")
    pool = train[train["y_main"].notna() & (train["代號"] != sid)]
    if pool.empty:
        st.caption("分析日之前還沒有已揭曉 5 年結果的樣本，沒有相似案例（DESIGN 5.5 第 6 條）。")
    else:
        pick = pool.sample(min(4, len(pool)), random_state=int(year))
        cols = st.columns(len(pick))
        for col, (_, p) in zip(cols, pick.iterrows()):
            with col.container(border=True):
                st.markdown(f"**{p['代號']} {p['名稱']}**　{p['特徵年']} 年")
                st.markdown("成功案例" if p["y_main"] == 1 else "未成為多倍股")
                st.caption(f"5 年含息報酬 {p['5 年含息報酬']:+.0%}　｜　相似度：假")
        st.caption("目前是隨機挑選的已揭曉樣本，只示範卡片版面；之後改為依特徵距離找最相近的案例。")

    # ── 報告與 F4 追問 ──
    st.markdown(f"#### Agent 報告 {FAKE}")
    st.markdown(f"> （LLM 報告位置）{row['名稱']} 在 {t:%Y-%m-%d} 的模型分數為 {score:.3f}，"
                f"基準率 {base:.1%}。" if base is not None else
                f"> （LLM 報告位置）{row['名稱']} 在 {t:%Y-%m-%d} 尚無基準率可對照。")
    st.caption("報告中的每個數字之後由 agent.py 與工具結果逐一核對（E4）。")
    key = f"chat_{sid}_{year}"
    history = st.session_state.setdefault(key, [])
    for role, msg in history:
        st.chat_message(role).write(msg)
    q = st.chat_input("針對這份報告追問（目前只回覆假訊息）", key=f"ci_{key}")
    if q:
        history.append(("user", q))
        history.append(("assistant", f"{FAKE}：Agent 尚未完成，之後會根據工具結果回答「{q}」。"))
        st.rerun()
