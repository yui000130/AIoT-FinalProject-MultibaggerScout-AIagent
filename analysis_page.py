"""
analysis_page.py －Streamlit 個股分析頁（F1–F4，DESIGN.md 2.1）

app.py 的分頁「④ 個股分析」。所有數字都來自 agent.py 的工具（程式計算），LLM 只整理文字：
  F1 模型分數、當年名次、潛力等級、訓練期基準率（agent.score_model）
  F2 支持與反對證據：SHAP（agent.score_model）、2 條領域規則與固定說明（agent.check_rules、knowledge/）
  F3 已揭曉的相似成功與失敗案例（agent.find_cases，特徵年 ≤ 分析日 − 5 年）
  報告：按下按鈕才呼叫 LLM（節省免費額度）；未呼叫前顯示程式產生的模板報告；每個數字都經 verify_report 核對
  F4 追問（agent.analyze 的 follow_up，只用這份報告的證據回答）

時間一致性：分析日 t 的模型、規則門檻、案例都只用 t 之前已揭曉的標記（DESIGN 5.5、5.6）。
輸入：data/dataset.parquet、data/live_features.parquet、models/（build_dataset.py、train.py）
輸出：網頁（Streamlit，只在本機執行）
"""

import pandas as pd
import streamlit as st

import agent
import demo_pipeline as dp

STATUS_TEXT = {
    "ok": "✅ LLM 報告（已通過數字核對）",
    "template": "🧾 模板報告（程式產生；尚未呼叫 LLM，或 LLM 三輪都未通過核對）",
    "limited": "➖ 限制版報告（此分析日不評分，只列特徵與領域規則）",
}


@st.cache_data(show_spinner=False)
def stock_options():
    df = agent.all_features()
    last = df.sort_values("特徵年").drop_duplicates("代號", keep="last").set_index("代號")["名稱"]
    years = df.groupby("代號")["特徵年"].apply(lambda s: sorted(s.tolist()))
    return last.sort_index(), years


def get_result(sid, year, use_llm):
    """同一 (代號, 年, 是否用 LLM) 只算一次，存在 session_state（LLM 回應另有磁碟快取）。"""
    key = ("ap_result", sid, year, use_llm)
    if key not in st.session_state:
        st.session_state[key] = agent.run_analysis(sid, year, use_llm=use_llm)
    return st.session_state[key]


def show_text_items(title, items):
    if not items:
        return
    st.markdown(f"**{title}**")
    for it in items:
        st.markdown(f"- {it['text'] if isinstance(it, dict) else it}")


def show_report(res):
    rep = res["report"]
    st.markdown(f"#### Agent 報告　{STATUS_TEXT.get(res['status'], res['status'])}")
    if res.get("llm_error"):
        st.warning(f"LLM 無法使用：{res['llm_error']}；改用模板報告。")
    st.markdown(rep["summary"])
    c1, c2 = st.columns(2)
    with c1:
        show_text_items("支持證據", rep.get("supporting"))
    with c2:
        show_text_items("反對證據與注意事項", rep.get("opposing"))
    show_text_items("相似案例", rep.get("cases"))
    show_text_items("領域規則", rep.get("rules"))
    show_text_items("不確定性", rep.get("uncertainty"))
    st.caption("方括號 [E###] 是證據編號，對應下方證據表；報告中的數字一律來自工具輸出並經程式核對。"
               "模型分數未經機率校準，不是上漲機率。本頁為研究與教學用途，不構成投資建議。")
    with st.expander("核對紀錄與證據表"):
        if res.get("attempts"):
            for a in res["attempts"]:
                st.markdown(f"第 {a['attempt']} 輪：" + ("通過" if not a["errors"] else f"{len(a['errors'])} 個問題"))
                for e in a["errors"][:20]:
                    st.caption(f"・{e}")
        else:
            st.caption("沒有呼叫 LLM。")
        ev = pd.DataFrame(res["evidence"])
        ev["顯示"] = [agent.fmt(v, u, f) for v, u, f in zip(ev["value"], ev["unit"], ev["field"])]
        st.dataframe(ev[["evidence_id", "stock_id", "as_of", "field", "顯示", "unit", "source"]],
                     hide_index=True, width="stretch")
        st.caption(f"模型：{res['model']}　提示詞版本：{res['prompt_version']}　工具呼叫 {res['tool_calls']} 次")


def section_analysis():
    st.subheader("個股分析")
    if not (agent.DATASET.exists() and agent.LIVE_DATASET.exists()):
        st.error("找不到資料集，請先執行 `python build_dataset.py` 與 `python train.py`。")
        return
    names, years_of = stock_options()

    # ── F1 輸入 ──
    if "ap_sid" not in st.session_state:
        st.session_state["ap_sid"] = "2330" if "2330" in names.index else names.index[0]
    c1, c2 = st.columns([2, 1])
    sid = c1.selectbox("股票代號", list(names.index), format_func=lambda s: f"{s} {names[s]}", key="ap_sid")
    years = years_of[sid]
    if st.session_state.get("ap_year") not in years:
        st.session_state["ap_year"] = years[-1]
    year = c2.selectbox("分析日（當年 4 月底特徵日）", years, key="ap_year",
                        format_func=lambda y: f"{dp.feature_date(y):%Y-%m-%d}")

    use_llm = st.session_state.get(("ap_llm", sid, year), False)
    with st.spinner("計算特徵、模型分數、相似案例與領域規則…" if not use_llm else "LLM 撰寫報告並核對數字…"):
        res = get_result(sid, year, use_llm)
    if res["status"] == "clarify":
        st.warning(res["message"])
        return
    sc, feats = res["score"], res["features"]
    st.markdown(f"#### {res['name']}（{sid}）　分析日 {res['as_of']}　產業：{feats['industry']}")

    # ── F1 模型分數、名次、潛力等級、基準率 ──
    c = st.columns(4)
    if sc.get("ok"):
        c[0].metric("潛力等級", sc["level"])
        c[1].metric("當年名次", f"{sc['rank']} / {sc['n']}")
        c[2].metric("模型分數（未校準）", f"{sc['score']:.3f}")
        c[3].metric("基準率（訓練期正例率）", f"{sc['base_rate']:.1%}")
        st.caption(f"潛力等級：當年全市場評分樣本前 5% 為高、5%～20% 為中、其餘為低。"
                   f"模型訓練特徵年 {sc['train_years'][0]}–{sc['train_years'][-1]}；{sc['tested']}。")
    else:
        st.info(sc["reason"])

    # ── F2 支持與反對證據 ──
    st.markdown("#### 支持與反對證據")
    left, right = st.columns(2)
    with left:
        st.markdown("**特徵貢獻（SHAP，LightGBM pred_contrib）**")
        if sc.get("ok"):
            f = feats["features"]
            rows = [{"特徵": s["feature"],
                     "數值": agent.fmt(f[s["feature"]]["value"], f[s["feature"]]["unit"], s["feature"])
                     if s["feature"] in f else feats["industry"],
                     "貢獻": s["value"], "方向": "推高" if s["value"] > 0 else "拉低"} for s in sc["shap"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"貢獻": st.column_config.NumberColumn(format="%+.3f")})
            st.caption("貢獻為 log-odds 單位；描述模型依賴，不是經濟因果。")
        else:
            st.caption("此分析日不評分，沒有 SHAP。")
    with right:
        st.markdown("**領域規則（`rules.py`，不改變模型分數）**")
        for r in res["rules"].get("rules", []):
            icon = {True: "🚩", False: "✅", None: "➖"}[r["triggered"]]
            st.markdown(f"{icon} **{r['rule']}**：{r['reason']}")
            st.caption(f"門檻來源：{r['threshold_source']}；文獻：{res['rule_docs'][r['rule']]['citation']}")
            with st.expander(f"「{r['rule']}」固定說明與注意事項（`{r['doc']}`）"):
                st.markdown(res["rule_docs"][r["rule"]]["what"])
                st.markdown(res["rule_docs"][r["rule"]]["caveats"])

    with st.expander("10 項特徵與規則輸入"):
        st.dataframe(pd.DataFrame([{"特徵": k, "數值": agent.fmt(v["value"], v["unit"], k), "來源": v["source"],
                                    "缺值原因": v["missing"]} for k, v in feats["features"].items()]),
                     hide_index=True, width="stretch")

    # ── F3 相似案例 ──
    st.markdown("#### 歷史相似案例（kNN，只用結果已揭曉的樣本）")
    cases = res["cases"]
    if not cases.get("success") and not cases.get("failure"):
        st.caption(cases.get("note") or "此時點尚無已揭曉的歷史案例（DESIGN 5.5 第 6 條）。")
    else:
        for label, key in (("成功案例（主要標記 = 1）", "success"), ("失敗案例（主要標記 = 0）", "failure")):
            st.markdown(f"**{label}**")
            cols = st.columns(agent.K_CASES)
            for col, cs in zip(cols, cases[key]):
                with col.container(border=True):
                    st.markdown(f"**{cs['stock_id']} {cs['name']}**　{cs['year']} 年")
                    ret = f"{cs['return_5y']:+.1%}" if cs["return_5y"] is not None else "缺值"
                    st.markdown(f"之後 5 年含息報酬 **{ret}**")
                    st.caption(f"產業：{cs['industry']}｜距離 {cs['distance']:.3f}｜案例填補 {cs['filled']} 項")
        st.caption(f"案例池：特徵年 {cases['pool_years'][0]}–{cases['pool_years'][-1]}、{cases['pool_n']} 筆，"
                   f"排除本公司；距離為 10 項特徵當年百分位排名的歐氏距離（本股填補 {cases['my_filled']} 項）。"
                   "相似不代表會有相同結果。")

    # ── 報告 ──
    if res["status"] != "limited" and not use_llm:
        if st.button("用 LLM 撰寫報告（會使用 Gemini 額度；結果會快取，同一檔同一分析日只呼叫一次）", type="primary"):
            st.session_state[("ap_llm", sid, year)] = True
            st.rerun()
    show_report(res)

    # ── F4 追問 ──
    st.markdown("#### 追問")
    hkey = ("ap_chat", sid, year)
    history = st.session_state.setdefault(hkey, [])
    for role, msg in history:
        st.chat_message(role).write(msg)
    q = st.chat_input("針對這份報告追問，例如「為什麼估值過高？」")
    if q:
        out = agent.analyze(q, session=res)
        plan = out.get("plan", {})
        if plan.get("intent") == "analyze" and out.get("stock_id") and out["stock_id"] != sid:
            st.session_state["ap_sid"] = out["stock_id"]        # 使用者改問另一檔：切換到該股
            st.session_state["ap_year"] = int(out["as_of"][:4])
            st.session_state[("ap_result", out["stock_id"], int(out["as_of"][:4]), True)] = out
            st.session_state[("ap_llm", out["stock_id"], int(out["as_of"][:4]))] = True
            st.rerun()
        history.append(("user", q))
        if out.get("status") == "clarify":
            history.append(("assistant", out["message"]))
        elif "answer" in out:
            tag = "" if out["status"] == "ok" else "（限制說明）"
            history.append(("assistant", f"{out['answer']}{tag}"))
        else:
            history.append(("assistant", "這個問題請改用上方的股票選單重新分析。"))
        st.rerun()
