"""
evaluate.py －回測與 Baseline 比較（規劃中）

執行 DESIGN.md 第 11 節的實驗：
  E1：預測主要標記；LightGBM vs 邏輯迴歸 vs 隨機 vs Yartseva 規則篩選 vs 單一因子
      指標：各輪與平均的 Precision@20/50、Recall@50、PR-AUC
  E2：每輪前 50 檔等權組合 vs 0050 買進持有
      指標：5 年累積報酬、最大回撤
  E3：美股多倍股因子在台股是否成立；四組對照：基本面暴漲股、炒作型暴漲股、普通股、期間下市公司
      指標：SHAP 特徵排名、四組特徵中位數、「有此特徵者成為暴漲股的比例」對比基準率
  E5（延伸）：敏感度分析：① 觀察期 3 年 vs 5 年；② 主要標記 vs 對照標記；③ 營收門檻 5% / 10% / 15%
      指標：各設定的 Precision@50 比較
（E4 為 Agent 報告評估，見 agent.py）

輸入：data/dataset.parquet、models/
輸出：results/ 表格與圖
"""


def precision_at_k(y_true, scores, k):
    """前 k 名中真正暴漲股的比例。"""
    # TODO
    raise NotImplementedError


def recall_at_k(y_true, scores, k):
    """全部暴漲股中，被排進前 k 名的比例。"""
    # TODO
    raise NotImplementedError


def baseline_scores(df, method):
    """產生 Baseline 分數：random、yartseva_rules、single_factor。"""
    # TODO
    raise NotImplementedError


def run_e1():
    """E1：模型與 Baseline 的排序指標比較。"""
    # TODO
    raise NotImplementedError


def run_e2():
    """E2：前 50 檔等權組合 vs 0050 的累積報酬與最大回撤。"""
    # TODO
    raise NotImplementedError


def run_e3():
    """E3：SHAP 特徵排名、四組特徵中位數與命中率。"""
    # TODO
    raise NotImplementedError


def run_e5():
    """（延伸）E5：觀察期、標記定義、營收門檻的敏感度分析。"""
    # TODO
    raise NotImplementedError


def main():
    """依序執行 E1、E2、E3、E5，結果存到 results/。"""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
