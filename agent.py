"""
agent.py －LLM Agent（規劃中）

核心原則：數字一律由程式與模型計算，LLM 只負責規劃與解釋；
報告中的數字會自動與工具輸出核對，不一致就重寫（最多 3 次，失敗則改用模板報告）。

流程見 DESIGN.md 第 8 節 Algorithm 1。也負責實驗 E4（Agent 報告 vs 單純 LLM；核心為數字正確率，延伸為人工評分）。

輸入：使用者查詢（股票代號、分析日期、篩選條件、追問）
輸出：結構化分析報告（潛力等級、模型分數 vs 基準率、支持與反對證據、相似案例、不確定性）
"""

BASE_RATE = None  # 訓練資料中主要標記的正例比例，由 train.py 計算後讀入
MAX_REWRITE = 3


def classify_intent(query):
    """用 LLM 判斷意圖：個股分析、全市場篩選或追問。"""
    # TODO
    raise NotImplementedError


def plan_tasks(intent, query):
    """用 LLM 把查詢拆成工具呼叫序列。"""
    # TODO
    raise NotImplementedError


def run_tools(plan):
    """依序執行工具：取得特徵、模型預測、SHAP、kNN、規則檢查。"""
    # TODO
    raise NotImplementedError


def retrieve_similar_cases(features, analysis_date, k=5):
    """kNN 只在「特徵日 ≤ 分析日 − 5 年」的歷史樣本中找相似的成功與失敗案例（結果已揭曉）。"""
    # TODO: 沒有可用樣本時（分析日早於 2018-04）回傳空清單，報告顯示「尚無已揭曉的歷史案例」
    raise NotImplementedError


def explain_rules(rule_results):
    """從 knowledge/ 讀取觸發規則的固定說明與文獻出處，供報告引用。"""
    # TODO
    raise NotImplementedError


def write_report(tool_outputs, rule_explanations):
    """用 LLM 把工具輸出寫成報告。"""
    # TODO
    raise NotImplementedError


def verify_numbers(report, tool_outputs):
    """抽出報告中的所有數字，檢查是否都能對應到工具輸出；回傳不一致清單。"""
    # TODO
    raise NotImplementedError


def template_report(tool_outputs):
    """LLM 重寫失敗時，以固定模板產生報告。"""
    # TODO
    raise NotImplementedError


def analyze(query):
    """Agent 主流程入口。"""
    # TODO: classify_intent → plan_tasks → run_tools → write_report ⇄ verify_numbers
    raise NotImplementedError
