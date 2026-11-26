"""
train.py －三輪 walk-forward 訓練 LightGBM 與邏輯迴歸

依 DESIGN.md 5.6 的切分訓練兩種模型，目標為主要標記 y_main（基本面驅動暴漲股）：
  - LightGBM：主模型（DESIGN 6）
  - 邏輯迴歸：Baseline（DESIGN 6）

切分（訓練視窗的 5 年觀察期必須在測試起點前結束）：
  第 1 輪：訓練 2013        → 測試 2018
  第 2 輪：訓練 2013–2014   → 測試 2019
  第 3 輪：訓練 2013–2015   → 測試 2020

資料規則：
  - 特徵只用 10 項＋產業類別（DESIGN 5.3），以白名單固定；標記、報酬、營收 CAGR 等欄位不進入模型
  - 訓練列只用 y_main 為 0 / 1 者；UNKNOWN 排除，不當成負例（DESIGN 5.4）
  - 測試年的所有樣本都產生模型分數（含 UNKNOWN），由 evaluate.py 決定可評估母體
  - 填補、縮放、類別編碼只用訓練集擬合（DESIGN 5.3）
  - 邏輯迴歸的數值特徵先轉為各特徵年內的百分位排名（只用同年特徵，DESIGN 6）；
    之後用 models/logit_round{n} 評分時，輸入也要先經 rank_by_year 轉換
  - 類別不平衡以類別權重處理（DESIGN 6）；輸出一律稱為「模型分數」，不是機率
  - 參數事先固定（2026-11-18 決定），不依測試結果調整（DESIGN 5.6）

時間一致性檢查（DESIGN 5.5 第 10 條，assert，不通過就停止）：
  1. 訓練特徵年 + 5 ≤ 測試年
  2. 所有訓練列的「標記揭曉日」≤ 測試特徵日（不能只用特徵年 + 5 代替）
  3. 訓練集與測試集沒有重疊的列；模型欄位剛好等於特徵白名單

個股分析用的上線版模型（DESIGN 5.6「個股分析的分析日與模型對應」，2026-11-24）：
  分析日 t ＝ 2021–2026：LightGBM 以特徵年 2013 ～ min(t − 5, 2020) 重訓，參數與三輪相同、不重新選參；
  這些模型沒有測試結果，不寫入 results/。時間一致性同樣檢查：訓練標記揭曉日 ≤ t 的特徵日。

輸入：data/dataset.parquet、data/live_features.parquet（build_dataset.py；後者只用來取 t 的特徵日）
輸出：models/lgbm_round{n}.joblib、models/logit_round{n}.joblib（模型，只在本機）
      models/scores_round{n}.parquet（測試年每家公司的模型分數，給 evaluate.py；只在本機）
      models/lgbm_live{t}.joblib（上線版模型，t ＝ 2021–2026；只在本機）
      results/train_summary.csv（各輪訓練／測試筆數、正例數、訓練正例率；只有統計數字）

執行方式：python train.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "dataset.parquet"
MODEL_DIR = BASE / "models"
SUMMARY_CSV = BASE / "results" / "train_summary.csv"

FOLDS = [
    {"train": [2013], "test": 2018},
    {"train": [2013, 2014], "test": 2019},
    {"train": [2013, 2014, 2015], "test": 2020},
]
HORIZON_YEARS = 5
LIVE_DATASET = BASE / "data" / "live_features.parquet"
LIVE_YEARS = list(range(2021, 2027))   # 上線版模型的分析日（特徵年）
LAST_LABEL_YEAR = 2020                 # 標記範圍只到 2020 特徵年（DESIGN 5.6）
SEED = 42
TARGET = "y_main"

# 10 項特徵（DESIGN 5.3）：資料集欄名 → 模型內部英文欄名（LightGBM 不接受部分特殊字元）
NUMERIC_FEATURES = {
    "市值（億元）": "mcap",
    "FCF 殖利率": "fcf_yield",
    "帳面/市值": "book_to_market",
    "本益比": "pe",
    "ROE": "roe",
    "毛利率": "gross_margin",
    "年營收成長率": "revenue_growth",
    "營業現金流/淨利": "ocf_to_ni",
    "負債比": "debt_ratio",
    "過去 12 個月含息報酬": "ret_12m",
}
CATEGORY_FEATURE = {"產業": "industry"}
FEATURE_COLS = list(NUMERIC_FEATURES.values()) + list(CATEGORY_FEATURE.values())

# 事先固定的參數（2026-11-18；第 1 輪只有 17 個正例，無法在訓練資料內選參，採保守的淺樹）
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    class_weight="balanced",
    random_state=SEED,
    deterministic=True,
    n_jobs=1,
    verbose=-1,
)
LOGIT_C = 1.0


def load_dataset():
    """讀取 data/dataset.parquet。"""
    if not DATASET.exists():
        sys.exit(f"找不到 {DATASET}，請先執行 python build_dataset.py")
    return pd.read_parquet(DATASET)


def to_model_frame(df, categories):
    """取出特徵白名單並改成英文欄名；產業轉成以訓練集類別固定的 category（新產業 → 缺值）。"""
    X = df[list(NUMERIC_FEATURES) + list(CATEGORY_FEATURE)].rename(
        columns={**NUMERIC_FEATURES, **CATEGORY_FEATURE})
    X["industry"] = pd.Categorical(X["industry"], categories=categories)
    assert list(X.columns) == FEATURE_COLS, "模型欄位與特徵白名單不一致"
    return X


def split(df, fold):
    """依 fold 回傳訓練集（y_main 可判定）與測試集（測試年全部樣本），並做時間一致性檢查。"""
    test_year = fold["test"]
    train_all = df[df["特徵年"].isin(fold["train"])]
    train = train_all[train_all[TARGET].notna()]
    test = df[df["特徵年"] == test_year]

    # 檢查 1：訓練特徵年 + 5 ≤ 測試年
    assert max(fold["train"]) + HORIZON_YEARS <= test_year, "訓練觀察期超過測試起點"
    # 檢查 2：訓練標記在測試特徵日（收盤後評分）之前已揭曉
    test_dates = test["特徵日"].unique()
    assert len(test_dates) == 1, "測試年的特徵日不唯一"
    test_date = test_dates[0]
    assert train["標記揭曉日"].notna().all(), "有訓練列沒有標記揭曉日"
    assert (train["標記揭曉日"] <= test_date).all(), "有訓練標記在測試特徵日之後才揭曉"
    # 檢查 3：訓練與測試沒有重疊
    assert train.index.intersection(test.index).empty, "訓練集與測試集有重疊的列"

    info = {
        "訓練 UNKNOWN 排除": int(len(train_all) - len(train)),
        "訓練標記最晚揭曉日": max(train["標記揭曉日"]),
        "測試特徵日": test_date,
    }
    return train, test, info


def train_lightgbm(X, y):
    """訓練 LightGBM：缺值直接交給模型，產業為原生類別特徵，類別權重處理不平衡。"""
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y, categorical_feature=["industry"])
    return model


def rank_by_year(df):
    """10 項數值特徵在各特徵年內的百分位排名（0～1；缺值保留）。

    每年分開排名（2026-11-18，與 Gu, Kelly & Xiu [3] 每期橫斷面排名相同）：只用同一特徵年、
    同一特徵日的全部樣本特徵（特徵日當天都已公開），不用標記，也不跨年；
    營業現金流／淨利等比率的極端值與各年整體水準差異都不會主導線性模型。
    """
    num = list(NUMERIC_FEATURES)
    ranks = df.groupby("特徵年")[num].rank(pct=True, method="average")
    return ranks.rename(columns=NUMERIC_FEATURES)


def to_logit_frame(X, ranks):
    """邏輯迴歸的輸入：數值特徵換成當年百分位排名，產業維持原樣。"""
    Xl = X.copy()
    Xl[list(NUMERIC_FEATURES.values())] = ranks.loc[X.index]
    Xl["industry"] = Xl["industry"].astype(object)
    return Xl


def train_logistic(X, y):
    """訓練邏輯迴歸 Baseline：輸入為當年百分位排名（rank_by_year）→ 訓練集中位數填補 → 標準化；產業 one-hot。"""
    numeric = list(NUMERIC_FEATURES.values())
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("industry", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="未知")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["industry"]),
    ])
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=LOGIT_C, class_weight="balanced",
                                   max_iter=2000, random_state=SEED)),
    ])
    model.fit(X, y)
    return model


def save_model(model, name, round_no):
    """把模型存到 models/。"""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}_round{round_no}.joblib"
    joblib.dump(model, path)
    return path


def train_years_for(t_year):
    """分析日（特徵年）t 可用的訓練特徵年：2013 ～ min(t − 5, 2020)；t < 2018 回傳空清單（不評分）。"""
    last = min(t_year - HORIZON_YEARS, LAST_LABEL_YEAR)
    return list(range(2013, last + 1))


def industry_categories(train):
    """LightGBM 的產業類別清單：取自訓練集（與三輪訓練相同口徑）。"""
    return sorted(train["產業"].dropna().unique())


def train_live(df):
    """上線版 LightGBM（t ＝ 2021–2026）：參數與三輪相同；只印出筆數，不寫入 results/。"""
    if not LIVE_DATASET.exists():
        sys.exit(f"找不到 {LIVE_DATASET}，請先執行 python build_dataset.py")
    live = pd.read_parquet(LIVE_DATASET)
    for t_year in LIVE_YEARS:
        years = train_years_for(t_year)
        train = df[df["特徵年"].isin(years) & df[TARGET].notna()]
        t_dates = live.loc[live["特徵年"] == t_year, "特徵日"].unique()
        assert len(t_dates) == 1, f"{t_year} 的特徵日不唯一"
        # 時間一致性（DESIGN 5.5 第 10 條）：訓練標記在 t 的特徵日之前已揭曉
        assert max(years) + HORIZON_YEARS <= t_year, "訓練觀察期超過分析日"
        assert train["標記揭曉日"].notna().all() and (train["標記揭曉日"] <= t_dates[0]).all(), \
            "有訓練標記在分析日之後才揭曉"
        X = to_model_frame(train, industry_categories(train))
        y = train[TARGET].astype(int).to_numpy()
        joblib.dump(train_lightgbm(X, y), MODEL_DIR / f"lgbm_live{t_year}.joblib")
        print(f"上線版 {t_year}：訓練 {years[0]}–{years[-1]}（{len(train)} 筆、正例 {int(y.sum())}）；"
              f"時間一致性檢查通過（無測試結果）")


def main():
    """三輪各訓練兩種模型，存模型、測試年模型分數與彙總表；再訓練個股分析用的上線版模型。"""
    df = load_dataset()
    ranks = rank_by_year(df)   # 每個特徵年內的排名，只用同年特徵
    rows = []
    for round_no, fold in enumerate(FOLDS, start=1):
        train, test, info = split(df, fold)
        categories = industry_categories(train)
        X_train = to_model_frame(train, categories)
        y_train = train[TARGET].astype(int).to_numpy()
        X_test = to_model_frame(test, categories)

        lgbm = train_lightgbm(X_train, y_train)
        logit = train_logistic(to_logit_frame(X_train, ranks), y_train)
        save_model(lgbm, "lgbm", round_no)
        save_model(logit, "logit", round_no)

        scores = test[["代號", "名稱", "產業", "特徵年", "特徵日", "y_raw", TARGET]].copy()
        scores["score_lgbm"] = lgbm.predict_proba(X_test)[:, 1]
        scores["score_logit"] = logit.predict_proba(to_logit_frame(X_test, ranks))[:, 1]
        assert scores[["score_lgbm", "score_logit"]].notna().all().all(), "有測試列沒有模型分數"
        scores.to_parquet(MODEL_DIR / f"scores_round{round_no}.parquet", index=False)

        test_known = test[TARGET].notna()
        unseen = int(X_test["industry"].isna().sum())
        rows.append({
            "輪次": round_no,
            "訓練特徵年": "–".join(map(str, (fold["train"][0], fold["train"][-1])))
                         if len(fold["train"]) > 1 else str(fold["train"][0]),
            "測試特徵年": fold["test"],
            "訓練筆數（y_main 可判定）": len(train),
            "訓練正例": int(y_train.sum()),
            "訓練正例率": round(y_train.mean(), 4),
            "訓練 UNKNOWN 排除": info["訓練 UNKNOWN 排除"],
            "訓練標記最晚揭曉日": info["訓練標記最晚揭曉日"],
            "測試特徵日": info["測試特徵日"],
            "測試樣本（全部評分）": len(test),
            "測試 y_main 可判定": int(test_known.sum()),
            "測試 y_main 正例": int(test.loc[test_known, TARGET].sum()),
            "測試新產業（訓練沒看過）": unseen,
        })
        print(f"第 {round_no} 輪：訓練 {rows[-1]['訓練特徵年']}（{len(train)} 筆、正例 {int(y_train.sum())}）"
              f" → 測試 {fold['test']}（{len(test)} 筆評分）；時間一致性檢查通過")

    summary = pd.DataFrame(rows)
    SUMMARY_CSV.parent.mkdir(exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已輸出 {SUMMARY_CSV.relative_to(BASE)}、models/（模型與測試年模型分數）")
    print(summary.to_string(index=False))
    print()
    train_live(df)


if __name__ == "__main__":
    main()
