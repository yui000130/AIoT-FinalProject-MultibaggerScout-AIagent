"""
train.py －模型訓練（規劃中）

依 walk-forward 切分訓練兩種模型，目標為主要標記 y_main（基本面驅動暴漲股）：
  - LightGBM：主模型
  - 邏輯迴歸：Baseline

切分（訓練視窗的 5 年觀察期必須在測試起點前結束）：
  第 1 輪：訓練 2013        → 測試 2018
  第 2 輪：訓練 2013–2014   → 測試 2019
  第 3 輪：訓練 2013–2015   → 測試 2020

輸入：data/dataset.parquet
輸出：models/{model}_round{n}.*
"""

FOLDS = [
    {"train": [2013], "test": 2018},
    {"train": [2013, 2014], "test": 2019},
    {"train": [2013, 2014, 2015], "test": 2020},
]
SEED = 42


def load_dataset():
    """讀取 data/dataset.parquet。"""
    # TODO
    raise NotImplementedError


def split(df, fold):
    """依 fold 回傳訓練集與測試集。"""
    # TODO: 檢查訓練年 + 5 ≤ 測試年
    raise NotImplementedError


def train_lightgbm(X, y):
    """訓練 LightGBM；主要標記正例少於 5%，需處理類別不平衡。"""
    # TODO
    raise NotImplementedError


def train_logistic(X, y):
    """訓練邏輯迴歸 Baseline（含標準化與缺值處理）。"""
    # TODO
    raise NotImplementedError


def save_model(model, name, round_no):
    """把模型存到 models/。"""
    # TODO
    raise NotImplementedError


def main():
    """三輪各訓練兩種模型並存檔。"""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
