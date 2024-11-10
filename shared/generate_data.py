import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from math import exp
import random

def generate_data(random_state=42, n_informative=3):
    """
    分類のためのデータセットを生成して返す。
    """
    X, y = make_classification(
        random_state = random_state,
        n_samples=30000,  # データ数 (train+test)
        n_features=10,  # 特徴量数
        n_redundant=2,  # 多重共線を起こす特徴量数
        n_informative=n_informative,  # yと相関の強い特徴量数
        n_clusters_per_class=4,  # 目的変数のクラスター数
        n_classes=2,  # クラス数 (今回二値分類なので2)
        weights=[0.95, 0.05],  # 不均衡データ (正例は5%程度)
        flip_y=0  # ノイズ付与率_同じ特徴量での違いを見たいので、ここでは付与しない
    )

    return X, y


def add_noise_to_y_epsDP(df, y_base_name, y_noised_name, epsilon):
    """
    Randomized Response Mechanism によるノイズ付与を行う関数。
    """
    p = 1 - (1 / (exp(epsilon) + 1))  # 真ラベルを与える確率
    df[y_noised_name] = df[y_base_name].apply(lambda x: x if random.random() <= p else 1-x)

    return df