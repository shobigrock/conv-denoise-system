import argparse
import numpy as np
import pandas as pd
import random
from math import exp
from sklearn.model_selection import StratifiedKFold
from shared.generate_data import generate_data
from shared.models import LightGBM
from shared.evaluate import evaluate
from tqdm import tqdm
import time


def main():
    """パラメータを受け取って処理する関数"""
    parser = argparse.ArgumentParser(description='教師あり学習+交差検証によるノイズ除去フレームワーク')
    parser.add_argument('-f', '--function', default='LightGBM', help='教師あり学習モデル')
    parser.add_argument('-k', '--kfold', default=10, help='交差検証のフォールド数')
    parser.add_argument('-s', '--seed', default=42, help='乱数シード')
    parser.add_argument('-i', '--info', default=3, help='情報を持つ特徴量の数, 0~10')
    # parser.add_argument('-d', '--delta', default=0.9, help='閾値')  # 関数内で自動検証する

    args = parser.parse_args()
    k = int(args.kfold)
    s = int(args.seed)
    i = int(args.info)
    random.seed(s)

    X, y = generate_data(random_state=s, n_informative=i)
    X_name, y_name = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"], ["cv"]
    X_df = pd.DataFrame(X, columns=X_name)
    y_df = pd.DataFrame(y, columns=y_name)
    df = pd.concat([X_df, y_df], axis=1)

    # 全ての結果を入れる箱
    # cv: 真ラベル、cv_noised: ε-label-DPによるノイズ付与済ラベル、cv_pred: CTR予測値
    all_df = pd.DataFrame([[-1]*14], columns=X_name+y_name+["cv_noised", "cv_pred", "epsilon"]) 
    y_name = ["cv_noised"]

    for epsilon in tqdm(np.arange(0, 10, 0.5), desc="epsilon loop"):
        """
        ノイズ付与率 ε を変化させながら様子を見る
        """
        print(f"epsilon: {epsilon}")
        tmp_df = df.copy()

        # 真ラベルを与える確率
        p = 1 - (1 / (exp(epsilon) + 1))  

        # ノイズ付与
        tmp_df["cv_noised"] = tmp_df["cv"].apply(lambda x: x if random.random() <= p else 1-x)
        
        # て層化k分割交差検証インスタンス
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=s)

        # 予測値を格納する配列を目的変数と同じ長さで初期化
        preds = np.zeros(len(tmp_df))

        # 層化k分割交差検証 (により各行に予測確率割り当て)
        for fold, (train_index, test_index) in enumerate(skf.split(tmp_df[X_name], tmp_df[y_name])):
            # print(f"Fold: {fold+1}")
            train_df = tmp_df.loc[train_index]
            test_df = tmp_df.loc[test_index]
            X_train, y_train = train_df[X_name], train_df[y_name]
            X_test = test_df[X_name]

            lgb = LightGBM()
            lgb.fit(X_train, y_train)
            preds[test_index] = lgb.pred(X_test)
            print("-" * 30)
        
        tmp_df["cv_pred"] = preds
        tmp_df["epsilon"] = tmp_df["cv_pred"].apply(lambda _: epsilon)
        all_df = pd.concat([all_df, tmp_df], axis=0)
    
    # 各ラベルの信頼度
    all_df["reliability"] = all_df.apply(lambda row: abs(row["cv_noised"] - row["cv_pred"]), axis=1)
    all_df = all_df[all_df["cv"] != -1]

    # ノイズ除去ができているか評価
    col_name = ["base"] + [f"thr_0{int(100*x)}" for x in np.arange(0.5, 0.96, 0.01)] + ["epsilon"]
    result_df = pd.DataFrame(columns=col_name)

    for epsilon in np.arange(0, 10, 0.5):
        tmp_df = all_df[all_df["epsilon"] == epsilon]
        ev = evaluate(tmp_df["cv"], tmp_df["cv_noised"], tmp_df["cv_pred"], tmp_df["reliability"])
        result_df = pd.concat([result_df, pd.DataFrame([np.array(ev.various_eval()+[epsilon])], columns=col_name)], axis=0)
    
    # 結果をcsvに保存
    result_df.to_csv(f'data/result_info{i}_seed{s}_k{k}.csv', index=False)


if __name__ == "__main__":
    main()