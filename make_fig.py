import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import argparse
import os

parser = argparse.ArgumentParser(description='グラフ表示保存')
parser.add_argument('-k', '--kfold', default=10, help='交差検証のフォールド数')
parser.add_argument('-s', '--seed', default=42, help='乱数シード')
parser.add_argument('-i', '--info', default=3, help='情報を持つ特徴量の数, 0~10')

args = parser.parse_args()

df = pd.read_csv(f"data/result_info{args.info}_seed{args.seed}_k{args.kfold}.csv")

# 横軸の値
x = np.arange(0.5, 0.96, 0.01)

# グラフのプロット
plt.figure(figsize=(10, 6))

# 作成するフォルダのパス
new_folder_path = os.path.join("fig", f"info{args.info}_seed{args.seed}_k{args.kfold}")

# フォルダを作成（存在しない場合のみ作成）
os.makedirs(new_folder_path, exist_ok=True)

for idx, row in df.iterrows():
    y = row.drop(["base", "epsilon"]).values
    
    threshold = row["base"]
    # 折れ線グラフを色分けして描画
    y_below_threshold = np.where(y < threshold, y, np.nan)
    y_above_threshold = np.where(y >= threshold, y, np.nan)

    # 閾値の水平線を追加
    plt.axhline(y=threshold, color='gray', linestyle='--', label='Threshold')
    # グラフのプロット（閾値以下の部分）
    plt.plot(x, y_below_threshold, label='ノイズ除去により精度悪化', color='blue')
    # グラフのプロット（閾値以上の部分）
    plt.plot(x, y_above_threshold, label=f'ノイズ除去により精度向上', color='red')

    plt.xlabel('閾値')
    plt.ylabel('ラベルの値が真である確率')
    plt.title(f'ラベルノイズ除去結果_info{args.info}_seed{args.seed}_k{args.kfold}ε{row["epsilon"]}')
    plt.legend()
    plt.savefig(f'fig/info{args.info}_seed{args.seed}_k{args.kfold}/graph_info{args.info}_seed{args.seed}_k{args.kfold}_eps{row["epsilon"]}.png', format='png', dpi=300)
    # plt.show()
    plt.clf()