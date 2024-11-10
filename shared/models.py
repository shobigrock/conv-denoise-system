import pandas as pd
import lightgbm as lgb

class LightGBM:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            objective='binary',  # 二値分類を指定
            metric='binary_logloss',  # 評価指標を指定
            boosting_type='gbdt',  # 勾配ブースティングを指定
            num_leaves=31,  # 決定木の葉の数を指定
            learning_rate=0.05,  # 学習率を指定
            verbose=-1,  # ログ出力なし
            feature_fraction=0.9  # 特徴量の割合を指定
        )
        self.is_learned = False
    
    def fit(self, X_train, y_train):
        self.model.fit(
            X_train, y_train,
            # callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        self.is_learned = True
        return
    
    def pred(self, X_test):
        if not self.is_learned:
            print("モデルが学習されていません")
            return
        y_pred = self.model.predict_proba(X_test)[:, 1] # 2クラスのそれぞれの確率が出力されるので後者だけ使う

        return y_pred