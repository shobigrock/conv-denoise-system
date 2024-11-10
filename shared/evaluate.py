import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class evaluate:
    """
    閾値を0.90から0.95まで変えて、ノイズ除去精度accuracyを測定
    """
    def __init__(self, y_true, y_test, y_pred, reliability):
        self.evaldata = pd.concat([y_true, y_test, y_pred, reliability], axis=1)

        # 閾値で補完ラベルにしたもの
        for threshold in np.arange(0.5, 0.96, 0.01):
            self.evaldata[f"cv_pred_0{int(100*threshold)}"] = self.evaldata.apply(lambda row: row["cv_noised"] if row["reliability"] < threshold else 1-row["cv_noised"], axis=1)


    def _accuracy(self):
        accuracies = [round(accuracy_score(self.evaldata["cv"], self.evaldata["cv_noised"]), 4)]
        for threshold in np.arange(0.5, 0.96, 0.01):
            accuracies.append(round(accuracy_score(self.evaldata["cv"], self.evaldata[f"cv_pred_0{int(100*threshold)}"]), 4))

        return accuracies
    
    def various_eval(self):  # accuracy以外の要素が入ったらここに追加する。
        return self._accuracy()
    