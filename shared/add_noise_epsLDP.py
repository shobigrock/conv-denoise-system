import pandas as pd
import numpy as np

class epsilon_label_DP:
    def __init__(self):
        self.epsilon_list_0_to_15 = list(range(0, 15))  # これで p = 0.5 ~ 1.0 まで網羅．まとめて処理する時に使う
    

    def random_response_mechanism(df, label_name, cardinality, epsilon):
        """
        DataFrameのラベルに、ε-label-DPを満たす Random Response Mechanism を用いてノイズを付与する機構。
        """
        