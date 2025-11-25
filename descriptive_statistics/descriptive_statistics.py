import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加父目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def descriptive_statistics(df):
    """計算a-x欄的描述性統計"""
    # 假設a-x欄是前24欄（需要根據實際情況調整）
    structured_data_cols = [col for col in df.columns if col not in ['diagnosis', 'chief', 'isSepsis']]
    
    stats_dict = {}
    
    for col in structured_data_cols:
        stats_dict[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'count': df[col].count(),
            'missing': df[col].isnull().sum(),
            'missing_ratio': (df[col].isnull().mean())*100
            #缺失直要加比例
        }
    
    # 轉換為DataFrame方便查看
    stats_df = pd.DataFrame(stats_dict).T

    print(f"描述性統計運行完成")

    return stats_df


if __name__ == "__main__":
    # 載入資料
    df = load_data("data\\1141112.xlsx")
    print(descriptive_statistics(df))
