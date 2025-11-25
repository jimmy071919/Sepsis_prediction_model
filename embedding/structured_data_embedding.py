import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

# 添加父目錄到路徑以便導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data
from split_data import split_data

def structured_data_process(x_train, x_test, output_dir="data"):

    
    # 找出 a-x 欄位 (所有非文字的欄位)
    ax_columns = [col for col in x_train.columns if col not in ['diagnosis', 'chief']]

    x_train_ax = x_train[ax_columns].values
    x_test_ax = x_test[ax_columns].values

    # 標準化
    scaler = StandardScaler()
    x_train_ax_scaled = scaler.fit_transform(x_train_ax)
    x_test_ax_scaled = scaler.transform(x_test_ax)


    # 儲存 a-x 欄位的結果
    np.save(os.path.join("structured_data_embedding/x_train_ax_scaled.npy"), x_train_ax_scaled)
    np.save(os.path.join("structured_data_embedding/x_test_ax_scaled.npy"), x_test_ax_scaled)
    
    # 儲存欄位名稱 "structured_data_embedding/ax_columns.txt"), "w", encoding="utf-8") as f:
    with open("structured_data_embedding/ax_columns.txt", "w", encoding="utf-8") as f:
        for col in ax_columns:
            f.write(f"{col}\n")
    
    # a-x 結構化資料處理並儲存完成
    
    return x_train_ax_scaled, x_test_ax_scaled, ax_columns, scaler

if __name__ == "__main__":
    # 載入和分割資料
    df = load_data("data\\1141112.xlsx")
    x_train, x_test, y_train, y_test = split_data(df)
    
    # 處理結構化資料
    x_train_ax_scaled, x_test_ax_scaled, ax_columns, scaler = structured_data_process(x_train, x_test)