import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from load_data import load_data
from split_data import split_data

def structured_data_process(x_train, x_test, output_dir="data"):
    """
    處理結構化資料 (a-x 欄) - 標準化
    
    Parameters:
    x_train: DataFrame，訓練集特徵
    x_test: DataFrame，測試集特徵
    output_dir: str，輸出目錄
    
    Returns:
    x_train_ax_scaled: ndarray，處理後的訓練集
    x_test_ax_scaled: ndarray，處理後的測試集
    ax_columns: list，a-x欄位名稱
    scaler: StandardScaler，標準化器
    """
    
    # 找出 a-x 欄位 (所有非文字的欄位)
    ax_columns = [col for col in x_train.columns if col not in ['diagnosis', 'chief']]

    # 直接使用原始資料（缺失值已在 load_data.py 處理）
    x_train_ax = x_train[ax_columns].values
    x_test_ax = x_test[ax_columns].values

    # 標準化
    scaler = StandardScaler()
    x_train_ax_scaled = scaler.fit_transform(x_train_ax)
    x_test_ax_scaled = scaler.transform(x_test_ax)

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 儲存 a-x 欄位的結果
    np.save(os.path.join(output_dir, "x_train_ax_scaled.npy"), x_train_ax_scaled)
    np.save(os.path.join(output_dir, "x_test_ax_scaled.npy"), x_test_ax_scaled)
    
    # 儲存欄位名稱
    with open(os.path.join(output_dir, "ax_columns.txt"), "w", encoding="utf-8") as f:
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