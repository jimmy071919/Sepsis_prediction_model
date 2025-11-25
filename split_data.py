import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from load_data import load_data

def split_data(df, test_size=1/3, random_state=42):
    """分割訓練集和測試集 (2:1比例)"""
    # AA欄為目標變數[y]
    X = df.drop('isSepsis', axis=1)
    y = df['isSepsis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # X_train : 訓練集特徵
    # y_train : 訓練集目標變數

    # X_test : 測試集特徵
    # y_test : 測試集目標變數
    
    print(f"2.資料分割完成")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    df = load_data("data\\1141112.xlsx")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")