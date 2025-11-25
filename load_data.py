import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(filename):
    """載入Excel資料並回傳DataFrame"""
    df = pd.read_excel(filename,na_values=['', ' ', 'N/A', 'NA', 'na', 'n/a',None])
    print(f"1.有讀到{filename}，資料總筆數: {len(df)}")

    columns_to_coerce = ['WBC', 'T-Bil', 'Lymph', 'PT', 'Hs-CRP', 'PCT']

    #進行資料轉換
    for col in columns_to_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


if __name__ == "__main__":
    df = load_data("data\\1141112.xlsx")
    df.info()