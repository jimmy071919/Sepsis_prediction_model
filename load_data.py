import pandas as pd
import numpy as np

def load_data(filename):
    """載入Excel資料並回傳DataFrame"""
    # 讀取資料
    df = pd.read_excel(filename, na_values=['', ' ', 'N/A', 'NA', 'na', 'n/a', None])
    print(f"已讀取 {filename}，資料總筆數: {len(df)}")

    # 將所有應該是數值型的欄位轉換為數值型態
    # 排除明確的文字欄位
    text_columns = ['diagnosis', 'chief', 'isSepsis']
    numeric_candidates = [col for col in df.columns if col not in text_columns]
    
    # 將數值型候選欄位轉換為數值格式
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"已將 {len(numeric_candidates)} 個欄位轉換為數值格式")

    

    # 定義醫學異常值處理規則
    medical_ranges = {
        'BT': (30, 45),           # 體溫 (攝氏)
        'MAP': (10, 300),         # 平均動脈壓
        'SBP': (40, 300),         # 收縮壓
        'DBP': (20, 200),         # 舒張壓
        'BMI': (10, 100),         # BMI
        'Height': (50, 250),      # 身高(cm)
        'Weight': (10, 300),      # 體重(kg)
        'WBC': (0.1, 300),        # 白血球
        'PLT': (1, 2000),         # 血小板
        'Crea': (0.1, 100),       # 肌酸酐
        'T-Bil': (0.1, 50),       # 總膽紅素
        'Lymph': (0.1, 99),       # 淋巴球百分比
        'Segment': (0.1, 99),     # 嗜中性球百分比
        'PT': (5, 200),           # 凝血酶原時間
        'PCT': (0.01, 500),       # 降鈣素原
        'BOXY': (0, 1000),        # 血氧飽和度
        'Pluse': (30, 250),       # 脈搏
        'LOS': (0, 10000)         # 住院天數
    }

    # 特殊處理：這些變數的 0 值視為異常 (因為有些生理數值不可能是0)
    zero_invalid_vars = ['Weight', 'WBC', 'PLT', 'Crea', 'T-Bil', 'Lymph', 
                         'Segment', 'PT', 'PCT', 'BMI', 'SBP', 'DBP', 'MAP']

    # 為了確保比大小不出錯，先把所有要檢查的欄位都轉為數值 (無法轉的變 NaN)
    # 使用集合運算找出 df 中實際存在的欄位
    cols_to_process = [col for col in medical_ranges.keys() if col in df.columns]
    
    for col in cols_to_process:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 用來儲存統計結果的列表
    stats_log = []

    # 處理每個變數的異常值
    for col in cols_to_process:
        min_val, max_val = medical_ranges[col]
        
        # 1. 記錄原始狀態
        original_notna = df[col].notna().sum() # 原始有效值數量
        original_missing = df[col].isna().sum() # 原始缺失值數量
        
        # 2. 標記異常值的 Mask (True 代表異常，需要變 NaN)
        is_outlier = (df[col] < min_val) | (df[col] > max_val)
        
        if col in zero_invalid_vars:
            # 如果該欄位不允許為 0，則 0 也是異常
            is_outlier = is_outlier | (df[col] == 0)
        
        # 計算這次清除了多少異常值 (只算原本有值但被判定為異常的)
        # 注意：NaN 不會被 < 或 > 判定為 True，所以不用擔心重複算
        outliers_count = is_outlier.sum()

        # 3. 執行清除 (將異常值設為 NaN)
        if outliers_count > 0:
            df.loc[is_outlier, col] = np.nan

        # 4. 記錄最終狀態
        final_missing = df[col].isna().sum()
        
        # 將統計存入列表
        stats_log.append({
            '欄位': col,
            '原始缺失': original_missing,
            '異常剔除': outliers_count,
            '缺失與異常總和': final_missing
        })

    # 將統計結果轉為 DataFrame 以便漂亮顯示
    stats_df = pd.DataFrame(stats_log)
    
    # 計算總結數據
    total_original_missing = stats_df['原始缺失'].sum()
    total_cleaned = stats_df['異常剔除'].sum()
    total_final_missing = stats_df['缺失與異常總和'].sum()

    print("=" * 50)
    print("📊 數據質量處理報告")
    print("=" * 50)
    
    # 設置pandas顯示選項，讓表格更美觀
    pd.set_option('display.width', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.unicode.east_asian_width', True)
    
    # 只顯示有變化的欄位
    filtered_stats = stats_df[(stats_df['原始缺失'] > 0) | (stats_df['異常剔除'] > 0)]
    
    if len(filtered_stats) > 0:
        print("\n📋 詳細處理記錄:")
        print(filtered_stats.to_string(index=True, justify='center'))
    
    print("\n" + "=" * 50)
    print("📈 統計總結:")
    print(f"   原始資料缺失值: {total_original_missing:,} 格")
    print(f"   異常值剔除數量: {total_cleaned:,} 格")
    print(f"   處理後總缺失值: {total_final_missing:,} 格")
    print(f"   驗算: {total_original_missing:,} + {total_cleaned:,} = {total_final_missing:,} ✓")
    print("=" * 50)

    return df

if __name__ == "__main__":
    df = load_data("data\\1141112.xlsx")
    pass