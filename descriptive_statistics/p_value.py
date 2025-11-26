import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
import sys
import os

# 添加父目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import load_data
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def calculate_p_value(df, column):
    """計算指定欄位與isSepsis之間的p-value
    
    Args:
        df: DataFrame包含數據
        column: 要檢驗的欄位名稱
        
    Returns:
        p_value: 統計檢驗的p值
        test_type: 使用的檢驗方法
    """
    # 判斷是否為類別變數
    # 1. 如果唯一值 <= 10，視為類別變數
    # 2. 特定變數如SEX等已知的類別變數
    categorical_vars = ['SEX']  # 明確指定的類別變數
    unique_count = df[column].nunique()
    
    is_categorical = (column in categorical_vars) or (unique_count <= 10 and unique_count >= 2)
    
    if not is_categorical and df[column].dtype in [np.float64, np.int64]:
        # 連續變數使用t檢定
        group1 = df[df['isSepsis'] == "Y"][column].dropna()
        group2 = df[df['isSepsis'] == "N"][column].dropna()
        
        if len(group1) == 0 or len(group2) == 0:
            return np.nan, 't-test'
            
        stat, p_value = ttest_ind(group1, group2, equal_var=False)
        return p_value, 't-test'
    else:
        # 類別變數使用卡方檢定
        contingency_table = pd.crosstab(df[column], df['isSepsis'])
        
        # 檢查是否有足夠的樣本進行卡方檢定
        if contingency_table.min().min() < 5:
            if contingency_table.shape == (2, 2):
                stat, p_value = fisher_exact(contingency_table)
                return p_value, 'fisher-exact'
        
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        return p_value, 'chi-square'


def analyze_all_variables(df, alpha=0.05):
    """計算所有結構化變數與isSepsis的p-value
    
    Args:
        df: DataFrame包含數據
        alpha: 顯著性水準，預設0.05
        
    Returns:
        result_df: 包含所有變數p-value結果的DataFrame
    """
    # 取得結構化數據欄位（排除文字欄位和目標變數）
    structured_cols = [col for col in df.columns if col not in ['diagnosis', 'chief', 'isSepsis']]
    
    results = []
    
    for col in structured_cols:
        # 檢查該欄位是否有有效數據
        if df[col].notna().sum() == 0:
            continue
            
        p_value, test_type = calculate_p_value(df, col)
        
        # 判斷是否顯著
        is_significant = p_value < alpha if not np.isnan(p_value) else False
        
        results.append({
            'variable': col,
            'p_value': p_value,
            'test_type': test_type,
            'significant': is_significant,
            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    result_df = pd.DataFrame(results)
    
    # 按p-value排序
    result_df = result_df.sort_values('p_value', na_position='last')
    
    return result_df


if __name__ == "__main__":
    # 載入資料
    df = load_data("data\\1141112.xlsx")
    
    # 計算所有變數的p-value
    p_value_results = analyze_all_variables(df)
    
    # 顯示顯著變數
    significant_vars = p_value_results[p_value_results['significant']]
    print(f"=== 統計顯著性分析結果 ===")
    print(f"共有 {len(significant_vars)} 個變數達到統計顯著 (p < 0.05)")
    
    if len(significant_vars) > 0:
        print(f"\n顯著變數清單:")
        print(significant_vars[['variable', 'p_value', 'significance_level', 'test_type']].to_string(index=False))
    
    print(f"\n完整p-value分析結果:")
    print(p_value_results.to_string(index=False))