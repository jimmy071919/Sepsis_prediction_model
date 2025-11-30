import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from scipy.stats import chi2_contingency, fisher_exact

# 添加父目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_data
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def sex_statics(df):
    """計算性別欄位的描述性統計（人數和百分比）"""
    
    # 建立交叉表
    contingency_table = pd.crosstab(df['SEX'], df['isSepsis'], margins=True)
    
    # 計算百分比（列百分比 - 每個性別中敗血症的比例）
    contingency_pct = pd.crosstab(df['SEX'], df['isSepsis'], normalize='index') * 100
    
    # 建立詳細統計表
    print("=" * 80)
    print("性別統計分析")
    print("=" * 80)
    print("\n1. 人數統計:")
    print(contingency_table)
    
    print("\n2. 百分比統計 (各性別內的敗血症比例):")
    print(contingency_pct.round(2))
    
    # 計算各組內的性別分布
    print("\n3. 各組內的性別分布:")
    sepsis_yes = df[df['isSepsis'] == 'Y']
    sepsis_no = df[df['isSepsis'] == 'N']
    
    print(f"\n敗血症組 (N={len(sepsis_yes)}):")
    sex_dist_yes = sepsis_yes['SEX'].value_counts()
    for sex, count in sex_dist_yes.items():
        pct = (count / len(sepsis_yes)) * 100
        print(f"  {sex}: {count} 人 ({pct:.2f}%)")
    
    print(f"\n非敗血症組 (N={len(sepsis_no)}):")
    sex_dist_no = sepsis_no['SEX'].value_counts()
    for sex, count in sex_dist_no.items():
        pct = (count / len(sepsis_no)) * 100
        print(f"  {sex}: {count} 人 ({pct:.2f}%)")
    
    # 統計檢驗
    print("\n4. 統計檢驗 (卡方檢驗或Fisher精確檢驗):")
    contingency_2x2 = pd.crosstab(df['SEX'], df['isSepsis'])
    
    # 檢查是否適合用卡方檢驗
    if contingency_2x2.min().min() < 5:
        if contingency_2x2.shape == (2, 2):
            stat, p_value = fisher_exact(contingency_2x2)
            test_method = "Fisher's Exact Test"
        else:
            chi2, p_value, dof, expected = chi2_contingency(contingency_2x2)
            stat = chi2
            test_method = "Chi-square Test (警告: 期望頻數<5)"
    else:
        chi2, p_value, dof, expected = chi2_contingency(contingency_2x2)
        stat = chi2
        test_method = "Chi-square Test"
    
    print(f"  檢驗方法: {test_method}")
    print(f"  檢驗統計量: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.001:
        significance = "*** (極顯著)"
    elif p_value < 0.01:
        significance = "** (高度顯著)"
    elif p_value < 0.05:
        significance = "* (顯著)"
    else:
        significance = "ns (不顯著)"
    
    print(f"  顯著性: {significance}")
    
    print("\n" + "=" * 80)
    
    # 返回結果字典
    results = {
        'contingency_table': contingency_table,
        'contingency_pct': contingency_pct,
        'p_value': p_value,
        'test_method': test_method,
        'significance': significance
    }
    
    return results


if __name__ == "__main__":
    # 載入資料
    # 因為此腳本在data資料夾內，所以直接使用檔案名稱
    df = load_data("1141112.xlsx")
    sex_statics(df)
