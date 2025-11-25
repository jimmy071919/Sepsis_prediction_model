import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data
from descriptive_statistics.descriptive_statistics import descriptive_statistics
from descriptive_statistics.p_value import calculate_p_value, analyze_all_variables
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def debug_missing_values(df):
    """調試關鍵變數的缺失值分布
    
    Args:
        df: DataFrame包含數據
    """
    key_cols = ['Height', 'Weight', 'BMI', 'AGE', 'SEX']
    
    print("=== 關鍵變數缺失值分析 ===")
    print(f"總樣本數: {len(df)}")
    print(f"Sepsis組: {len(df[df['isSepsis'] == 'Y'])} 人")
    print(f"Non-sepsis組: {len(df[df['isSepsis'] == 'N'])} 人")
    print()
    
    for col in key_cols:
        if col not in df.columns:
            continue
            
        total_valid = df[col].notna().sum()
        sepsis_valid = df[(df['isSepsis'] == "Y") & (df[col].notna())].shape[0]
        non_sepsis_valid = df[(df['isSepsis'] == "N") & (df[col].notna())].shape[0]
        
        print(f"{col:10} | 有效數據: {total_valid:4} | Sepsis: {sepsis_valid:3} | Non-sepsis: {non_sepsis_valid:4}")
    
    # 檢查Height的具體情況
    if 'Height' in df.columns:
        print(f"\nHeight詳細分析:")
        print(f"Height總缺失: {df['Height'].isna().sum()}")
        print(f"Sepsis組Height缺失: {df[(df['isSepsis'] == 'Y') & (df['Height'].isna())].shape[0]}")
        print(f"Non-sepsis組Height缺失: {df[(df['isSepsis'] == 'N') & (df['Height'].isna())].shape[0]}")


def comprehensive_analysis(df):
    """進行完整的描述性統計分析，包含分組統計和顯著性檢驗
    
    Args:
        df: DataFrame包含數據
        
    Returns:
        dict: 包含各種分析結果的字典
    """
    results = {}
    
    # 1. 基本數據信息
    results['basic_info'] = {
        'total_samples': len(df),
        'sepsis_distribution': df['isSepsis'].value_counts().to_dict(),
        'sepsis_unique_values': sorted(df['isSepsis'].unique())
    }
    
    # 2. 總體描述性統計
    results['overall_stats'] = descriptive_statistics(df)
    
    # 3. 分組描述性統計
    results['group_stats'] = group_descriptive_statistics(df)
    
    # 4. 統計顯著性分析
    results['significance_analysis'] = analyze_all_variables(df)
    
    return results


def group_descriptive_statistics(df):
    """依據isSepsis分組計算描述性統計
    
    Args:
        df: DataFrame包含數據
        
    Returns:
        summary_df: 包含分組統計和p-value的DataFrame
    """
    # 使用現有的描述性統計功能
    total_stats = descriptive_statistics(df)
    
    # 分別計算各組的統計
    sepsis_df = df[df['isSepsis'] == "Y"]
    non_sepsis_df = df[df['isSepsis'] == "N"]
    
    sepsis_stats = descriptive_statistics(sepsis_df) if len(sepsis_df) > 0 else None
    non_sepsis_stats = descriptive_statistics(non_sepsis_df) if len(non_sepsis_df) > 0 else None
    
    # 獲取結構化變數列表
    structured_cols = [col for col in df.columns if col not in ['diagnosis', 'chief', 'isSepsis']]
    
    summary_results = []
    
    for col in structured_cols:
        if col not in total_stats.index:
            continue
            
        # 獲取總體統計
        total_mean = total_stats.loc[col, 'mean']
        total_std = total_stats.loc[col, 'std']
        total_count = total_stats.loc[col, 'count']
        
        # 獲取敗血症組統計
        if sepsis_stats is not None and col in sepsis_stats.index:
            sepsis_mean = sepsis_stats.loc[col, 'mean']
            sepsis_std = sepsis_stats.loc[col, 'std']
            sepsis_count = sepsis_stats.loc[col, 'count']
            sepsis_mean_sd = f"{sepsis_mean:.2f} ({sepsis_std:.2f})" if not pd.isna(sepsis_mean) else "無數據"
        else:
            sepsis_count = 0
            sepsis_mean_sd = "無數據"
        
        # 獲取非敗血症組統計
        if non_sepsis_stats is not None and col in non_sepsis_stats.index:
            non_sepsis_mean = non_sepsis_stats.loc[col, 'mean']
            non_sepsis_std = non_sepsis_stats.loc[col, 'std']
            non_sepsis_count = non_sepsis_stats.loc[col, 'count']
            non_sepsis_mean_sd = f"{non_sepsis_mean:.2f} ({non_sepsis_std:.2f})" if not pd.isna(non_sepsis_mean) else "無數據"
        else:
            non_sepsis_count = 0
            non_sepsis_mean_sd = "無數據"
        
        # 計算p-value
        p_value, test_type = calculate_p_value(df, col)
        
        summary_results.append({
            'variable': col,
            'total_mean_sd': f"{total_mean:.2f} ({total_std:.2f})" if not pd.isna(total_mean) else "無數據",
            'sepsis_mean_sd': sepsis_mean_sd,
            'non_sepsis_mean_sd': non_sepsis_mean_sd,
            'p_value': p_value,
            'test_type': test_type,
            'total_n': int(total_count) if not pd.isna(total_count) else 0,
            'sepsis_n': int(sepsis_count) if not pd.isna(sepsis_count) else 0,
            'non_sepsis_n': int(non_sepsis_count) if not pd.isna(non_sepsis_count) else 0
        })
    
    summary_df = pd.DataFrame(summary_results)
    return summary_df


def display_results(results):
    """顯示分析結果
    
    Args:
        results: comprehensive_analysis返回的結果字典
    """
    # 基本信息
    basic_info = results['basic_info']
    print("=== 數據基本信息 ===")
    print(f"總樣本數: {basic_info['total_samples']}")
    print(f"isSepsis分布: {basic_info['sepsis_distribution']}")
    print(f"isSepsis唯一值: {basic_info['sepsis_unique_values']}")
    
    # 統計顯著性分析
    significance_results = results['significance_analysis']
    significant_vars = significance_results[significance_results['significant']]
    
    print(f"\n=== 統計顯著性分析 ===")
    print(f"共有 {len(significant_vars)} 個變數達到統計顯著 (p < 0.05)")
    
    if len(significant_vars) > 0:
        print(f"\n顯著變數清單:")
        print(significant_vars[['variable', 'p_value', 'significance_level', 'test_type']].to_string(index=False))
    
    # 分組描述性統計
    group_stats = results['group_stats']
    print(f"\n=== 分組描述性統計 ===")
    display_columns = ['variable', 'total_mean_sd', 'sepsis_mean_sd', 'non_sepsis_mean_sd', 'p_value', 'sepsis_n', 'non_sepsis_n']
    print(group_stats[display_columns].to_string(index=False))



if __name__ == "__main__":
    # 載入資料
    df = load_data("data\\1141112.xlsx")
    
    # 先進行缺失值調試分析
    debug_missing_values(df)
    
    print("\n" + "="*60 + "\n")
    
    # 進行完整分析
    results = comprehensive_analysis(df)
    
    # 顯示結果
    display_results(results)
    
    # 可選：儲存結果到檔案
    # results['group_stats'].to_csv('group_statistics.csv', index=False, encoding='utf-8-sig')
    # results['significance_analysis'].to_csv('significance_analysis.csv', index=False, encoding='utf-8-sig')
