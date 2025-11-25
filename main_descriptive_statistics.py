import pandas as pd
import numpy as np
from load_data import load_data
from descriptive_statistics.descriptive_statistics import descriptive_statistics
from descriptive_statistics.p_value import calculate_p_value, analyze_all_variables
pd.set_option('display.float_format', lambda x: '%.2f' % x)

def comprehensive_analysis(df):
    """進行完整的描述性統計分析，直接使用descriptive_statistics模組的功能
    
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
    
    # 2. 總體描述性統計 - 直接使用descriptive_statistics模組
    results['overall_stats'] = descriptive_statistics(df)
    
    # 3. 分組描述性統計 - 直接使用descriptive_statistics模組
    sepsis_df = df[df['isSepsis'] == "Y"]
    non_sepsis_df = df[df['isSepsis'] == "N"]
    
    results['sepsis_stats'] = descriptive_statistics(sepsis_df) if len(sepsis_df) > 0 else None
    results['non_sepsis_stats'] = descriptive_statistics(non_sepsis_df) if len(non_sepsis_df) > 0 else None
    
    # 4. 統計顯著性分析 - 直接使用p_value模組
    results['significance_analysis'] = analyze_all_variables(df)
    
    # 5. 整合分組統計結果
    results['group_summary'] = create_group_summary(results)
    
    return results


def create_group_summary(results):
    """整合總體、分組統計和p值結果
    
    Args:
        results: comprehensive_analysis的中間結果
        
    Returns:
        DataFrame: 整合的分組比較表
    """
    overall_stats = results['overall_stats']
    sepsis_stats = results['sepsis_stats']
    non_sepsis_stats = results['non_sepsis_stats']
    significance_analysis = results['significance_analysis']
    
    summary_results = []
    
    for variable in overall_stats.index:
        # 總體統計
        total_mean = overall_stats.loc[variable, 'mean']
        total_std = overall_stats.loc[variable, 'std']
        total_count = overall_stats.loc[variable, 'count']
        
        # 敗血症組統計
        if sepsis_stats is not None and variable in sepsis_stats.index:
            sepsis_mean = sepsis_stats.loc[variable, 'mean']
            sepsis_std = sepsis_stats.loc[variable, 'std']
            sepsis_count = sepsis_stats.loc[variable, 'count']
            sepsis_mean_sd = f"{sepsis_mean:.2f} ({sepsis_std:.2f})" if not pd.isna(sepsis_mean) else "無數據"
        else:
            sepsis_count = 0
            sepsis_mean_sd = "無數據"
        
        # 非敗血症組統計
        if non_sepsis_stats is not None and variable in non_sepsis_stats.index:
            non_sepsis_mean = non_sepsis_stats.loc[variable, 'mean']
            non_sepsis_std = non_sepsis_stats.loc[variable, 'std']
            non_sepsis_count = non_sepsis_stats.loc[variable, 'count']
            non_sepsis_mean_sd = f"{non_sepsis_mean:.2f} ({non_sepsis_std:.2f})" if not pd.isna(non_sepsis_mean) else "無數據"
        else:
            non_sepsis_count = 0
            non_sepsis_mean_sd = "無數據"
        
        # 獲取p值和檢驗類型
        sig_row = significance_analysis[significance_analysis['variable'] == variable]
        if len(sig_row) > 0:
            p_value = sig_row.iloc[0]['p_value']
            test_type = sig_row.iloc[0]['test_type']
        else:
            p_value = np.nan
            test_type = "未檢驗"
        
        summary_results.append({
            'variable': variable,
            'total_mean_sd': f"{total_mean:.2f} ({total_std:.2f})" if not pd.isna(total_mean) else "無數據",
            'sepsis_mean_sd': sepsis_mean_sd,
            'non_sepsis_mean_sd': non_sepsis_mean_sd,
            'p_value': p_value,
            'test_type': test_type,
            'total_n': int(total_count) if not pd.isna(total_count) else 0,
            'sepsis_n': int(sepsis_count) if not pd.isna(sepsis_count) else 0,
            'non_sepsis_n': int(non_sepsis_count) if not pd.isna(non_sepsis_count) else 0
        })
    
    return pd.DataFrame(summary_results)


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
    group_summary = results['group_summary']
    print(f"\n=== 分組描述性統計 ===")
    display_columns = ['variable', 'total_mean_sd', 'sepsis_mean_sd', 'non_sepsis_mean_sd', 'p_value', 'sepsis_n', 'non_sepsis_n']
    print(group_summary[display_columns].to_string(index=False))


def save_results_to_excel(results, filename='result/descriptive_statistics_results.xlsx'):
    """將描述性統計結果保存到Excel文件
    
    Args:
        results: comprehensive_analysis函數返回的結果字典
        filename: 輸出Excel文件名
    """
    import os
    
    # 確保result目錄存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1. 基本信息工作表
        basic_info_df = pd.DataFrame([
            ['總樣本數', results['basic_info']['total_samples']],
            ['敗血症組人數', results['basic_info']['sepsis_distribution']['Y']],
            ['非敗血症組人數', results['basic_info']['sepsis_distribution']['N']],
            ['敗血症比例(%)', round(results['basic_info']['sepsis_distribution']['Y'] / results['basic_info']['total_samples'] * 100, 2)]
        ], columns=['項目', '數值'])
        basic_info_df.to_excel(writer, sheet_name='基本信息', index=False)
        
        # 2. 分組描述性統計工作表
        group_summary = results['group_summary'].copy()
        # 格式化p_value欄位為4位小數
        if 'p_value' in group_summary.columns:
            group_summary['p_value'] = group_summary['p_value'].apply(lambda x: round(float(x), 4) if pd.notna(x) else x)
        group_summary.to_excel(writer, sheet_name='分組描述性統計', index=False)
        
        # 3. 統計顯著性分析工作表
        significance_results = results['significance_analysis'].copy()
        # 格式化p_value欄位為4位小數
        if 'p_value' in significance_results.columns:
            significance_results['p_value'] = significance_results['p_value'].apply(lambda x: round(float(x), 4) if pd.notna(x) else x)
        significance_results.to_excel(writer, sheet_name='統計顯著性分析', index=False)
        
        # 4. 顯著變數清單工作表
        significant_vars = significance_results[significance_results['significant']]
        if len(significant_vars) > 0:
            significant_summary = significant_vars[['variable', 'p_value', 'significance_level', 'test_type']].copy()
            significant_summary = significant_summary.sort_values('p_value')
            # 格式化p_value欄位為4位小數
            if 'p_value' in significant_summary.columns:
                significant_summary['p_value'] = significant_summary['p_value'].apply(lambda x: round(float(x), 4) if pd.notna(x) else x)
            significant_summary.to_excel(writer, sheet_name='顯著變數清單', index=False)
    
    print(f"\n=== Excel結果已保存 ===")
    print(f"文件位置: {filename}")
    print(f"包含工作表: 基本信息、分組描述性統計、統計顯著性分析、顯著變數清單")
    print(f"數值格式: p值已格式化為4位小數的一般數字形式")



if __name__ == "__main__":
    # 載入資料
    df = load_data("data\\1141112.xlsx")
    
    # 進行完整分析 - 直接使用descriptive_statistics模組功能
    results = comprehensive_analysis(df)
    
    # 顯示結果
    display_results(results)
    
    # 保存結果到Excel文件
    save_results_to_excel(results)
    
    print("\n描述性統計分析完成！")
