"""
特徵重要性數據分析 - 僅輸出數據結果
"""

from variables_importance import VariableImportanceAnalyzer
import pandas as pd

if __name__ == "__main__":
    print("=== 特徵重要性數據分析 ===")
    print("僅計算和輸出數據結果，不生成圖表\n")
    
    # 創建分析器
    analyzer = VariableImportanceAnalyzer()
    
    # 載入和預處理數據
    print("載入數據...")
    if not analyzer.load_and_preprocess_data():
        print("數據載入失敗")
        exit()
    
    # 計算所有重要性方法
    print("計算隨機森林特徵重要性...")
    rf_df = analyzer.calculate_random_forest_importance()
    
    print("計算互信息...")
    mi_df = analyzer.calculate_mutual_information()
    
    print("計算F統計量重要性...")
    f_df = analyzer.calculate_f_statistic_importance()
    
    print("計算相關係數重要性...")
    corr_df = analyzer.calculate_correlation_importance()
    
    print("\n" + "="*80)
    print("數據集基本信息")
    print("="*80)
    print(f"數據集大小: {analyzer.X.shape}")
    print(f"敗血症病例: {analyzer.y.sum()}/{len(analyzer.y)} ({analyzer.y.mean():.2%})")
    print(f"結構化特徵總數: {len(analyzer.feature_names)}")
    
    # 輸出完整的特徵重要性數據
    print("\n" + "="*80)
    print("隨機森林特徵重要性 (完整24個特徵)")
    print("="*80)
    print(f"{'排名':<4} {'特徵名稱':<15} {'重要性分數':<12}")
    print("-" * 35)
    for i, (_, row) in enumerate(rf_df.iterrows(), 1):
        print(f"{i:<4} {row['Feature']:<15} {row['Importance']:<12.6f}")
    
    print("\n" + "="*80)
    print("互信息特徵重要性 (完整24個特徵)")
    print("="*80)
    print(f"{'排名':<4} {'特徵名稱':<15} {'互信息分數':<12}")
    print("-" * 35)
    for i, (_, row) in enumerate(mi_df.iterrows(), 1):
        print(f"{i:<4} {row['Feature']:<15} {row['Mutual_Information']:<12.6f}")
    
    print("\n" + "="*80)
    print("F統計量特徵重要性 (完整24個特徵)")
    print("="*80)
    print(f"{'排名':<4} {'特徵名稱':<15} {'F統計量':<12} {'p值':<12}")
    print("-" * 50)
    for i, (_, row) in enumerate(f_df.iterrows(), 1):
        p_val_str = f"{row['P_Value']:.2e}" if row['P_Value'] > 0 else "< 1e-16"
        print(f"{i:<4} {row['Feature']:<15} {row['F_Score']:<12.4f} {p_val_str:<12}")
    
    print("\n" + "="*80)
    print("相關係數特徵重要性 (完整24個特徵)")
    print("="*80)
    print(f"{'排名':<4} {'特徵名稱':<15} {'相關係數':<12} {'p值':<12}")
    print("-" * 50)
    for i, (_, row) in enumerate(corr_df.iterrows(), 1):
        p_val_str = f"{row['P_Value']:.2e}" if row['P_Value'] > 0 else "< 1e-16"
        print(f"{i:<4} {row['Feature']:<15} {row['Correlation']:<12.6f} {p_val_str:<12}")
    
    # 創建綜合比較表
    print("\n" + "="*100)
    print("四種方法綜合比較表 (按互信息排序)")
    print("="*100)
    
    # 合併所有結果
    rf_dict = dict(zip(rf_df['Feature'], rf_df['Importance']))
    mi_dict = dict(zip(mi_df['Feature'], mi_df['Mutual_Information']))
    f_dict = dict(zip(f_df['Feature'], f_df['F_Score']))
    corr_dict = dict(zip(corr_df['Feature'], corr_df['Correlation']))
    
    print(f"{'特徵名稱':<15} {'隨機森林':<12} {'互信息':<12} {'F統計量':<12} {'相關係數':<12}")
    print("-" * 75)
    
    for _, row in mi_df.iterrows():  # 按互信息排序
        feature = row['Feature']
        rf_score = rf_dict.get(feature, 0)
        mi_score = mi_dict.get(feature, 0)
        f_score = f_dict.get(feature, 0)
        corr_score = corr_dict.get(feature, 0)
        
        print(f"{feature:<15} {rf_score:<12.6f} {mi_score:<12.6f} {f_score:<12.4f} {corr_score:<12.6f}")
    
    # 輸出CSV格式數據
    print("\n" + "="*80)
    print("CSV格式輸出 (可直接複製到Excel)")
    print("="*80)
    
    # 創建綜合DataFrame
    comprehensive_df = pd.DataFrame({
        'Feature': mi_df['Feature'],
        'Random_Forest_Importance': [rf_dict.get(f, 0) for f in mi_df['Feature']],
        'Mutual_Information': mi_df['Mutual_Information'],
        'F_Statistic': [f_dict.get(f, 0) for f in mi_df['Feature']],
        'Correlation': [corr_dict.get(f, 0) for f in mi_df['Feature']],
        'F_P_Value': [f_df[f_df['Feature']==f]['P_Value'].iloc[0] if f in f_dict else 1 for f in mi_df['Feature']],
        'Corr_P_Value': [corr_df[corr_df['Feature']==f]['P_Value'].iloc[0] if f in corr_dict else 1 for f in mi_df['Feature']]
    })
    
    print("\n特徵重要性完整數據表 (CSV格式):")
    print(comprehensive_df.to_csv(index=False))
    
    # 保存為Excel文件
    with pd.ExcelWriter('特徵重要性分析結果.xlsx', engine='openpyxl') as writer:
        # 工作表1: 綜合比較
        comprehensive_df.to_excel(writer, sheet_name='綜合比較', index=False)
        
        # 工作表2: 隨機森林
        rf_df.to_excel(writer, sheet_name='隨機森林重要性', index=False)
        
        # 工作表3: 互信息
        mi_df.to_excel(writer, sheet_name='互信息重要性', index=False)
        
        # 工作表4: F統計量
        f_df.to_excel(writer, sheet_name='F統計量重要性', index=False)
        
        # 工作表5: 相關係數
        corr_df.to_excel(writer, sheet_name='相關係數重要性', index=False)
        
        # 工作表6: 基本信息
        info_df = pd.DataFrame({
            '項目': ['總樣本數', '敗血症病例數', '敗血症比例(%)', '結構化特徵數'],
            '數值': [len(analyzer.y), analyzer.y.sum(), f"{analyzer.y.mean()*100:.2f}%", len(analyzer.feature_names)]
        })
        info_df.to_excel(writer, sheet_name='數據集信息', index=False)
    
    print("Excel文件已保存為: 特徵重要性分析結果.xlsx")
    print("包含以下工作表:")
    print("- 綜合比較: 所有方法的比較結果")
    print("- 隨機森林重要性: Random Forest結果")
    print("- 互信息重要性: Mutual Information結果")
    print("- F統計量重要性: ANOVA F-test結果")
    print("- 相關係數重要性: Correlation結果")
    print("- 數據集信息: 基本統計信息")
    
    print("\n" + "="*80)
    print("分析完成！所有數據已輸出完畢。")
    print("="*80)