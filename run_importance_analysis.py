"""
運行完整的特徵重要性分析
包含所有24個結構化特徵和4種分析方法
"""

from variables_importance import VariableImportanceAnalyzer

if __name__ == "__main__":
    print("=== 開始完整的特徵重要性分析 ===")
    print("包含所有24個結構化特徵")
    print("使用4種分析方法：隨機森林、互信息、F統計量、相關係數")
    print()
    
    # 創建分析器
    analyzer = VariableImportanceAnalyzer()
    
    # 執行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n=== 分析完成！生成的文件包括 ===")
        print("1. feature_importance_methods_explanation.png - 方法解釋圖")
        print("2. feature_importance_random_forest_24.png - 隨機森林重要性")
        print("3. variable_importance_gain_ratio_professional_24.png - 專業版增益比圖")
        print("4. feature_importance_comparison.png - 雙方法對比")
        print("5. comprehensive_feature_importance_comparison.png - 四方法綜合對比")
        print("\n所有24個結構化特徵均已分析並可視化！")
    else:
        print("分析失敗，請檢查數據文件。")