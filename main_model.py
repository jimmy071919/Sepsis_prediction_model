"""主模型訓練腳本

負責執行所有模型相關的函數，包括：
- 10-fold 交叉驗證
- 消融研究
- 模型評估和結果保存
"""

from cross_valibation import CrossValidationTrainer
from load_data import load_data


def main():
    """主函數 - 負責執行所有模型相關任務"""
    print("=== 敗血症預測模型訓練開始 ===")
    
    # 載入數據
    print("載入數據...")
    df = load_data('data/1141112.xlsx')
    
    # 創建交叉驗證訓練器
    print("初始化10-fold交叉驗證訓練器...")
    cv_trainer = CrossValidationTrainer(df, random_state=42, n_folds=10)
    
    # 執行交叉驗證研究
    print("開始執行消融研究與10-fold交叉驗證...")
    cv_trainer.run_cross_validation_study()
    
    # 保存結果
    print("保存結果到CSV文件...")
    cv_trainer.save_results_to_csv()
    
    print("\n=== 敗血症預測模型訓練完成 ===")
    print("所有結果已保存到 result/ 目錄中")


if __name__ == "__main__":
    main()
