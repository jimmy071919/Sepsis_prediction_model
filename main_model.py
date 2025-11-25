"""主模型訓練腳本 - 快速版本

這是 mian_refresh_model.py 的快速版本，主要差異：
- 跳過重新生成嵌入的步驟
- 直接讀取已生成的嵌入檔案進行模型訓練
- 執行10-fold交叉驗證和消融研究

需要的檔案：
- structured_data_embedding/x_train_ax_scaled.npy
- structured_data_embedding/x_test_ax_scaled.npy
- unstructured_data_embedding/x_train_diagnosis_embed.npy
- unstructured_data_embedding/x_test_diagnosis_embed.npy
- unstructured_data_embedding/x_train_chief_embed.npy
- unstructured_data_embedding/x_test_chief_embed.npy
- answer_embedding/y_train.npy
- answer_embedding/y_test.npy
"""

import os
from cross_valibation import CrossValidationTrainer


def check_embedding_files():
    """檢查所需的嵌入檔案是否存在"""
    required_files = [
        "structured_data_embedding/x_train_ax_scaled.npy",
        "structured_data_embedding/x_test_ax_scaled.npy",
        "unstructured_data_embedding/x_train_diagnosis_embed.npy",
        "unstructured_data_embedding/x_test_diagnosis_embed.npy",
        "unstructured_data_embedding/x_train_chief_embed.npy",
        "unstructured_data_embedding/x_test_chief_embed.npy",
        "answer_embedding/y_train.npy",
        "answer_embedding/y_test.npy"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files


def main():
    """主函數 - 直接使用已生成的嵌入檔案進行模型訓練"""
    
    print("=== 敗血症預測模型訓練 (快速版本) ===\n")
    
    # 檢查嵌入檔案是否存在
    print("1. 檢查嵌入檔案...")
    existing_files, missing_files = check_embedding_files()
    
    print(f"   找到 {len(existing_files)} 個嵌入檔案")
    if missing_files:
        print(f"   缺少 {len(missing_files)} 個嵌入檔案:")
        for file_path in missing_files:
            print(f"     - {file_path}")
        print(f"\n   請先運行 mian_refresh_model.py 來生成缺少的嵌入檔案")
        return
    else:
        print("   ✅ 所有必需的嵌入檔案都已存在\n")
    
    # 創建交叉驗證訓練器 (使用新的介面，不需要傳入df)
    print("2. 初始化10-fold交叉驗證訓練器...")
    cv_trainer = CrossValidationTrainer(random_state=42, n_folds=10)
    
    # 執行交叉驗證研究和消融實驗
    print("3. 開始執行消融研究與10-fold交叉驗證...")
    print("   消融研究包含以下特徵組合: a-x, y, z, a-y, a-x,z, a-z")
    cv_trainer.run_cross_validation_study()
    
    # 保存結果
    print("4. 保存結果到Excel文件...")
    cv_trainer.save_results_to_xlsx()
    
    # 保存訓練好的模型
    print("5. 保存訓練好的模型...")
    cv_trainer.save_trained_models()
    
    print("\n=== 敗血症預測模型訓練完成 ===")



if __name__ == "__main__":
    main()
