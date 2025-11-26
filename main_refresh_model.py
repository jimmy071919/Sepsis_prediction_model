"""
完整的模型重新訓練流程
從數據載入、分割、嵌入到交叉驗證的完整pipeline
不包含自定義函數，全部使用其他模組的現有函數
"""

import pandas as pd
import numpy as np
import os
import sys

# 導入所需的模組函數
from load_data import load_data
from split_data import split_data
from embedding.structured_data_embedding import structured_data_process
from cross_valibation import CrossValidationTrainer

def main():
    """完整的模型重新訓練流程"""
    
    print("=== 開始完整的模型重新訓練流程 ===\n")
    
    # 步驟 1: 數據載入 (使用load_data模組)
    print("1. 載入數據...")
    df = load_data("data/1141112.xlsx")
    print(f"   數據載入完成，共 {len(df)} 筆資料\n")
    
    # 步驟 2: 數據分割 (使用split_data模組)  
    print("2. 分割訓練集和測試集...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=1/3, random_state=42)
    print(f"   訓練集: {X_train.shape}, 測試集: {X_test.shape}")
    print(f"   y_train: {y_train.shape}, y_test: {y_test.shape}\n")
    
    # 步驟 3: 結構化數據嵌入處理 (使用structured_data_embedding模組)
    print("3. 處理結構化數據嵌入...")
    x_train_ax_scaled, x_test_ax_scaled, ax_columns, scaler = structured_data_process(X_train, X_test)
    print(f"   結構化數據處理完成，特徵數量: {len(ax_columns)}")
    print(f"   x_train_ax_scaled: {x_train_ax_scaled.shape}")
    print(f"   x_test_ax_scaled: {x_test_ax_scaled.shape}\n")
    
    # 步驟 4: 非結構化數據嵌入處理 (執行unstructured_data_embedding邏輯)
    print("4. 處理非結構化數據嵌入...")
    try:
        # 直接執行非結構化數據嵌入的主要邏輯
        from embedding.unstructured_data_embedding import get_and_save_bert_embedding
        
        # 確保目錄存在
        os.makedirs("unstructured_data_embedding", exist_ok=True)
        
        # 處理 diagnosis 欄位
        if 'diagnosis' in X_train.columns:
            get_and_save_bert_embedding(
                X_train['diagnosis'], 
                save_path="unstructured_data_embedding/x_train_diagnosis_embed.npy"
            )
            get_and_save_bert_embedding(
                X_test['diagnosis'], 
                save_path="unstructured_data_embedding/x_test_diagnosis_embed.npy"
            )
        
        # 處理 chief 欄位
        if 'chief' in X_train.columns:
            get_and_save_bert_embedding(
                X_train['chief'], 
                save_path="unstructured_data_embedding/x_train_chief_embed.npy"
            )
            get_and_save_bert_embedding(
                X_test['chief'], 
                save_path="unstructured_data_embedding/x_test_chief_embed.npy"
            )
        
        print("   非結構化數據處理完成\n")
    except Exception as e:
        print(f"   非結構化數據處理跳過: {str(e)}\n")
    
    # 步驟 5: 答案標籤嵌入處理 (執行answer_embedding邏輯)
    print("5. 處理答案標籤嵌入...")
    try:
        # 直接執行答案標籤嵌入的主要邏輯
        from embedding.answer_embedding import encode_labels
        
        # 確保目錄存在
        os.makedirs("answer_embedding", exist_ok=True)
        
        # 編碼標籤
        y_train_encoded, y_test_encoded, label_encoder, label_mapping = encode_labels(y_train, y_test)
        
        # 保存編碼後的標籤
        np.save("answer_embedding/y_train.npy", y_train_encoded)
        np.save("answer_embedding/y_test.npy", y_test_encoded)
        np.save("answer_embedding/label_mapping.npy", label_mapping)
        
        print("   答案標籤處理完成\n")
    except Exception as e:
        print(f"   答案標籤處理跳過: {str(e)}\n")
    
    # 步驟 6: 交叉驗證訓練和消融研究 (使用cross_valibation模組)
    print("6. 開始10-fold交叉驗證訓練和消融研究...")
    
    cv_trainer = CrossValidationTrainer(random_state=42, n_folds=10)
    
    # 執行完整的交叉驗證研究和消融研究 (6種特徵組合 x 4種模型)
    print("   執行交叉驗證研究和消融研究...")
    print("   消融研究包含以下特徵組合: a-x, y, z, a-y, a-x,z, a-z")
    cv_trainer.run_cross_validation_study()
    print("   交叉驗證和消融研究完成\n")
    
    # 步驟 7: 保存結果
    print("7. 保存交叉驗證結果...")
    cv_trainer.save_results_to_xlsx()
    print("   結果已保存到 result/ 目錄\n")
    
    # 步驟 8: 保存訓練好的模型
    print("8. 保存訓練好的模型...")
    cv_trainer.save_trained_models()
    print("   模型已保存到 models/ 目錄\n")
    
    print("=== 完整的模型重新訓練流程完成 ===")


if __name__ == "__main__":
    main()
