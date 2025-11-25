import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import sys
import os
# 添加父目錄到路徑以便導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from load_data import load_data
from split_data import split_data

# 1. 選擇一個 Clinical BERT 模型
model_name = "emilyalsentzer/Bio_ClinicalBERT" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 建立函數來獲取句子嵌入
def get_and_save_bert_embedding(text_series, save_path=None):
    """
    將文字序列轉換為 BERT 嵌入向量並可選擇性儲存
    
    Parameters:
    text_series: pandas Series，要轉換的文字資料
    save_path: str，儲存路徑（可選）。如果提供，會將結果儲存為 .npy 檔案
    
    Returns:
    numpy.ndarray: 嵌入向量陣列，形狀為 (n_samples, 768)
    """
    embeddings = []
    total_texts = len(text_series)
    
    for i, text in enumerate(text_series):
        if pd.isna(text):
            # 處理缺失值：回傳一個 768 維的零向量
            embeddings.append(np.zeros(model.config.hidden_size))
        else:
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, 
                              max_length=128, padding="max_length")
            with torch.no_grad():
                outputs = model(**inputs)
            # 取 [CLS] token (第一個 token) 的 hidden state
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            embeddings.append(cls_embedding)
    
    embeddings_array = np.array(embeddings)
    
    # 如果有提供儲存路徑，則儲存結果
    if save_path:
        # 確保儲存目錄存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embeddings_array)
        print(f"Embedding 結果已儲存至: {save_path}")
    
    return embeddings_array


if __name__ == "__main__":
    df = load_data("data\\1141112.xlsx")
    x_train, x_test, y_train, y_test = split_data(df)
    #-----------------------------------------------

    # 處理 x_train和x_test的diagnosis 欄位並儲存結果
    x_train_diagnosis_embed = get_and_save_bert_embedding(
        x_train['diagnosis'], 
        save_path="unstructured_data_embedding/x_train_diagnosis_embed.npy"
    )
    x_test_diagnosis_embed = get_and_save_bert_embedding(
        x_test['diagnosis'], 
        save_path="unstructured_data_embedding/x_test_diagnosis_embed.npy"
    )
    
    # 處理 x_train和x_test的chief 欄位並儲存結果
    x_train_chief_embed = get_and_save_bert_embedding(
        x_train['chief'], 
        save_path="unstructured_data_embedding/x_train_chief_embed.npy"
    )
    x_test_chief_embed = get_and_save_bert_embedding(
        x_test['chief'], 
        save_path="unstructured_data_embedding/x_test_chief_embed.npy"
    )
    
    print("所有 embedding 處理完成！")
    
    