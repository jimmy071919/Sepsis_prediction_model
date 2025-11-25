import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from load_data import load_data
from split_data import split_data

def encode_labels(y_train, y_test):
    """
    將字串標籤（如 'Y', 'N'）轉換為數值標籤（如 1, 0）
    
    Parameters:
    y_train: Series，訓練集標籤
    y_test: Series，測試集標籤
    
    Returns:
    y_train_encoded: ndarray，編碼後的訓練集標籤
    y_test_encoded: ndarray，編碼後的測試集標籤
    label_encoder: LabelEncoder，標籤編碼器
    label_mapping: dict，標籤映射關係
    """
    
    # 檢查標籤的唯一值
    unique_labels = np.unique(np.concatenate([y_train.values, y_test.values]))
    
    # 創建標籤編碼器
    label_encoder = LabelEncoder()
    
    # 合併訓練集和測試集來訓練編碼器（確保一致性）
    all_labels = np.concatenate([y_train.values, y_test.values])
    label_encoder.fit(all_labels)
    
    # 轉換標籤
    y_train_encoded = label_encoder.transform(y_train.values)
    y_test_encoded = label_encoder.transform(y_test.values)
    
    # 創建標籤映射關係
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return y_train_encoded, y_test_encoded, label_encoder, label_mapping

def save_labels(y_train_encoded, y_test_encoded, label_encoder, label_mapping, output_dir="answer_embedding"):
    """
    將編碼後的標籤和編碼器儲存為檔案
    """
    
    # 確保儲存目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 儲存編碼後的標籤
    np.save(os.path.join(output_dir, "y_train.npy"), y_train_encoded)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test_encoded)
    
    # 儲存標籤映射關係
    np.save(os.path.join(output_dir, "label_mapping.npy"), label_mapping)
    
    # 儲存編碼器（用於後續預測）
    import joblib
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

if __name__ == "__main__":
    
    # 載入和分割資料
    df = load_data("data\\1141112.xlsx")
    _, _, y_train, y_test = split_data(df)

    # 編碼標籤
    y_train_encoded, y_test_encoded, label_encoder, label_mapping = encode_labels(y_train, y_test)

    # 儲存編碼後的標籤
    save_labels(y_train_encoded, y_test_encoded, label_encoder, label_mapping)