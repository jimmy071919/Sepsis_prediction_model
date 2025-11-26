# 敗血症預測模型 - 資料載入與分割模組筆記

## 檔案：`laod_data.py`

### 模組功能
負責載入Excel資料並進行訓練集/測試集分割

### 程式碼結構

```python
# 導入必要套件
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 資料載入函數
def load_data(filename):
    """載入Excel資料並回傳DataFrame"""
    df = pd.read_excel(filename, na_values=['', ' ', 'N/A', 'NA', 'na', 'n/a', None])
    print(f"有讀到{filename}，資料總筆數: {len(df)}")
    
    # 指定需要轉換為數值型的欄位
    columns_to_coerce = ['WBC', 'T-Bil', 'Lymph', 'PT', 'Hs-CRP', 'PCT']
    
    # 進行資料轉換
    for col in columns_to_coerce:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# 2. 資料分割函數
def split_data(df, test_size=0.33, random_state=42):
    """分割訓練集和測試集 (2:1比例)"""
    # 假設AA欄是目標變數
    X = df.drop('AA', axis=1)  # 特徵變數（所有欄位除了AA）
    y = df['AA']               # 目標變數（是否有敗血症）
    
    # 使用train_test_split進行分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,      # 測試集比例33%
        random_state=random_state, # 隨機種子確保可重現
        stratify=y                # 保持敗血症比例一致
    )
    
    print(f"訓練集大小: {len(X_train)}")
    print(f"測試集大小: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

# 3. 主要執行區塊
if __name__ == "__main__":
    df = pd.read_excel("data\\1141112.xlsx")
    df.info()  # 顯示資料框架結構資訊
```

## 資料處理細節

### 缺失值處理
使用 `na_values` 參數處理各種可能的缺失值表示：
```python
na_values=['', ' ', 'N/A', 'NA', 'na', 'n/a', None]
```
- **空字串**：`''`
- **空格**：`' '`
- **標準缺失值**：`'N/A'`, `'NA'`
- **小寫變體**：`'na'`, `'n/a'`
- **Python None**：`None`

### 資料型別轉換
某些醫療檢驗數值在Excel中被識別為文字型別，需要轉換：
```python
columns_to_coerce = ['WBC', 'T-Bil', 'Lymph', 'PT', 'Hs-CRP', 'PCT']
for col in columns_to_coerce:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
```
- **目的**：將object型別轉換為float64數值型別
- **`errors='coerce'`**：無法轉換的值會變成NaN，避免程式錯誤
- **檢查欄位存在**：使用`if col in df.columns`確保安全性

### 實際資料欄位分析
根據 `df.info()` 結果：
- **總筆數**：1886筆
- **數值型欄位**：15個 float64 + 3個 int64
- **文字型欄位**：9個 object
- **主要缺失值欄位**：
  - `PCT` (347筆，缺失率82%)
  - `Segment` (385筆，缺失率80%) 
  - `PT` (802筆，缺失率57%)
  - `T-Bil` (824筆，缺失率56%)

### 目標變數確認
- **實際目標欄位**：`isSepsis`（不是AA欄）
- **診斷相關欄位**：`diagnosis`、`chief`需要embedding處理

### 資料顯示格式設定
```python
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```
- 將浮點數顯示限制為小數點後兩位
- 讓描述性統計結果更易讀，避免科學記號
- 適合醫療數據的精度要求
```

## 重要概念說明

### 資料分割邏輯
| 變數 | 內容 | 用途 |
|------|------|------|
| `X_train` | 訓練集特徵資料 | 讓模型學習特徵與結果的關係 |
| `y_train` | 訓練集目標變數 | 訓練時的正確答案 |
| `X_test` | 測試集特徵資料 | 測試模型預測能力 |
| `y_test` | 測試集目標變數 | 測試時的正確答案（用來評估準確度） |

### 資料分割結果
執行 `split_data()` 後的實際結果：
```
X_train shape: (1257, 26), y_train shape: (1257,)
X_test shape: (629, 26), y_test shape: (629,)
```

**Shape 含義解釋**：
- **`X_train.shape = (1257, 26)`**：
  - 1257筆訓練資料
  - 26個特徵欄位（移除 `isSepsis` 目標變數後）
- **`y_train.shape = (1257,)`**：
  - 1257個對應的目標值（一維陣列）
- **`X_test.shape = (629, 26)`**：
  - 629筆測試資料
  - 同樣26個特徵欄位
- **`y_test.shape = (629,)`**：
  - 629個對應的目標值

**驗證結果**：
- 總筆數：1257 + 629 = 1886 ✓
- 訓練/測試比例：1257:629 ≈ 2:1 ✓
- 特徵數量：26個（原27欄位 - 1個目標變數） ✓

### 重要參數
- `test_size=0.33`：測試集佔33%
- `random_state=42`：固定隨機種子，確保每次分割結果一致
- `stratify=y`：確保訓練集和測試集中敗血症的比例相同

## 資料欄位假設
根據專案文件和實際資料：
- **結構化數據欄位**：AGE, SEX, Height, Weight, BMI, LOS, BT, Pluse, SBP, DBP, RR, MAP, BOXY, WBC, Crea, PLT, T-Bil, Lymph, Segment, PT, Hs-CRP, PCT, Urine_Volumn, GCS
- **diagnosis欄位**：需要embedding處理的診斷文字
- **chief欄位**：需要embedding處理的主訴文字  
- **目標變數**：`isSepsis`（是否有敗血症）

## 後續使用方式
```python
# 載入資料
df = load_data("data\\1141112.xlsx")

# 分割資料
X_train, X_test, y_train, y_test = split_data(df)

# 用於模型訓練
model.fit(X_train, y_train)

# 用於模型評估
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

## 注意事項
1. **目標變數**：實際目標欄位是 `isSepsis`，不是 `AA` 欄
2. 檔案路徑使用反斜線（Windows格式）
3. 需要確認實際欄位名稱是否與假設一致
4. 後續需要處理文字欄位（`diagnosis`、`chief`）的embedding
5. **缺失值處理**：已設定多種缺失值識別格式，確保資料清潔
6. **資料型別轉換**：醫療檢驗數值需要從object轉換為數值型別
7. **高缺失率欄位**：PCT、Segment等欄位缺失率超過80%，需要評估是否使用

## 下一步
- ✅ 建立描述性統計分析
- ✅ 實作文字欄位embedding
- 建立機器學習模型（DT、SVM、ANN、RF、LR、NN、SGD）
- 進行消融研究

---

# 文字欄位 Embedding 處理筆記

## 檔案：`data_embedding.py`

### 模型選擇
使用 Clinical BERT 模型專門處理醫療文字：
```python
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### BERT Embedding 函數（更新版）
```python
def get_and_save_bert_embedding(text_series, save_path=None):
    """
    將文字序列轉換為 BERT 嵌入向量並可選擇性儲存
    
    Parameters:
    text_series: pandas Series，要轉換的文字資料
    save_path: str，儲存路徑（可選）。如果提供，會將結果儲存為 .npy 檔案
    
    Returns:
    numpy.ndarray: 嵌入向量陣列，形狀為 (n_samples, 768)
    """
```

**新增功能特點**：

#### 1. 可選擇性儲存功能
```python
# 有儲存路徑 - 會儲存結果到檔案
embeddings = get_and_save_bert_embedding(text_series, save_path="embeddings/result.npy")

# 無儲存路徑 - 只回傳結果
embeddings = get_and_save_bert_embedding(text_series)
```

#### 2. 進度顯示機制
```python
# 每處理 100 筆或處理完成時顯示進度
if (i + 1) % 100 == 0 or (i + 1) == total_texts:
    print(f"已處理: {i + 1}/{total_texts} ({((i + 1)/total_texts)*100:.1f}%)")
```
**輸出範例**：
```
開始處理 1257 筆文字資料...
已處理: 100/1257 (8.0%)
已處理: 200/1257 (15.9%)
已處理: 300/1257 (23.9%)
...
已處理: 1257/1257 (100.0%)
文字欄位 embedding 完成！形狀: (1257, 768)
```

#### 3. 詳細資訊輸出
- **開始處理**：顯示總文字數量
- **完成確認**：顯示最終 embedding 陣列形狀
- **儲存確認**：顯示檔案儲存路徑

#### 4. 組織化檔案管理
所有 embedding 結果統一儲存在 `embeddings/` 資料夾：
```
embeddings/
├── X_train_diagnosis_embed.npy  # 訓練集診斷文字嵌入 (1257, 768)
├── X_test_diagnosis_embed.npy   # 測試集診斷文字嵌入 (629, 768)
├── X_train_chief_embed.npy      # 訓練集主訴文字嵌入 (1257, 768)
└── X_test_chief_embed.npy       # 測試集主訴文字嵌入 (629, 768)
```

### 實際執行流程
```python
if __name__ == "__main__":
    df = load_data("data\\1141112.xlsx")
    X_train, X_test, y_train, y_test = split_data(df)

    # 處理 diagnosis 欄位並儲存結果
    print("正在處理 diagnosis 欄位...")
    X_train_diagnosis_embed = get_and_save_bert_embedding(
        X_train['diagnosis'], 
        save_path="embeddings/X_train_diagnosis_embed.npy"
    )
    X_test_diagnosis_embed = get_and_save_bert_embedding(
        X_test['diagnosis'], 
        save_path="embeddings/X_test_diagnosis_embed.npy"
    )
    
    # 處理 chief 欄位並儲存結果
    print("正在處理 chief 欄位...")
    X_train_chief_embed = get_and_save_bert_embedding(
        X_train['chief'], 
        save_path="embeddings/X_train_chief_embed.npy"
    )
    X_test_chief_embed = get_and_save_bert_embedding(
        X_test['chief'], 
        save_path="embeddings/X_test_chief_embed.npy"
    )
    
    print("所有 embedding 處理完成！")
```

### 載入已儲存的 Embedding
後續可使用以下方式快速載入已處理的嵌入向量：
```python
# 載入儲存的 embedding 結果
X_train_diagnosis_embed = np.load("embeddings/X_train_diagnosis_embed.npy")
X_test_diagnosis_embed = np.load("embeddings/X_test_diagnosis_embed.npy")
X_train_chief_embed = np.load("embeddings/X_train_chief_embed.npy")
X_test_chief_embed = np.load("embeddings/X_test_chief_embed.npy")

print(f"載入完成 - 訓練集診斷嵌入: {X_train_diagnosis_embed.shape}")
print(f"載入完成 - 測試集診斷嵌入: {X_test_diagnosis_embed.shape}")
```

### 儲存格式說明
- **檔案格式**：`.npy`（NumPy 原生二進制格式）
- **優點**：
  - 載入速度快
  - 保持原始資料型態和精度
  - 檔案大小較小
- **適用場景**：
  - 避免重複計算 embedding
  - 模型實驗時快速載入特徵
  - 與其他 Python 程式共享資料

### 函數架構
兩個主要函數分工合作：

**1. `get_bert_embedding(text_series)` - 基礎工具函數**
```python
def get_bert_embedding(text_series):
    """將文字序列轉換為 BERT 嵌入向量"""
```
- **輸入**：pandas Series 的文字資料
- **輸出**：numpy array 的嵌入向量 (n_samples, 768)
- **功能**：專門處理文字轉換為向量的核心邏輯

**2. `process_data_with_embedding()` - 主流程函數**
```python
def process_data_with_embedding():
    """完整的資料處理流程：載入、分割、embedding、標準化"""
```
- **輸入**：無參數（內部載入資料）
- **輸出**：完整處理後的訓練集和測試集
- **功能**：協調所有處理步驟，呼叫基礎函數進行embedding

### BERT Embedding 詳細處理流程

#### 步驟1：文字預處理和 Tokenization
```python
inputs = tokenizer(str(text), return_tensors="pt", truncation=True, 
                  max_length=128, padding="max_length")
```
**詳細解釋**：
- **`str(text)`**：確保輸入是字串格式，防止數值型資料錯誤
- **`return_tensors="pt"`**：回傳 PyTorch 張量格式，供 BERT 模型使用
- **`truncation=True`**：如果文字超過 max_length 會自動截斷
- **`max_length=128`**：設定最大處理長度為128個 token
  - BERT 最大支援 512 tokens，但醫療短文通常不需要這麼長
  - 128 tokens 平衡了處理速度和文字完整性
- **`padding="max_length"`**：不足128的文字會用特殊 token 補齊到128

#### 步驟2：BERT 模型推理
```python
with torch.no_grad():
    outputs = model(**inputs)
```
**詳細解釋**：
- **`torch.no_grad()`**：關閉梯度計算，節省記憶體和計算資源
  - 因為我們只做推理（inference），不需要訓練
- **`model(**inputs)`**：將 tokenized 的輸入餵給 Clinical BERT
- **`outputs`**：包含所有 hidden states 和 attention weights 的複雜結構

#### 步驟3：提取句子表示向量
```python
cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
```
**詳細解釋**：
- **`outputs.last_hidden_state`**：BERT 最後一層的所有 token 隱藏狀態
  - Shape: (batch_size, sequence_length, hidden_size)
  - 即: (1, 128, 768)
- **`[:, 0, :]`**：選取第一個 token（[CLS] token）的向量
  - [CLS] 是 BERT 的特殊 token，代表整個句子的語意
  - Shape 變成: (1, 768)
- **`.cpu()`**：將 GPU 張量移到 CPU（如果有使用 GPU）
- **`.numpy()`**：轉換為 NumPy 陣列格式
- **`.squeeze()`**：移除多餘的維度，Shape 變成: (768,)

#### 步驟4：缺失值特殊處理
```python
if pd.isna(text):
    embeddings.append(np.zeros(model.config.hidden_size))
```
**詳細解釋**：
- **`pd.isna(text)`**：檢查是否為缺失值（NaN, None, 空字串等）
- **`np.zeros(model.config.hidden_size)`**：建立768維的零向量
- **為什麼用零向量**：
  - 零向量在向量空間中代表"無資訊"
  - 不會影響後續的數值計算（如歐式距離）
  - 比隨機向量更合理，比刪除資料更保守

### 資料合併和維度說明

#### 特徵維度構成
```python
X_train_final = np.concatenate([
    X_train_numeric_scaled,    # 24維數值特徵
    X_train_diagnosis_embed,   # 768維 diagnosis embedding  
    X_train_chief_embed        # 768維 chief embedding
], axis=1)
```

**維度詳細分析**：
- **數值特徵 (24維)**：生理指標、檢驗數值等結構化資料
- **Diagnosis Embedding (768維)**：診斷文字的語意向量表示
- **Chief Embedding (768維)**：主訴文字的語意向量表示
- **總計**：24 + 768 + 768 = **1560維**

#### 為什麼要合併不同類型特徵？
1. **互補性**：數值特徵提供精確測量，文字特徵提供語意資訊
2. **完整性**：涵蓋所有可用資訊，提升預測準確度
3. **標準化**：所有特徵都轉換為數值向量，便於機器學習演算法處理

### Clinical BERT 模型特點
- **專門訓練**：使用醫療文獻和臨床筆記訓練
- **自動下載**：首次執行會從 Hugging Face 自動下載
- **快取機制**：下載後儲存在 `~/.cache/huggingface/`
- **向量維度**：768維（標準 BERT 隱藏層大小）
- **處理長度**：最大128個 token（適合醫療短文）

### Embedding 處理技術細節

#### Tokenization 過程詳解
1. **文字切分**：將醫療文字切分成有意義的單位（tokens）
   - 例如："pneumonia with fever" → ["pneumonia", "with", "fever"]
   - 醫療專有名詞會被正確識別（Clinical BERT 的優勢）

2. **特殊 Token 添加**：
   - **[CLS]**：句子開頭，代表整句語意
   - **[SEP]**：句子結尾標記
   - **[PAD]**：填充 token，補齊到固定長度

3. **Token ID 轉換**：每個 token 轉換為對應的數字 ID
   - Clinical BERT 有專門的醫療詞彙表
   - 包含常見醫療術語和縮寫

#### 注意力機制 (Attention Mechanism)
BERT 使用 Multi-Head Self-Attention：
- **Self-Attention**：每個 token 都會關注句子中的其他 tokens
- **醫療文字優勢**：能理解症狀間的關聯性
  - 例如："chest pain" 和 "shortness of breath" 的關聯
- **語境理解**：相同詞彙在不同語境下有不同表示
  - "fever" 在不同疾病語境下會有不同的向量表示

#### 向量空間特性
- **768維空間**：每個維度捕捉不同的語意特徵
- **連續向量**：相似語意的文字在向量空間中距離較近
- **可計算性**：支援向量運算（加法、相似度計算等）

### 最終輸出格式
```
X_train_final.shape: (1257, 1560)  # 訓練集：1257筆 × 1560維特徵
X_test_final.shape: (629, 1560)    # 測試集：629筆 × 1560維特徵
y_train.shape: (1257,)             # 訓練集目標
y_test.shape: (629,)               # 測試集目標
```