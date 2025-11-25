# 敗血症預測模型
## 計畫內容
-  有1886人的資料
	- (依變項)
	- (獨變項)
- 訓練集 : 測試集 => 2:1
- 先用訓練集做出model，後用測試集進行測試
- model要用多種方式進行=> DT，SVM，ANN，RF =>去用python的scikit套件就好
- 要用10-fold cross varidation進行


## 數據集
- a-x欄 : 結構化數據
- y欄 : dignosis
- z欄 : chief
- AA欄 : 是否有敗血病

## embedding
- y欄要進行embedding 
- z欄要進行embedding 
- 可以用的工具「clinical BERT」

## 消融研究
- 用a-x欄建立模型
- 用Y欄建立模型
- 用Z欄建立模型
- 用a-y欄建立模型
- 用a-x，z欄建立模型
- 用a-z欄建立模型

每一種狀況都要產出一個Train和Test的圖
例: 
訓練集:

|     | AUC | precision | recall | F1  |
| --- | --- | --------- | ------ | --- |
| DT  |     |           |        |     |
| SVM |     |           |        |     |
| RF  |     |           |        |     |
| CNN |     |           |        |     |

測試集:

|     | AUC | precision | recall | F1  |
| --- | --- | --------- | ------ | --- |
| DT  |     |           |        |     |
| SVM |     |           |        |     |
| RF  |     |           |        |     |
| CNN |     |           |        |     |

## 描述性統計
- a-x每一個都要有描述性統計
- 包含min，max，mean，std，dev