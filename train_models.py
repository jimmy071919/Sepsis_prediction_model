import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

from load_data import load_data

class ModelTrainer:
    """機器學習模型訓練器，支援消融研究"""
    
    def __init__(self, df, random_state=42):
        """
        初始化訓練器
        
        Args:
            df: DataFrame，包含所有數據
            random_state: 隨機種子
        """
        self.df = df
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = {}
        # 初始化數據填充器 (用於處理NaN值)
        self.imputer = SimpleImputer(strategy='median')
        
    def _initialize_models(self):
        """初始化模型"""
        models = {
            'DT': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'SVM': SVC(random_state=self.random_state, probability=True, kernel='rbf', C=1.0),
            'RF': RandomForestClassifier(random_state=self.random_state, n_estimators=50, max_depth=10),
            'CNN': MLPClassifier(random_state=self.random_state, hidden_layer_sizes=(50,), 
                                max_iter=300, early_stopping=True, validation_fraction=0.1)
        }
        return models
    
    def train_and_evaluate(self, feature_type):
        """
        訓練和評估模型
        
        Args:
            feature_type: 特徵類型
            
        Returns:
            dict: 包含訓練和測試結果的字典
        """
        print(f"\\n=== 開始 {feature_type} 特徵的模型訓練 ===")
        
        # 載入已經分割好的數據
        try:
            # 載入結構化數據
            X_train_structured = np.load('structured_data_embedding/x_train_ax_scaled.npy')
            X_test_structured = np.load('structured_data_embedding/x_test_ax_scaled.npy')
            
            # 載入文本嵌入
            X_train_chief = np.load('unstructured_data_embedding/x_train_chief_embed.npy')
            X_test_chief = np.load('unstructured_data_embedding/x_test_chief_embed.npy')
            
            X_train_diagnosis = np.load('unstructured_data_embedding/x_train_diagnosis_embed.npy')
            X_test_diagnosis = np.load('unstructured_data_embedding/x_test_diagnosis_embed.npy')
            
            # 載入標籤
            y_train = np.load('answer_embedding/y_train.npy')
            y_test = np.load('answer_embedding/y_test.npy')
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"數據文件不存在: {e}")
        
        # 根據特徵類型組合訓練和測試數據
        if feature_type == 'a-x':
            X_train = X_train_structured
            X_test = X_test_structured
        elif feature_type == 'y':
            X_train = X_train_chief
            X_test = X_test_chief
        elif feature_type == 'z':
            X_train = X_train_diagnosis
            X_test = X_test_diagnosis
        elif feature_type == 'a-y':
            X_train = np.concatenate([X_train_structured, X_train_chief], axis=1)
            X_test = np.concatenate([X_test_structured, X_test_chief], axis=1)
        elif feature_type == 'a-x,z':
            X_train = np.concatenate([X_train_structured, X_train_diagnosis], axis=1)
            X_test = np.concatenate([X_test_structured, X_test_diagnosis], axis=1)
        elif feature_type == 'a-z':
            X_train = np.concatenate([X_train_structured, X_train_chief, X_train_diagnosis], axis=1)
            X_test = np.concatenate([X_test_structured, X_test_chief, X_test_diagnosis], axis=1)
        else:
            raise ValueError(f"不支援的特徵類型: {feature_type}")
        
        # 處理NaN值 (只對包含結構化數據的特徵組合進行填補)
        if feature_type in ['a-x', 'a-y', 'a-x,z', 'a-z']:
            # 檢查是否有NaN值
            if np.isnan(X_train).any() or np.isnan(X_test).any():
                print(f"檢測到NaN值，正在使用中位數填補...")
                # 在訓練集上擬合填補器
                self.imputer.fit(X_train)
                # 對訓練集和測試集進行填補
                X_train = self.imputer.transform(X_train)
                X_test = self.imputer.transform(X_test)
                print(f"NaN值填補完成")
        
        print(f"特徵矩陣 - 訓練集: {X_train.shape}, 測試集: {X_test.shape}")
        print(f"標籤向量 - 訓練集: {y_train.shape}, 測試集: {y_test.shape}")
        
        results = {
            'feature_type': feature_type,
            'train_metrics': {},
            'test_metrics': {},
            'models': {}
        }
        
        # 訓練各個模型
        for model_name, model in self.models.items():
            print(f"\\n訓練 {model_name} 模型...")
            
            try:
                # 訓練模型
                model.fit(X_train, y_train)
                
                # 預測
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # 計算訓練集指標
                train_metrics = self._calculate_metrics(y_train, y_train_pred, model, X_train)
                results['train_metrics'][model_name] = train_metrics
                
                # 計算測試集指標
                test_metrics = self._calculate_metrics(y_test, y_test_pred, model, X_test)
                results['test_metrics'][model_name] = test_metrics
                
                # 保存模型
                results['models'][model_name] = model
                
                print(f"{model_name} 完成 - 測試集 AUC: {test_metrics['AUC']:.3f}")
                
            except Exception as e:
                print(f"{model_name} 訓練失敗: {str(e)}")
                # 填充空結果
                empty_metrics = {'AUC': 0, 'precision': 0, 'recall': 0, 'F1': 0}
                results['train_metrics'][model_name] = empty_metrics
                results['test_metrics'][model_name] = empty_metrics
        
        self.results[feature_type] = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, model, X):
        """計算評估指標"""
        try:
            # 預測概率（用於AUC）
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_pred)
            
            metrics = {
                'AUC': auc,
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'F1': f1_score(y_true, y_pred, average='binary')
            }
        except Exception as e:
            print(f"指標計算錯誤: {str(e)}")
            metrics = {'AUC': 0, 'precision': 0, 'recall': 0, 'F1': 0}
        
        return metrics
    
    def run_ablation_study(self):
        """執行消融研究"""
        feature_types = ['a-x', 'y', 'z', 'a-y', 'a-x,z', 'a-z']
        
        print("=== 開始消融研究 ===")
        
        for feature_type in feature_types:
            try:
                self.train_and_evaluate(feature_type)
            except Exception as e:
                print(f"特徵類型 {feature_type} 訓練失敗: {str(e)}")
                continue
    
    def create_results_tables(self):
        """創建結果表格"""
        tables = {}
        
        for feature_type, results in self.results.items():
            print(f"\\n=== {feature_type} 特徵結果 ===")
            
            # 訓練集表格
            train_table = self._create_single_table(results['train_metrics'], "訓練集")
            print("\\n訓練集:")
            print(train_table.to_string(index=True, float_format='%.3f'))
            
            # 測試集表格
            test_table = self._create_single_table(results['test_metrics'], "測試集")
            print("\\n測試集:")
            print(test_table.to_string(index=True, float_format='%.3f'))
            
            tables[feature_type] = {
                'train': train_table,
                'test': test_table
            }
        
        return tables
    
    def _create_single_table(self, metrics_dict, dataset_type):
        """創建單個結果表格"""
        df = pd.DataFrame(metrics_dict).T
        df = df[['AUC', 'precision', 'recall', 'F1']]  # 確保列順序
        return df
    
    def save_results(self, filename_prefix='model_results'):
        """保存結果到CSV檔案"""
        for feature_type, results in self.results.items():
            # 保存訓練集結果
            train_table = self._create_single_table(results['train_metrics'], "訓練集")
            train_table.to_csv(f'result\\{filename_prefix}_{feature_type}_train.csv', encoding='utf-8-sig')
            
            # 保存測試集結果
            test_table = self._create_single_table(results['test_metrics'], "測試集")
            test_table.to_csv(f'result\\{filename_prefix}_{feature_type}_test.csv', encoding='utf-8-sig')
            
        print(f"結果已保存到 {filename_prefix}_*.csv 檔案")

