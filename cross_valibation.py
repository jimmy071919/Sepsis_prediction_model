import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from load_data import load_data

class CrossValidationTrainer:
    """使用10-fold交叉驗證的模型訓練器"""
    
    def __init__(self, df, random_state=42, n_folds=10):
        """
        初始化交叉驗證訓練器
        
        Args:
            df: DataFrame，包含所有數據
            random_state: 隨機種子
            n_folds: 交叉驗證折數
        """
        self.df = df
        self.random_state = random_state
        self.n_folds = n_folds
        self.models = self._initialize_models()
        self.cv_results = {}
        self.test_results = {}
        # 初始化數據填充器 (用於處理NaN值)
        self.imputer = SimpleImputer(strategy='median')
        # 初始化交叉驗證策略
        self.cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
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
    
    def _get_scoring_metrics(self):
        """定義評估指標"""
        scoring = {
            'auc': 'roc_auc',
            'precision': make_scorer(precision_score, average='binary', zero_division=0),
            'recall': make_scorer(recall_score, average='binary', zero_division=0),
            'f1': make_scorer(f1_score, average='binary', zero_division=0)
        }
        return scoring
    
    def load_and_prepare_data(self, feature_type):
        """載入和準備指定特徵類型的數據"""
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
        
        return X_train, X_test, y_train, y_test
    
    def perform_cross_validation(self, feature_type):
        """對指定特徵類型執行10-fold交叉驗證"""
        print(f"\\n=== 開始 {feature_type} 特徵的10-fold交叉驗證 ===")
        
        # 載入數據
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(feature_type)
        
        print(f"訓練集特徵矩陣: {X_train.shape}")
        print(f"測試集特徵矩陣: {X_test.shape}")
        print(f"訓練集標籤: {y_train.shape}")
        print(f"測試集標籤: {y_test.shape}")
        
        # 獲取評估指標
        scoring = self._get_scoring_metrics()
        
        cv_results = {}
        test_results = {}
        
        # 對每個模型進行交叉驗證
        for model_name, model in self.models.items():
            print(f"\\n正在進行 {model_name} 模型的10-fold交叉驗證...")
            
            try:
                # 執行交叉驗證
                cv_scores = cross_validate(
                    model, X_train, y_train,
                    cv=self.cv_strategy,
                    scoring=scoring,
                    n_jobs=-1,  # 使用所有可用核心
                    return_train_score=False
                )
                
                # 計算交叉驗證平均分數和標準差
                cv_result = {
                    'AUC_mean': np.mean(cv_scores['test_auc']),
                    'AUC_std': np.std(cv_scores['test_auc']),
                    'precision_mean': np.mean(cv_scores['test_precision']),
                    'precision_std': np.std(cv_scores['test_precision']),
                    'recall_mean': np.mean(cv_scores['test_recall']),
                    'recall_std': np.std(cv_scores['test_recall']),
                    'f1_mean': np.mean(cv_scores['test_f1']),
                    'f1_std': np.std(cv_scores['test_f1'])
                }
                
                cv_results[model_name] = cv_result
                
                print(f"{model_name} 交叉驗證完成:")
                print(f"  AUC: {cv_result['AUC_mean']:.3f} (±{cv_result['AUC_std']:.3f})")
                print(f"  F1: {cv_result['f1_mean']:.3f} (±{cv_result['f1_std']:.3f})")
                
                # 在整個訓練集上訓練模型，然後在測試集上評估
                print(f"正在訓練 {model_name} 並在測試集上評估...")
                model.fit(X_train, y_train)
                
                # 在測試集上預測
                y_test_pred = model.predict(X_test)
                y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
                
                # 計算測試集指標
                test_result = {
                    'AUC': roc_auc_score(y_test, y_test_prob),
                    'precision': precision_score(y_test, y_test_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_test_pred, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_test_pred, average='binary', zero_division=0)
                }
                
                test_results[model_name] = test_result
                
                print(f"{model_name} 測試集結果:")
                print(f"  AUC: {test_result['AUC']:.3f}")
                print(f"  F1: {test_result['f1']:.3f}")
                
            except Exception as e:
                print(f"{model_name} 訓練失敗: {str(e)}")
                # 填充空結果
                empty_cv_result = {
                    'AUC_mean': 0, 'AUC_std': 0,
                    'precision_mean': 0, 'precision_std': 0,
                    'recall_mean': 0, 'recall_std': 0,
                    'f1_mean': 0, 'f1_std': 0
                }
                empty_test_result = {'AUC': 0, 'precision': 0, 'recall': 0, 'f1': 0}
                cv_results[model_name] = empty_cv_result
                test_results[model_name] = empty_test_result
        
        # 保存結果
        self.cv_results[feature_type] = cv_results
        self.test_results[feature_type] = test_results
        
        return cv_results, test_results
    
    def run_cross_validation_study(self):
        """執行完整的交叉驗證研究"""
        feature_types = ['a-x', 'y', 'z', 'a-y', 'a-x,z', 'a-z']
        
        print(f"=== 開始 {self.n_folds}-fold 交叉驗證研究 ===")
        
        for feature_type in feature_types:
            try:
                self.perform_cross_validation(feature_type)
            except Exception as e:
                print(f"特徵類型 {feature_type} 交叉驗證失敗: {str(e)}")
                continue
        
        # 生成總結報告
        self.create_summary_report()
    
    def create_summary_report(self):
        """創建交叉驗證結果總結報告"""
        print(f"\\n\\n=== {self.n_folds}-fold 交叉驗證結果總結 ===")
        
        for feature_type in self.cv_results:
            print(f"\\n=== {feature_type} 特徵 ===")
            
            # 交叉驗證結果
            print(f"\\n{self.n_folds}-fold 交叉驗證結果 (平均值 ± 標準差):")
            cv_df_data = []
            for model_name, results in self.cv_results[feature_type].items():
                cv_df_data.append({
                    'Model': model_name,
                    'AUC': f"{results['AUC_mean']:.3f} ± {results['AUC_std']:.3f}",
                    'Precision': f"{results['precision_mean']:.3f} ± {results['precision_std']:.3f}",
                    'Recall': f"{results['recall_mean']:.3f} ± {results['recall_std']:.3f}",
                    'F1': f"{results['f1_mean']:.3f} ± {results['f1_std']:.3f}"
                })
            
            cv_df = pd.DataFrame(cv_df_data)
            print(cv_df.to_string(index=False))
            
            # 測試集結果
            print(f"\\n測試集評估結果:")
            test_df_data = []
            for model_name, results in self.test_results[feature_type].items():
                test_df_data.append({
                    'Model': model_name,
                    'AUC': f"{results['AUC']:.3f}",
                    'Precision': f"{results['precision']:.3f}",
                    'Recall': f"{results['recall']:.3f}",
                    'F1': f"{results['f1']:.3f}"
                })
            
            test_df = pd.DataFrame(test_df_data)
            print(test_df.to_string(index=False))
    
    def save_results_to_csv(self):
        """將結果保存為CSV文件"""
        # 保存交叉驗證結果
        for feature_type in self.cv_results:
            cv_data = []
            test_data = []
            
            for model_name in self.cv_results[feature_type]:
                # 交叉驗證結果
                cv_results = self.cv_results[feature_type][model_name]
                cv_data.append({
                    'Model': model_name,
                    'AUC_mean': cv_results['AUC_mean'],
                    'AUC_std': cv_results['AUC_std'],
                    'precision_mean': cv_results['precision_mean'],
                    'precision_std': cv_results['precision_std'],
                    'recall_mean': cv_results['recall_mean'],
                    'recall_std': cv_results['recall_std'],
                    'f1_mean': cv_results['f1_mean'],
                    'f1_std': cv_results['f1_std']
                })
                
                # 測試集結果
                test_results = self.test_results[feature_type][model_name]
                test_data.append({
                    'Model': model_name,
                    'AUC': test_results['AUC'],
                    'precision': test_results['precision'],
                    'recall': test_results['recall'],
                    'f1': test_results['f1']
                })
            
            # 保存到CSV
            cv_df = pd.DataFrame(cv_data)
            test_df = pd.DataFrame(test_data)
            
            cv_df.to_csv(f'result/cv_results_{feature_type.replace("-", "_").replace(",", "_")}.csv', index=False)
            test_df.to_csv(f'result/test_results_{feature_type.replace("-", "_").replace(",", "_")}.csv', index=False)
        
        print(f"\\n結果已保存到 result/cv_results_*.csv 和 result/test_results_*.csv 檔案")


def main():
    """主函數"""
    # 載入數據
    df = load_data('data/1141112.xlsx')
    
    # 創建交叉驗證訓練器
    cv_trainer = CrossValidationTrainer(df, random_state=42, n_folds=10)
    
    # 執行交叉驗證研究
    cv_trainer.run_cross_validation_study()
    
    # 保存結果
    cv_trainer.save_results_to_csv()


if __name__ == "__main__":
    main()
