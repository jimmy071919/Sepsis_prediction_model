"""
特徵優化處理模組
包含降維、類別平衡等功能
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import os

def apply_pca_to_embeddings(X_train_embed, X_test_embed, n_components=30, feature_name="embedding"):
    """
    對文字嵌入進行PCA降維
    
    Parameters:
    X_train_embed: ndarray，訓練集文字嵌入
    X_test_embed: ndarray，測試集文字嵌入
    n_components: int，降維後的維度數
    feature_name: str，特徵名稱
    
    Returns:
    X_train_pca: ndarray，降維後的訓練集
    X_test_pca: ndarray，降維後的測試集
    pca: PCA object，PCA轉換器
    explained_variance_ratio: float，保留的變異量比例
    """
    
    print(f"   正在對{feature_name}進行PCA降維...")
    print(f"   原始維度: {X_train_embed.shape[1]} -> 目標維度: {n_components}")
    
    # 初始化PCA
    pca = PCA(n_components=n_components, random_state=42)
    
    # 在訓練集上擬合PCA
    X_train_pca = pca.fit_transform(X_train_embed)
    X_test_pca = pca.transform(X_test_embed)
    
    # 計算保留的變異量
    total_variance_ratio = np.sum(pca.explained_variance_ratio_)
    
    print(f"   PCA降維完成，保留變異量: {total_variance_ratio:.3f}")
    print(f"   訓練集形狀: {X_train_pca.shape}, 測試集形狀: {X_test_pca.shape}")
    
    return X_train_pca, X_test_pca, pca, total_variance_ratio

def save_pca_embeddings(X_train_pca, X_test_pca, pca, feature_name, output_dir="optimized_embeddings"):
    """保存PCA降維後的嵌入"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存降維後的數據
    np.save(os.path.join(output_dir, f"x_train_{feature_name}_pca.npy"), X_train_pca)
    np.save(os.path.join(output_dir, f"x_test_{feature_name}_pca.npy"), X_test_pca)
    
    # 保存PCA轉換器
    import joblib
    joblib.dump(pca, os.path.join(output_dir, f"{feature_name}_pca.pkl"))
    
    print(f"   PCA結果已保存到 {output_dir}/ 目錄")

def apply_smote_to_features(X_combined, y, random_state=42):
    """
    使用SMOTE處理類別不平衡
    
    Parameters:
    X_combined: ndarray，特徵矩陣
    y: ndarray，標籤
    random_state: int，隨機種子
    
    Returns:
    X_balanced: ndarray，平衡後的特徵矩陣
    y_balanced: ndarray，平衡後的標籤
    """
    
    print(f"   原始數據分布: {np.bincount(y)}")
    
    # 使用SMOTE + Tomek Links來平衡數據
    smote_tomek = SMOTETomek(random_state=random_state)
    X_balanced, y_balanced = smote_tomek.fit_resample(X_combined, y)
    
    print(f"   SMOTE後數據分布: {np.bincount(y_balanced)}")
    print(f"   數據形狀: {X_balanced.shape}")
    
    return X_balanced, y_balanced

def optimize_text_embeddings(feature_configs):
    """
    批量優化文字嵌入特徵
    
    Parameters:
    feature_configs: list，特徵配置列表
    格式: [{'name': 'chief', 'train_path': '...', 'test_path': '...', 'n_components': 30}, ...]
    
    Returns:
    optimized_features: dict，優化後的特徵字典
    """
    
    print("=== 開始文字嵌入優化 ===")
    optimized_features = {}
    
    for config in feature_configs:
        feature_name = config['name']
        train_path = config['train_path']
        test_path = config['test_path']
        n_components = config['n_components']
        
        print(f"\n處理 {feature_name} 特徵...")
        
        # 載入原始嵌入
        X_train_embed = np.load(train_path)
        X_test_embed = np.load(test_path)
        
        # 應用PCA降維
        X_train_pca, X_test_pca, pca, variance_ratio = apply_pca_to_embeddings(
            X_train_embed, X_test_embed, n_components, feature_name
        )
        
        # 保存結果
        save_pca_embeddings(X_train_pca, X_test_pca, pca, feature_name)
        
        # 存儲到字典
        optimized_features[feature_name] = {
            'train': X_train_pca,
            'test': X_test_pca,
            'pca': pca,
            'variance_ratio': variance_ratio,
            'n_components': n_components
        }
    
    print("\n=== 文字嵌入優化完成 ===")
    return optimized_features

def create_optimization_report(optimized_features, output_path="optimized_embeddings/optimization_report.txt"):
    """創建優化報告"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 特徵優化報告\n\n")
        f.write(f"優化時間: {pd.Timestamp.now()}\n\n")
        
        f.write("## PCA降維結果\n\n")
        for feature_name, info in optimized_features.items():
            f.write(f"### {feature_name} 特徵\n")
            f.write(f"- 降維維度: {info['n_components']}\n")
            f.write(f"- 保留變異量: {info['variance_ratio']:.3f}\n")
            f.write(f"- 訓練集形狀: {info['train'].shape}\n")
            f.write(f"- 測試集形狀: {info['test'].shape}\n\n")
        
        f.write("## 優化目標\n")
        f.write("1. 降低維度災難問題\n")
        f.write("2. 減少特徵間的雜訊干擾\n")
        f.write("3. 提高模型訓練效率\n")
        f.write("4. 改善類別不平衡問題\n")
    
    print(f"優化報告已保存: {output_path}")

if __name__ == "__main__":
    print("特徵優化模組載入完成")
    print("主要功能:")
    print("1. apply_pca_to_embeddings() - PCA降維")
    print("2. apply_smote_to_features() - SMOTE類別平衡")
    print("3. optimize_text_embeddings() - 批量優化文字嵌入")