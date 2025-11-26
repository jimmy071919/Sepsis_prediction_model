"""
特徵重要性分析模組
分析並可視化敗血症預測模型中各特徵的重要性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VariableImportanceAnalyzer:
    """變數重要性分析器"""
    
    def __init__(self, data_path="data/1141112.xlsx"):
        """
        初始化分析器
        
        Args:
            data_path: 數據文件路徑
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.importance_results = {}
        
    def load_and_preprocess_data(self):
        """載入並預處理數據"""
        try:
            # 載入數據
            self.df = pd.read_excel(self.data_path)
            print(f"成功載入數據，形狀: {self.df.shape}")
            
            # 預處理目標變量
            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(self.df['isSepsis'])
            
            # 選擇結構化特徵（排除文本和目標變量）
            exclude_cols = ['isSepsis', 'diagnosis', 'chief']
            feature_cols = [col for col in self.df.columns if col not in exclude_cols]
            
            # 處理特徵數據
            X_raw = self.df[feature_cols].copy()
            
            # 處理缺失值和異常值
            X_processed = self._handle_missing_and_outliers(X_raw)
            
            # 數值化處理
            self.X, self.feature_names = self._encode_features(X_processed, feature_cols)
            
            print(f"預處理完成，特徵數量: {self.X.shape[1]}")
            return True
            
        except Exception as e:
            print(f"數據載入和預處理失敗: {e}")
            return False
    
    def _handle_missing_and_outliers(self, X):
        """處理缺失值和異常值"""
        X_processed = X.copy()
        
        # 定義醫學合理範圍
        medical_ranges = {
            'BT': (30, 45),
            'SBP': (40, 300),
            'DBP': (20, 200),
            'MAP': (10, 300),
            'BMI': (10, 100),
            'Pulse': (30, 250),
            'LOS': (0, 10000)
        }
        
        # 處理異常值
        for col, (min_val, max_val) in medical_ranges.items():
            if col in X_processed.columns:
                mask = (X_processed[col] < min_val) | (X_processed[col] > max_val)
                X_processed.loc[mask, col] = np.nan
        
        # 處理不應為零的變量
        zero_invalid_cols = ['Weight', 'WBC', 'PLT', 'Crea', 'T-Bil', 
                           'Lymph', 'Segment', 'PT', 'PCT', 'BMI', 
                           'SBP', 'DBP', 'MAP']
        
        for col in zero_invalid_cols:
            if col in X_processed.columns:
                X_processed.loc[X_processed[col] == 0, col] = np.nan
        
        return X_processed
    
    def _encode_features(self, X_processed, feature_cols):
        """編碼特徵"""
        X_encoded = X_processed.copy()
        encoded_feature_names = []
        
        for col in feature_cols:
            if X_processed[col].dtype == 'object':
                # 類別變量編碼
                le = LabelEncoder()
                # 填充缺失值為'Unknown'
                X_encoded[col] = X_encoded[col].fillna('Unknown')
                X_encoded[col] = le.fit_transform(X_encoded[col])
            
            encoded_feature_names.append(col)
        
        # 使用中位數填補數值特徵的缺失值
        imputer = SimpleImputer(strategy='median')
        X_final = imputer.fit_transform(X_encoded)
        
        return X_final, encoded_feature_names
    
    def calculate_random_forest_importance(self):
        """計算隨機森林特徵重要性"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                  class_weight='balanced')
        rf.fit(self.X, self.y)
        
        importance_scores = rf.feature_importances_
        
        # 創建重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        self.importance_results['Random_Forest'] = importance_df
        return importance_df
    
    def calculate_mutual_information(self):
        """計算互信息（信息增益比的近似）"""
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # 創建互信息DataFrame
        mi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
        
        self.importance_results['Mutual_Information'] = mi_df
        return mi_df
    
    def calculate_f_statistic_importance(self):
        """計算F統計量特徵重要性（ANOVA F-test）"""
        f_scores, p_values = f_classif(self.X, self.y)
        
        # 創建F統計量DataFrame
        f_df = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Score': f_scores,
            'P_Value': p_values
        }).sort_values('F_Score', ascending=False)
        
        self.importance_results['F_Statistic'] = f_df
        return f_df
    
    def calculate_correlation_importance(self):
        """計算皮爾遜相關係數重要性"""
        correlations = []
        p_values = []
        
        for i, feature_name in enumerate(self.feature_names):
            corr, p_val = pearsonr(self.X[:, i], self.y)
            correlations.append(abs(corr))  # 使用絕對值
            p_values.append(p_val)
        
        # 創建相關係數DataFrame
        corr_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Correlation': correlations,
            'P_Value': p_values
        }).sort_values('Correlation', ascending=False)
        
        self.importance_results['Correlation'] = corr_df
        return corr_df
    
    def plot_feature_importance(self, method='Random_Forest', top_n=24, figsize=(14, 12)):
        """
        繪製特徵重要性圖
        
        Args:
            method: 重要性計算方法 ('Random_Forest' 或 'Mutual_Information')
            top_n: 顯示前N個重要特徵
            figsize: 圖形大小
        """
        if method not in self.importance_results:
            print(f"請先計算 {method} 重要性")
            return
        
        importance_df = self.importance_results[method]
        top_features = importance_df.head(top_n)
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 繪製水平條形圖
        bars = ax.barh(range(len(top_features)), 
                      top_features.iloc[:, 1].values,
                      color='steelblue', alpha=0.7)
        
        # 設置標籤
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.iloc[:, 0].values)
        
        # 添加數值標籤
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        # 設置標題和標籤
        title_map = {
            'Random_Forest': '隨機森林特徵重要性',
            'Mutual_Information': '互信息特徵重要性'
        }
        ax.set_title(f'{title_map[method]} (Top {top_n})', fontsize=16, fontweight='bold')
        
        xlabel_map = {
            'Random_Forest': '重要性分數',
            'Mutual_Information': '互信息分數'
        }
        ax.set_xlabel(xlabel_map[method], fontsize=12)
        ax.set_ylabel('特徵變數', fontsize=12)
        
        # 反轉y軸使最重要的特徵在頂部
        ax.invert_yaxis()
        
        # 添加網格
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖片
        filename = f'feature_importance_{method.lower()}_{top_n}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"圖片已保存: {filename}")
        
        plt.show()
        
        return fig, ax
    
    def plot_gain_ratio_style(self, top_n=24, figsize=(16, 14)):
        """
        繪製類似您提供圖片的增益比風格圖
        """
        if 'Mutual_Information' not in self.importance_results:
            print("請先計算互信息")
            return
            
        importance_df = self.importance_results['Mutual_Information'].copy()
        importance_df = importance_df.head(top_n)
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用漸變藍色配色方案
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), 
                      importance_df['Mutual_Information'].values,
                      color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # 設置標籤，使用英文以匹配圖片風格
        feature_mapping = {
            'PCT': 'PCT',
            'PT': 'PT', 
            'Hs-CRP': 'Hs-CRP',
            'MAP': 'MAP',
            'T-Bil': 'T-Bil',
            'GCS': 'GCS',
            'BOXY': 'BOXY',
            'BMI': 'BMI',
            'LOS': 'LOS',
            'PLT': 'PLT',
            'WBC': 'WBC',
            'SBP': 'SBP',
            'DBP': 'DBP',
            'Pluse': 'Pulse',  # 注意原始數據是Pluse
            'AGE': 'Age',
            'BT': 'BT',
            'Weight': 'Weight',
            'Height': 'Height',
            'Crea': 'Crea',
            'Lymph': 'Lymph',
            'Segment': 'Segment',
            'RR': 'Respiratory rate',
            'SEX': 'Sex',
            'Urine_Volumn': 'Urine Volume'
        }
        
        feature_labels = [feature_mapping.get(f, f) for f in importance_df['Feature'].values]
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(feature_labels, fontsize=12)
        
        # 添加數值標籤
        for i, (bar, score) in enumerate(zip(bars, importance_df['Mutual_Information'].values)):
            ax.text(score + score*0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontsize=10, 
                   fontweight='bold', color='black')
        
        # 設置標題和標籤，匹配圖片風格
        ax.set_title('FIGURE 1. Variable importance.', fontsize=16, fontweight='bold', 
                    loc='left', pad=20)
        ax.set_xlabel('GAIN RATIO', fontsize=14, fontweight='bold')
        
        # 反轉y軸，使最重要的在頂部
        ax.invert_yaxis()
        
        # 設置x軸範圍和刻度
        max_val = importance_df['Mutual_Information'].max()
        ax.set_xlim(0, max_val * 1.15)
        
        # 設置x軸刻度
        x_ticks = np.arange(0, max_val * 1.15, 0.005)
        ax.set_xticks(x_ticks)
        
        # 添加垂直網格線
        ax.grid(axis='x', alpha=0.4, linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        # 移除頂部和右側邊框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # 設置背景色
        ax.set_facecolor('#f8f9fa')
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖片
        filename = f'variable_importance_gain_ratio_professional_{top_n}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"專業版圖片已保存: {filename}")
        
        plt.show()
        
        return fig, ax
    
    def create_comparison_plot(self, top_n=24, figsize=(20, 14)):
        """
        創建隨機森林和互信息的對比圖
        """
        if len(self.importance_results) < 2:
            print("需要計算兩種重要性方法才能進行比較")
            return
            
        rf_df = self.importance_results['Random_Forest'].head(top_n)
        mi_df = self.importance_results['Mutual_Information'].head(top_n)
        
        # 合併數據
        merged_df = pd.merge(rf_df[['Feature', 'Importance']], 
                           mi_df[['Feature', 'Mutual_Information']], 
                           on='Feature', how='outer').fillna(0)
        
        # 按隨機森林重要性排序
        merged_df = merged_df.sort_values('Importance', ascending=True)
        
        # 創建圖形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 隨機森林重要性圖
        bars1 = ax1.barh(range(len(merged_df)), merged_df['Importance'], 
                        color='lightcoral', alpha=0.8)
        ax1.set_yticks(range(len(merged_df)))
        ax1.set_yticklabels(merged_df['Feature'])
        ax1.set_title('隨機森林特徵重要性', fontweight='bold')
        ax1.set_xlabel('重要性分數')
        
        # 互信息重要性圖
        bars2 = ax2.barh(range(len(merged_df)), merged_df['Mutual_Information'], 
                        color='steelblue', alpha=0.8)
        ax2.set_yticks(range(len(merged_df)))
        ax2.set_yticklabels(merged_df['Feature'])
        ax2.set_title('互信息特徵重要性', fontweight='bold')
        ax2.set_xlabel('互信息分數')
        
        # 添加數值標籤
        for bars, values in [(bars1, merged_df['Importance']), 
                           (bars2, merged_df['Mutual_Information'])]:
            for bar, val in zip(bars, values):
                if val > 0:
                    ax = bar.axes
                    ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("對比圖已保存: feature_importance_comparison.png")
        plt.show()
        
        return fig, (ax1, ax2)
    
    def create_comprehensive_comparison(self, figsize=(24, 16)):
        """
        創建包含所有方法的綜合對比圖
        """
        if len(self.importance_results) < 4:
            print("需要計算所有四種重要性方法")
            return
        
        # 準備數據
        rf_df = self.importance_results['Random_Forest'].set_index('Feature')['Importance']
        mi_df = self.importance_results['Mutual_Information'].set_index('Feature')['Mutual_Information']
        f_df = self.importance_results['F_Statistic'].set_index('Feature')['F_Score']
        corr_df = self.importance_results['Correlation'].set_index('Feature')['Correlation']
        
        # 合併所有數據
        all_features = set(rf_df.index) | set(mi_df.index) | set(f_df.index) | set(corr_df.index)
        
        # 創建完整的DataFrame
        comparison_df = pd.DataFrame({
            'Random_Forest': [rf_df.get(f, 0) for f in all_features],
            'Mutual_Information': [mi_df.get(f, 0) for f in all_features],
            'F_Statistic': [f_df.get(f, 0) for f in all_features],
            'Correlation': [corr_df.get(f, 0) for f in all_features]
        }, index=list(all_features))
        
        # 標準化數據（0-1範圍）
        comparison_df_norm = comparison_df.div(comparison_df.max())
        
        # 計算綜合得分（簡單平均）
        comparison_df_norm['Composite_Score'] = comparison_df_norm.mean(axis=1)
        comparison_df_norm = comparison_df_norm.sort_values('Composite_Score', ascending=False)
        
        # 創建圖形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        methods = ['Random_Forest', 'Mutual_Information', 'F_Statistic', 'Correlation']
        axes = [ax1, ax2, ax3, ax4]
        colors = ['lightcoral', 'steelblue', 'lightgreen', 'orange']
        titles = ['隨機森林重要性', '互信息重要性', 'F統計量重要性', '相關係數重要性']
        
        # 繪製四個子圖
        for method, ax, color, title in zip(methods, axes, colors, titles):
            data = comparison_df_norm[method]
            bars = ax.barh(range(len(data)), data, color=color, alpha=0.8)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=8)
            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('標準化重要性分數', fontsize=10)
            
            # 添加數值標籤
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0.01:  # 只顯示大於0.01的值
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{width:.2f}', ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        plt.savefig('comprehensive_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        print("綜合對比圖已保存: comprehensive_feature_importance_comparison.png")
        plt.show()
        
        return fig, comparison_df_norm
    
    def create_method_explanation_plot(self, figsize=(16, 10)):
        """
        創建方法解釋圖，說明為什麼選擇這些方法
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        methods_info = {
            '隨機森林\n(Random Forest)': {
                'description': '基於決策樹的集成方法\n衡量特徵在分割中的貢獻',
                'advantages': '• 處理非線性關係\n• 抗過擬合\n• 適合醫學數據',
                'principle': '基尼不純度/熵減少',
                'y_pos': 3
            },
            '互信息\n(Mutual Information)': {
                'description': '信息理論方法\n衡量變數間信息共享量',
                'advantages': '• 類似增益比\n• 不假設線性關係\n• 捕捉統計依賴',
                'principle': '信息熵理論',
                'y_pos': 2
            },
            'F統計量\n(ANOVA F-test)': {
                'description': '方差分析方法\n檢驗組間差異顯著性',
                'advantages': '• 統計學經典方法\n• 提供p值\n• 適合連續變數',
                'principle': '組間/組內變異比',
                'y_pos': 1
            },
            '相關係數\n(Correlation)': {
                'description': '線性關聯強度\n衡量變數間線性關係',
                'advantages': '• 直觀易懂\n• 醫學研究常用\n• 快速篩選',
                'principle': '皮爾遜相關',
                'y_pos': 0
            }
        }
        
        # 設置背景
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 4.5)
        
        # 繪製每個方法的信息框
        for method, info in methods_info.items():
            y = info['y_pos']
            
            # 方法名稱
            ax.text(1, y, method, fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            # 描述
            ax.text(3, y, info['description'], fontsize=11, va='center')
            
            # 優勢
            ax.text(5.5, y, info['advantages'], fontsize=10, va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.5))
            
            # 原理
            ax.text(8.5, y, info['principle'], fontsize=10, va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))
        
        # 設置標題和標籤
        ax.set_title('特徵重要性分析方法比較', fontsize=18, fontweight='bold', pad=20)
        
        # 添加列標題
        ax.text(1, 4.3, '方法', fontsize=12, fontweight='bold', ha='center')
        ax.text(3, 4.3, '描述', fontsize=12, fontweight='bold', ha='center')
        ax.text(5.5, 4.3, '優勢', fontsize=12, fontweight='bold', ha='center')
        ax.text(8.5, 4.3, '統計原理', fontsize=12, fontweight='bold', ha='center')
        
        # 隱藏坐標軸
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_importance_methods_explanation.png', dpi=300, bbox_inches='tight')
        print("方法解釋圖已保存: feature_importance_methods_explanation.png")
        plt.show()
        
        return fig
    
    def generate_importance_summary(self):
        """生成重要性分析摘要報告"""
        if not self.importance_results:
            print("請先計算特徵重要性")
            return
            
        print("=== 特徵重要性分析摘要 ===")
        print(f"數據集大小: {self.X.shape}")
        print(f"敗血症比例: {self.y.sum()}/{len(self.y)} ({self.y.mean():.2%})")
        print(f"結構化特徵總數: {len(self.feature_names)}")
        print()
        
        # 顯示每種方法的前10重要特徵
        method_names = {
            'Random_Forest': '隨機森林',
            'Mutual_Information': '互信息',
            'F_Statistic': 'F統計量',
            'Correlation': '相關係數'
        }
        
        for method, df in self.importance_results.items():
            method_name = method_names.get(method, method)
            print(f"--- {method_name} 前10重要特徵 ---")
            top_10 = df.head(10)
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                feature_name = row.iloc[0]
                score = row.iloc[1]
                if method == 'F_Statistic' and len(row) > 2:
                    p_val = row.iloc[2]
                    print(f"{i:2d}. {feature_name:<15} {score:.4f} (p={p_val:.3e})")
                elif method == 'Correlation' and len(row) > 2:
                    p_val = row.iloc[2]
                    print(f"{i:2d}. {feature_name:<15} {score:.4f} (p={p_val:.3e})")
                else:
                    print(f"{i:2d}. {feature_name:<15} {score:.4f}")
            print()
        
        # 顯示方法選擇的理論依據
        print("=== 方法選擇理論依據 ===")
        print("1. 隨機森林: 基於基尼不純度，適合處理複雜非線性關係")
        print("2. 互信息: 基於信息理論，類似增益比概念，不假設特定分布")
        print("3. F統計量: 基於ANOVA，提供統計顯著性檢驗")
        print("4. 相關係數: 基於線性關聯，醫學研究的經典方法")
        print()
    
    def run_complete_analysis(self):
        """執行完整的特徵重要性分析"""
        print("=== 開始特徵重要性分析 ===")
        
        # 1. 載入和預處理數據
        if not self.load_and_preprocess_data():
            return False
        
        # 2. 計算隨機森林重要性
        print("計算隨機森林特徵重要性...")
        self.calculate_random_forest_importance()
        
        # 3. 計算互信息
        print("計算互信息...")
        self.calculate_mutual_information()
        
        # 4. 計算F統計量
        print("計算F統計量重要性...")
        self.calculate_f_statistic_importance()
        
        # 5. 計算相關係數
        print("計算相關係數重要性...")
        self.calculate_correlation_importance()
        
        # 6. 生成摘要報告
        self.generate_importance_summary()
        
        # 7. 繪製圖表
        print("生成可視化圖表...")
        self.create_method_explanation_plot()  # 先顯示方法解釋
        self.plot_feature_importance('Random_Forest', top_n=24)
        self.plot_gain_ratio_style(top_n=24)
        self.create_comparison_plot(top_n=24)
        self.create_comprehensive_comparison()
        
        print("=== 特徵重要性分析完成 ===")
        return True


def main():
    """主函數"""
    analyzer = VariableImportanceAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
