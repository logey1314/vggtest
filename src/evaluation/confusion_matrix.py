"""
混淆矩阵分析和可视化模块
专门针对混凝土坍落度分类任务的混淆矩阵分析
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


class ConfusionMatrixAnalyzer:
    """
    混淆矩阵分析器
    提供混淆矩阵的计算、可视化和深度分析功能
    """
    
    def __init__(self, class_names=None):
        """
        初始化混淆矩阵分析器
        
        Args:
            class_names (list): 类别名称列表
        """
        if class_names is None:
            self.class_names = ["125-175mm", "180-230mm", "233-285mm"]
        else:
            self.class_names = class_names
            
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def calculate_confusion_matrix(self, y_true, y_pred, normalize=None):
        """
        计算混淆矩阵
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            normalize (str): 归一化方式 ('true', 'pred', 'all', None)
            
        Returns:
            numpy.ndarray: 混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()
            
        return cm
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=None, 
                            save_path=None, title="混淆矩阵", figsize=(10, 8)):
        """
        绘制混淆矩阵热力图
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            normalize (str): 归一化方式
            save_path (str): 保存路径
            title (str): 图表标题
            figsize (tuple): 图像大小
        """
        cm = self.calculate_confusion_matrix(y_true, y_pred, normalize)
        
        plt.figure(figsize=figsize)
        
        # 选择颜色映射
        if normalize:
            fmt = '.2%' if normalize == 'all' else '.2f'
            cmap = 'Blues'
        else:
            fmt = 'd'
            cmap = 'Blues'
        
        # 绘制热力图
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap=cmap,
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': '样本数量' if not normalize else '比例'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)
        
        # 添加统计信息
        accuracy = np.trace(cm) / np.sum(cm) if not normalize else np.trace(cm)
        if normalize:
            plt.figtext(0.02, 0.02, f'准确率: {accuracy:.2%}', fontsize=12)
        else:
            plt.figtext(0.02, 0.02, f'总样本: {np.sum(cm)}, 准确率: {accuracy:.2%}', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 混淆矩阵已保存: {save_path}")
        
        plt.show()
        return cm
    
    def plot_multiple_confusion_matrices(self, y_true, y_pred, save_dir=None):
        """
        绘制多种形式的混淆矩阵
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            save_dir (str): 保存目录
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 原始数量矩阵
        save_path1 = os.path.join(save_dir, "confusion_matrix_counts.png") if save_dir else None
        cm_counts = self.plot_confusion_matrix(
            y_true, y_pred, 
            normalize=None,
            save_path=save_path1,
            title="混淆矩阵 - 样本数量",
            figsize=(8, 6)
        )
        
        # 按真实标签归一化
        save_path2 = os.path.join(save_dir, "confusion_matrix_recall.png") if save_dir else None
        cm_recall = self.plot_confusion_matrix(
            y_true, y_pred,
            normalize='true',
            save_path=save_path2,
            title="混淆矩阵 - 召回率视角",
            figsize=(8, 6)
        )
        
        # 按预测标签归一化
        save_path3 = os.path.join(save_dir, "confusion_matrix_precision.png") if save_dir else None
        cm_precision = self.plot_confusion_matrix(
            y_true, y_pred,
            normalize='pred',
            save_path=save_path3,
            title="混淆矩阵 - 精确率视角",
            figsize=(8, 6)
        )
        
        return cm_counts, cm_recall, cm_precision
    
    def analyze_confusion_patterns(self, y_true, y_pred):
        """
        分析混淆模式
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        Returns:
            dict: 混淆模式分析结果
        """
        cm = confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)
        
        analysis = {
            'total_samples': total_samples,
            'correct_predictions': np.trace(cm),
            'accuracy': np.trace(cm) / total_samples
        }
        
        # 各类别分析
        for i, class_name in enumerate(self.class_names):
            class_total = np.sum(cm[i, :])  # 该类别的真实样本总数
            class_correct = cm[i, i]        # 该类别的正确预测数
            
            if class_total > 0:
                analysis[f'{class_name}_total'] = class_total
                analysis[f'{class_name}_correct'] = class_correct
                analysis[f'{class_name}_recall'] = class_correct / class_total
                
                # 该类别的主要混淆对象
                wrong_predictions = cm[i, :].copy()
                wrong_predictions[i] = 0  # 排除正确预测
                if np.sum(wrong_predictions) > 0:
                    most_confused_idx = np.argmax(wrong_predictions)
                    most_confused_count = wrong_predictions[most_confused_idx]
                    analysis[f'{class_name}_most_confused_with'] = self.class_names[most_confused_idx]
                    analysis[f'{class_name}_most_confused_count'] = most_confused_count
        
        # 整体混淆模式
        # 找出最容易混淆的类别对
        max_confusion = 0
        most_confused_pair = None
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and cm[i, j] > max_confusion:
                    max_confusion = cm[i, j]
                    most_confused_pair = (self.class_names[i], self.class_names[j])
        
        if most_confused_pair:
            analysis['most_confused_pair'] = most_confused_pair
            analysis['most_confused_count'] = max_confusion
        
        # 相邻类别混淆分析（针对坍落度的连续性）
        adjacent_confusion = 0
        severe_confusion = 0
        
        # 相邻混淆：0-1, 1-0, 1-2, 2-1
        adjacent_confusion += cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]
        
        # 严重混淆：0-2, 2-0
        severe_confusion += cm[0, 2] + cm[2, 0]
        
        analysis['adjacent_confusion_count'] = adjacent_confusion
        analysis['adjacent_confusion_rate'] = adjacent_confusion / total_samples
        analysis['severe_confusion_count'] = severe_confusion
        analysis['severe_confusion_rate'] = severe_confusion / total_samples
        
        return analysis
    
    def plot_confusion_analysis(self, y_true, y_pred, save_path=None):
        """
        绘制混淆分析图表
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            save_path (str): 保存路径
        """
        analysis = self.analyze_confusion_patterns(y_true, y_pred)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各类别召回率
        recalls = [analysis.get(f'{name}_recall', 0) for name in self.class_names]
        bars1 = ax1.bar(self.class_names, recalls, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('各类别召回率', fontsize=14, fontweight='bold')
        ax1.set_ylabel('召回率')
        ax1.set_ylim(0, 1)
        for bar, recall in zip(bars1, recalls):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{recall:.2%}', ha='center', va='bottom')
        
        # 2. 混淆类型分布
        confusion_types = ['正确分类', '相邻混淆', '严重混淆']
        correct = analysis['correct_predictions']
        adjacent = analysis['adjacent_confusion_count']
        severe = analysis['severe_confusion_count']
        other = analysis['total_samples'] - correct - adjacent - severe
        
        confusion_counts = [correct, adjacent, severe, other]
        confusion_labels = confusion_types + ['其他错误']
        colors = ['#2ECC71', '#F39C12', '#E74C3C', '#95A5A6']
        
        wedges, texts, autotexts = ax2.pie(confusion_counts, labels=confusion_labels, 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('混淆类型分布', fontsize=14, fontweight='bold')
        
        # 3. 各类别样本分布
        class_totals = [analysis.get(f'{name}_total', 0) for name in self.class_names]
        bars3 = ax3.bar(self.class_names, class_totals, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('各类别样本分布', fontsize=14, fontweight='bold')
        ax3.set_ylabel('样本数量')
        for bar, total in zip(bars3, class_totals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_totals)*0.01, 
                    f'{total}', ha='center', va='bottom')
        
        # 4. 混淆矩阵热力图（简化版）
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax4)
        ax4.set_title('混淆矩阵', fontsize=14, fontweight='bold')
        ax4.set_xlabel('预测标签')
        ax4.set_ylabel('真实标签')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 混淆分析图表已保存: {save_path}")
        
        plt.show()
        
        return analysis
