"""
模型评估指标计算模块
包含精确率、召回率、F1-Score等核心指标的计算
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
import torch
import torch.nn.functional as F


class ModelMetrics:
    """
    模型评估指标计算器
    专门针对混凝土坍落度三分类任务优化
    """
    
    def __init__(self, class_names=None):
        """
        初始化评估指标计算器
        
        Args:
            class_names (list): 类别名称列表，默认为坍落度范围
        """
        if class_names is None:
            self.class_names = ["125-175mm", "180-230mm", "233-285mm"]
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
        
    def calculate_basic_metrics(self, y_true, y_pred):
        """
        计算基础分类指标
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        Returns:
            dict: 包含各种指标的字典
        """
        metrics = {}
        
        # 整体准确率
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 各类别和整体的精确率、召回率、F1-Score
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # 加权平均（考虑类别不平衡）
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # 各类别详细指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def calculate_confusion_matrix(self, y_true, y_pred):
        """
        计算混淆矩阵
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        Returns:
            numpy.ndarray: 混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)
    
    def calculate_top_k_accuracy(self, y_true, y_prob, k=2):
        """
        计算Top-K准确率
        
        Args:
            y_true (array): 真实标签
            y_prob (array): 预测概率矩阵
            k (int): Top-K的K值
            
        Returns:
            float: Top-K准确率
        """
        return top_k_accuracy_score(y_true, y_prob, k=k)
    
    def calculate_confidence_stats(self, y_prob):
        """
        计算置信度统计信息
        
        Args:
            y_prob (array): 预测概率矩阵
            
        Returns:
            dict: 置信度统计信息
        """
        # 最大概率作为置信度
        max_probs = np.max(y_prob, axis=1)
        
        stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'median_confidence': np.median(max_probs)
        }
        
        # 置信度分布统计
        confidence_ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in confidence_ranges:
            count = np.sum((max_probs >= low) & (max_probs < high))
            stats[f'confidence_{low}-{high}'] = count / len(max_probs)
        
        return stats
    
    def analyze_adjacent_class_confusion(self, y_true, y_pred):
        """
        分析相邻类别混淆情况
        专门针对坍落度范围的连续性特点
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        Returns:
            dict: 相邻类别混淆分析结果
        """
        cm = confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)
        
        analysis = {}
        
        # 相邻类别混淆率（0-1, 1-2）
        adjacent_errors = cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]
        analysis['adjacent_confusion_rate'] = adjacent_errors / total_samples
        
        # 跨类别严重错误率（0-2, 2-0）
        severe_errors = cm[0, 2] + cm[2, 0]
        analysis['severe_error_rate'] = severe_errors / total_samples
        
        # 各类别的相邻混淆详情
        for i, class_name in enumerate(self.class_names):
            class_total = np.sum(cm[i, :])
            if class_total > 0:
                if i == 0:  # 第一类只有右邻居
                    adjacent_error = cm[i, i+1]
                elif i == len(self.class_names) - 1:  # 最后一类只有左邻居
                    adjacent_error = cm[i, i-1]
                else:  # 中间类有左右邻居
                    adjacent_error = cm[i, i-1] + cm[i, i+1]
                
                analysis[f'{class_name}_adjacent_error_rate'] = adjacent_error / class_total
        
        return analysis
    
    def generate_classification_report(self, y_true, y_pred):
        """
        生成详细的分类报告
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        Returns:
            str: 分类报告字符串
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None):
        """
        计算所有评估指标
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            y_prob (array, optional): 预测概率矩阵
            
        Returns:
            dict: 包含所有指标的综合结果
        """
        results = {}
        
        # 基础指标
        results.update(self.calculate_basic_metrics(y_true, y_pred))
        
        # 混淆矩阵
        results['confusion_matrix'] = self.calculate_confusion_matrix(y_true, y_pred)
        
        # 相邻类别混淆分析
        results.update(self.analyze_adjacent_class_confusion(y_true, y_pred))
        
        # 分类报告
        results['classification_report'] = self.generate_classification_report(y_true, y_pred)
        
        # 如果提供了概率，计算相关指标
        if y_prob is not None:
            results['top2_accuracy'] = self.calculate_top_k_accuracy(y_true, y_prob, k=2)
            results.update(self.calculate_confidence_stats(y_prob))
        
        return results
