"""
错误样本分析模块
分析和可视化模型的错误预测样本
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn.functional as F


class ErrorAnalyzer:
    """
    错误样本分析器
    提供错误样本的统计、分析和可视化功能
    """
    
    def __init__(self, class_names=None):
        """
        初始化错误样本分析器
        
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
    
    def find_error_samples(self, y_true, y_pred, y_prob=None, image_paths=None):
        """
        找出所有错误预测的样本
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            y_prob (array, optional): 预测概率
            image_paths (list, optional): 图像路径列表
            
        Returns:
            dict: 错误样本信息
        """
        # 找出错误预测的索引
        error_indices = np.where(y_true != y_pred)[0]
        
        error_info = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'error_indices': error_indices,
            'true_labels': y_true[error_indices],
            'pred_labels': y_pred[error_indices]
        }
        
        if y_prob is not None:
            error_info['pred_probs'] = y_prob[error_indices]
            error_info['confidence'] = np.max(y_prob[error_indices], axis=1)
        
        if image_paths is not None:
            error_info['image_paths'] = [image_paths[i] for i in error_indices]
        
        return error_info
    
    def categorize_errors(self, error_info):
        """
        对错误进行分类
        
        Args:
            error_info (dict): 错误样本信息
            
        Returns:
            dict: 分类后的错误信息
        """
        true_labels = error_info['true_labels']
        pred_labels = error_info['pred_labels']
        
        categories = {
            'adjacent_errors': [],      # 相邻类别错误
            'severe_errors': [],        # 跨类别严重错误
            'high_confidence_errors': [], # 高置信度错误
            'low_confidence_errors': []   # 低置信度错误
        }
        
        for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            # 相邻类别错误 vs 严重错误
            if abs(true_label - pred_label) == 1:
                categories['adjacent_errors'].append(i)
            else:
                categories['severe_errors'].append(i)
            
            # 高低置信度错误（如果有概率信息）
            if 'confidence' in error_info:
                confidence = error_info['confidence'][i]
                if confidence > 0.8:
                    categories['high_confidence_errors'].append(i)
                else:
                    categories['low_confidence_errors'].append(i)
        
        return categories
    
    def analyze_error_patterns(self, error_info):
        """
        分析错误模式
        
        Args:
            error_info (dict): 错误样本信息
            
        Returns:
            dict: 错误模式分析结果
        """
        if error_info['total_errors'] == 0:
            return {'message': '没有错误样本'}
        
        true_labels = error_info['true_labels']
        pred_labels = error_info['pred_labels']
        
        analysis = {
            'total_errors': error_info['total_errors'],
            'error_rate': error_info['error_rate']
        }
        
        # 各类别的错误分析
        for class_idx, class_name in enumerate(self.class_names):
            class_errors = np.sum(true_labels == class_idx)
            analysis[f'{class_name}_errors'] = class_errors
            
            if class_errors > 0:
                # 该类别最常被误分为哪个类别
                class_error_mask = (true_labels == class_idx)
                misclassified_as = pred_labels[class_error_mask]
                unique, counts = np.unique(misclassified_as, return_counts=True)
                most_common_idx = unique[np.argmax(counts)]
                analysis[f'{class_name}_most_misclassified_as'] = self.class_names[most_common_idx]
                analysis[f'{class_name}_most_misclassified_count'] = np.max(counts)
        
        # 错误类型统计
        categories = self.categorize_errors(error_info)
        analysis['adjacent_errors'] = len(categories['adjacent_errors'])
        analysis['severe_errors'] = len(categories['severe_errors'])
        analysis['adjacent_error_rate'] = len(categories['adjacent_errors']) / error_info['total_errors']
        analysis['severe_error_rate'] = len(categories['severe_errors']) / error_info['total_errors']
        
        # 置信度分析（如果有）
        if 'confidence' in error_info:
            confidence = error_info['confidence']
            analysis['mean_error_confidence'] = np.mean(confidence)
            analysis['std_error_confidence'] = np.std(confidence)
            analysis['high_confidence_errors'] = len(categories['high_confidence_errors'])
            analysis['low_confidence_errors'] = len(categories['low_confidence_errors'])
        
        return analysis
    
    def plot_error_analysis(self, error_info, save_path=None):
        """
        绘制错误分析图表
        
        Args:
            error_info (dict): 错误样本信息
            save_path (str): 保存路径
        """
        if error_info['total_errors'] == 0:
            print("没有错误样本，无需绘制分析图表")
            return
        
        analysis = self.analyze_error_patterns(error_info)
        categories = self.categorize_errors(error_info)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各类别错误数量
        class_errors = [analysis.get(f'{name}_errors', 0) for name in self.class_names]
        bars1 = axes[0, 0].bar(self.class_names, class_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('各类别错误数量', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('错误数量')
        for bar, count in zip(bars1, class_errors):
            if count > 0:
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_errors)*0.01, 
                               f'{count}', ha='center', va='bottom')
        
        # 2. 错误类型分布
        error_types = ['相邻类别错误', '严重错误']
        error_counts = [analysis['adjacent_errors'], analysis['severe_errors']]
        colors = ['#F39C12', '#E74C3C']
        
        wedges, texts, autotexts = axes[0, 1].pie(error_counts, labels=error_types, 
                                                 colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('错误类型分布', fontsize=14, fontweight='bold')
        
        # 3. 置信度分布（如果有）
        if 'confidence' in error_info:
            confidence = error_info['confidence']
            axes[1, 0].hist(confidence, bins=20, color='#9B59B6', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(np.mean(confidence), color='red', linestyle='--', 
                              label=f'平均值: {np.mean(confidence):.3f}')
            axes[1, 0].set_title('错误样本置信度分布', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('置信度')
            axes[1, 0].set_ylabel('频次')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, '无置信度信息', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=16)
            axes[1, 0].set_title('置信度分布', fontsize=14, fontweight='bold')
        
        # 4. 错误转移矩阵（真实->预测）
        true_labels = error_info['true_labels']
        pred_labels = error_info['pred_labels']
        
        # 创建错误转移矩阵
        error_matrix = np.zeros((len(self.class_names), len(self.class_names)))
        for true_label, pred_label in zip(true_labels, pred_labels):
            error_matrix[true_label, pred_label] += 1
        
        im = axes[1, 1].imshow(error_matrix, cmap='Reds', aspect='auto')
        axes[1, 1].set_title('错误转移矩阵', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('预测标签')
        axes[1, 1].set_ylabel('真实标签')
        axes[1, 1].set_xticks(range(len(self.class_names)))
        axes[1, 1].set_yticks(range(len(self.class_names)))
        axes[1, 1].set_xticklabels(self.class_names, rotation=45)
        axes[1, 1].set_yticklabels(self.class_names)
        
        # 添加数值标注
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if error_matrix[i, j] > 0:
                    axes[1, 1].text(j, i, f'{int(error_matrix[i, j])}', 
                                   ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1])
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 错误分析图表已保存: {save_path}")
        
        plt.show()
        
        return analysis
    
    def visualize_error_samples(self, error_info, max_samples=16, save_path=None):
        """
        可视化错误样本图像
        
        Args:
            error_info (dict): 错误样本信息
            max_samples (int): 最大显示样本数
            save_path (str): 保存路径
        """
        if 'image_paths' not in error_info:
            print("没有图像路径信息，无法可视化错误样本")
            return
        
        if error_info['total_errors'] == 0:
            print("没有错误样本")
            return
        
        # 选择要显示的样本
        num_samples = min(max_samples, error_info['total_errors'])
        sample_indices = np.random.choice(error_info['total_errors'], num_samples, replace=False)
        
        # 计算网格大小
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_idx in enumerate(sample_indices):
            row = idx // cols
            col = idx % cols
            
            # 加载图像
            img_path = error_info['image_paths'][sample_idx]
            try:
                img = Image.open(img_path).convert('RGB')
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # 添加标题信息
                true_label = error_info['true_labels'][sample_idx]
                pred_label = error_info['pred_labels'][sample_idx]
                
                title = f"真实: {self.class_names[true_label]}\n预测: {self.class_names[pred_label]}"
                
                if 'confidence' in error_info:
                    confidence = error_info['confidence'][sample_idx]
                    title += f"\n置信度: {confidence:.3f}"
                
                axes[row, col].set_title(title, fontsize=10, pad=10)
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f"无法加载图像\n{str(e)}", 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(f"错误样本 {sample_idx}", fontsize=10)
        
        # 隐藏多余的子图
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'错误样本可视化 (共{error_info["total_errors"]}个错误，显示{num_samples}个)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 错误样本可视化已保存: {save_path}")
        
        plt.show()
    
    def generate_error_report(self, error_info):
        """
        生成错误分析报告
        
        Args:
            error_info (dict): 错误样本信息
            
        Returns:
            str: 错误分析报告
        """
        if error_info['total_errors'] == 0:
            return "🎉 模型预测完全正确，没有错误样本！"
        
        analysis = self.analyze_error_patterns(error_info)
        
        report = f"""
📊 错误样本分析报告
{'='*50}

📈 总体统计:
  总错误数: {analysis['total_errors']}
  错误率: {analysis['error_rate']:.2%}

🔍 错误类型分析:
  相邻类别错误: {analysis['adjacent_errors']} ({analysis['adjacent_error_rate']:.1%})
  严重错误: {analysis['severe_errors']} ({analysis['severe_error_rate']:.1%})

📋 各类别错误详情:"""
        
        for class_name in self.class_names:
            errors = analysis.get(f'{class_name}_errors', 0)
            if errors > 0:
                most_confused = analysis.get(f'{class_name}_most_misclassified_as', '未知')
                confused_count = analysis.get(f'{class_name}_most_misclassified_count', 0)
                report += f"""
  {class_name}:
    错误数: {errors}
    最常误分为: {most_confused} ({confused_count}次)"""
        
        if 'mean_error_confidence' in analysis:
            report += f"""

🎯 置信度分析:
  错误样本平均置信度: {analysis['mean_error_confidence']:.3f}
  置信度标准差: {analysis['std_error_confidence']:.3f}
  高置信度错误: {analysis['high_confidence_errors']}
  低置信度错误: {analysis['low_confidence_errors']}"""
        
        return report
