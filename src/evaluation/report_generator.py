"""
评估报告生成器
生成完整的模型评估报告
"""

import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.font_manager import setup_matplotlib_font, has_chinese_font, get_text_labels
from .metrics import ModelMetrics
from .confusion_matrix import ConfusionMatrixAnalyzer
from .error_analysis import ErrorAnalyzer


class ReportGenerator:
    """
    评估报告生成器
    整合所有评估结果，生成完整的评估报告
    """
    
    def __init__(self, class_names=None, model_name="VGG16"):
        """
        初始化报告生成器

        Args:
            class_names (list): 类别名称列表，如果为None则动态推断
            model_name (str): 模型名称
        """
        self.class_names = class_names  # 允许为None，在使用时动态推断
        self.num_classes = len(class_names) if class_names else None
            
        self.model_name = model_name

        # 设置字体
        setup_matplotlib_font()
        self.use_english = not has_chinese_font()

        # 初始化各个分析器（允许class_names为None）
        self.metrics_calculator = ModelMetrics(class_names)
        self.cm_analyzer = ConfusionMatrixAnalyzer(class_names)
        self.error_analyzer = ErrorAnalyzer(class_names)
    
    def generate_comprehensive_report(self, y_true, y_pred, y_prob=None, 
                                    image_paths=None, save_dir="outputs/evaluation"):
        """
        生成综合评估报告
        
        Args:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            y_prob (array, optional): 预测概率
            image_paths (list, optional): 图像路径列表
            save_dir (str): 保存目录
            
        Returns:
            dict: 完整的评估结果
        """
        print("🚀 开始生成综合评估报告...")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "confusion_matrices"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "error_analysis"), exist_ok=True)
        
        # 1. 计算所有评估指标
        print("📊 计算评估指标...")
        metrics_results = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # 2. 混淆矩阵分析
        print("🔍 分析混淆矩阵...")
        cm_save_dir = os.path.join(save_dir, "confusion_matrices")
        cm_counts, cm_recall, cm_precision = self.cm_analyzer.plot_multiple_confusion_matrices(
            y_true, y_pred, cm_save_dir
        )
        
        # 混淆模式分析
        confusion_analysis = self.cm_analyzer.analyze_confusion_patterns(y_true, y_pred)
        cm_analysis_path = os.path.join(save_dir, "confusion_matrices", "confusion_analysis.png")
        self.cm_analyzer.plot_confusion_analysis(y_true, y_pred, cm_analysis_path)
        
        # 3. 错误样本分析
        print("❌ 分析错误样本...")
        error_info = self.error_analyzer.find_error_samples(y_true, y_pred, y_prob, image_paths)
        error_analysis = self.error_analyzer.analyze_error_patterns(error_info)
        
        # 错误分析可视化
        error_analysis_path = os.path.join(save_dir, "error_analysis", "error_analysis.png")
        self.error_analyzer.plot_error_analysis(error_info, error_analysis_path)
        
        # 错误样本可视化（如果有图像路径）
        if image_paths:
            error_samples_path = os.path.join(save_dir, "error_analysis", "error_samples.png")
            self.error_analyzer.visualize_error_samples(error_info, save_path=error_samples_path)
        
        # 4. 生成性能总览图
        print("📈 生成性能总览...")
        overview_path = os.path.join(save_dir, "performance_overview.png")
        self.plot_performance_overview(metrics_results, overview_path)
        
        # 5. 整合所有结果
        comprehensive_results = {
            'model_info': {
                'model_name': self.model_name,
                'evaluation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_samples': len(y_true),
                'class_names': self.class_names
            },
            'metrics': metrics_results,
            'confusion_analysis': confusion_analysis,
            'error_analysis': error_analysis,
            'error_info': {
                'total_errors': error_info['total_errors'],
                'error_rate': error_info['error_rate']
            }
        }
        
        # 6. 保存结果到JSON文件
        results_path = os.path.join(save_dir, "reports", "evaluation_results.json")
        self.save_results_to_json(comprehensive_results, results_path)
        
        # 7. 生成文本报告
        text_report = self.generate_text_report(comprehensive_results)
        report_path = os.path.join(save_dir, "reports", "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"✅ 综合评估报告已生成完成！")
        print(f"📁 报告保存位置: {save_dir}")
        print(f"📄 主要文件:")
        print(f"   - 评估结果: {results_path}")
        print(f"   - 文本报告: {report_path}")
        print(f"   - 性能总览: {overview_path}")
        
        return comprehensive_results
    
    def plot_performance_overview(self, metrics_results, save_path):
        """
        绘制性能总览图
        
        Args:
            metrics_results (dict): 评估指标结果
            save_path (str): 保存路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 获取标签文本
        labels = get_text_labels('performance', self.use_english)

        # 1. 整体性能指标
        overall_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        overall_values = [metrics_results[metric] for metric in overall_metrics]

        if self.use_english:
            overall_labels = [labels.get('accuracy', 'Accuracy'),
                            labels.get('precision', 'Precision'),
                            labels.get('recall', 'Recall'),
                            labels.get('f1_score', 'F1-Score')]
        else:
            overall_labels = ['准确率', '精确率', '召回率', 'F1-Score']

        bars1 = ax1.bar(overall_labels, overall_values, color=['#2ECC71', '#3498DB', '#E74C3C', '#F39C12'])
        title1 = f'{self.model_name} {labels.get("overall_performance", "Overall Performance Metrics")}'
        ax1.set_title(title1, fontsize=14, fontweight='bold')
        ax1.set_ylabel(labels.get('score', 'Score'))
        ax1.set_ylim(0, 1)
        for bar, value in zip(bars1, overall_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. 各类别精确率对比
        precision_values = [metrics_results[f'precision_{name}'] for name in self.class_names]
        bars2 = ax2.bar(self.class_names, precision_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title(labels.get('class_precision', 'Class Precision'), fontsize=14, fontweight='bold')
        ax2.set_ylabel(labels.get('precision', 'Precision'))
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars2, precision_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 各类别召回率对比
        recall_values = [metrics_results[f'recall_{name}'] for name in self.class_names]
        bars3 = ax3.bar(self.class_names, recall_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title(labels.get('class_recall', 'Class Recall'), fontsize=14, fontweight='bold')
        ax3.set_ylabel(labels.get('recall', 'Recall'))
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars3, recall_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 4. 各类别F1-Score对比
        f1_values = [metrics_results[f'f1_{name}'] for name in self.class_names]
        bars4 = ax4.bar(self.class_names, f1_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title(labels.get('class_f1', 'Class F1-Score'), fontsize=14, fontweight='bold')
        ax4.set_ylabel(labels.get('f1_score', 'F1-Score'))
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars4, f1_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能总览图已保存: {save_path}")
        plt.show()
    
    def save_results_to_json(self, results, save_path):
        """
        保存结果到JSON文件
        
        Args:
            results (dict): 评估结果
            save_path (str): 保存路径
        """
        # 处理numpy数组，转换为列表
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"💾 评估结果已保存到JSON: {save_path}")
    
    def generate_text_report(self, results):
        """
        生成文本格式的评估报告
        
        Args:
            results (dict): 评估结果
            
        Returns:
            str: 文本报告
        """
        report = f"""
{'='*80}
🏗️  {results['model_info']['model_name']} 混凝土坍落度分类模型评估报告
{'='*80}

📅 评估时间: {results['model_info']['evaluation_date']}
📊 总样本数: {results['model_info']['total_samples']}
🏷️  分类类别: {', '.join(results['model_info']['class_names'])}

{'='*80}
📈 整体性能指标
{'='*80}

准确率 (Accuracy):           {results['metrics']['accuracy']:.4f}
宏平均精确率 (Precision):     {results['metrics']['precision_macro']:.4f}
宏平均召回率 (Recall):        {results['metrics']['recall_macro']:.4f}
宏平均F1-Score:              {results['metrics']['f1_macro']:.4f}

加权平均精确率:               {results['metrics']['precision_weighted']:.4f}
加权平均召回率:               {results['metrics']['recall_weighted']:.4f}
加权平均F1-Score:            {results['metrics']['f1_weighted']:.4f}"""

        if 'top2_accuracy' in results['metrics']:
            report += f"\nTop-2准确率:                 {results['metrics']['top2_accuracy']:.4f}"

        report += f"""

{'='*80}
🎯 各类别详细性能
{'='*80}
"""

        for class_name in self.class_names:
            precision = results['metrics'][f'precision_{class_name}']
            recall = results['metrics'][f'recall_{class_name}']
            f1 = results['metrics'][f'f1_{class_name}']
            
            report += f"""
{class_name}:
  精确率: {precision:.4f}
  召回率: {recall:.4f}
  F1-Score: {f1:.4f}"""

        report += f"""

{'='*80}
🔍 混淆分析
{'='*80}

总样本数: {results['confusion_analysis']['total_samples']}
正确预测: {results['confusion_analysis']['correct_predictions']}
整体准确率: {results['confusion_analysis']['accuracy']:.4f}

相邻类别混淆率: {results['metrics']['adjacent_confusion_rate']:.4f}
严重错误率: {results['metrics']['severe_error_rate']:.4f}"""

        if 'most_confused_pair' in results['confusion_analysis']:
            pair = results['confusion_analysis']['most_confused_pair']
            count = results['confusion_analysis']['most_confused_count']
            report += f"\n最易混淆类别对: {pair[0]} ↔ {pair[1]} ({count}次)"

        report += f"""

{'='*80}
❌ 错误样本分析
{'='*80}

总错误数: {results['error_info']['total_errors']}
错误率: {results['error_info']['error_rate']:.4f}"""

        if results['error_info']['total_errors'] > 0:
            report += f"""
相邻类别错误: {results['error_analysis']['adjacent_errors']}
严重错误: {results['error_analysis']['severe_errors']}"""

            if 'mean_error_confidence' in results['error_analysis']:
                report += f"""
错误样本平均置信度: {results['error_analysis']['mean_error_confidence']:.4f}"""

        report += f"""

{'='*80}
📋 分类报告
{'='*80}

{results['metrics']['classification_report']}

{'='*80}
✅ 评估完成
{'='*80}
"""

        return report
