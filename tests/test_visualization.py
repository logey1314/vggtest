"""
测试可视化功能的脚本
用于验证新的可视化模块是否正常工作
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.visualization import (
    plot_enhanced_training_history,
    plot_confusion_matrices, 
    plot_roc_curves,
    plot_pr_curves,
    plot_class_performance_radar
)

def generate_test_data():
    """生成测试数据"""
    # 模拟训练历史数据
    epochs = 20
    train_losses = np.random.exponential(0.5, epochs) + 0.1
    val_losses = train_losses + np.random.normal(0, 0.1, epochs)
    train_accuracies = 100 * (1 - train_losses / 2)
    val_accuracies = train_accuracies - np.random.uniform(0, 10, epochs)
    
    # 模拟评估结果
    n_samples = 1000
    n_classes = 3
    class_names = ["class1_125-175", "class2_180-230", "class3_233-285"]
    
    # 生成模拟的预测和标签
    labels = np.random.randint(0, n_classes, n_samples)
    predictions = labels.copy()
    # 添加一些错误预测
    error_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
    predictions[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    # 生成模拟的概率
    probabilities = np.random.dirichlet([1, 1, 1], n_samples)
    
    # 生成模拟的分类报告
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(labels, predictions)
    cm_normalized = confusion_matrix(labels, predictions, normalize='true')
    
    evaluation_results = {
        'predictions': predictions,
        'labels': labels,
        'probabilities': probabilities,
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'class_names': class_names
    }
    
    return train_losses, val_losses, train_accuracies, val_accuracies, evaluation_results

def test_visualization():
    """测试所有可视化功能"""
    print("🧪 开始测试可视化功能...")
    
    # 创建测试输出目录
    test_output_dir = "outputs/plots/test"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 生成测试数据
    train_losses, val_losses, train_accuracies, val_accuracies, evaluation_results = generate_test_data()
    
    try:
        # 1. 测试训练历史图表
        print("📈 测试训练历史图表...")
        plot_enhanced_training_history(
            train_losses, val_losses, train_accuracies, val_accuracies,
            os.path.join(test_output_dir, "test_training_history.png")
        )
        
        # 2. 测试混淆矩阵
        print("📊 测试混淆矩阵...")
        plot_confusion_matrices(evaluation_results, test_output_dir)
        
        # 3. 测试ROC曲线
        print("📈 测试ROC曲线...")
        plot_roc_curves(evaluation_results, os.path.join(test_output_dir, "test_roc_curves.png"))
        
        # 4. 测试PR曲线
        print("📈 测试PR曲线...")
        plot_pr_curves(evaluation_results, os.path.join(test_output_dir, "test_pr_curves.png"))
        
        # 5. 测试雷达图
        print("🎯 测试类别性能雷达图...")
        plot_class_performance_radar(evaluation_results, os.path.join(test_output_dir, "test_class_performance_radar.png"))
        
        print("✅ 所有可视化功能测试通过!")
        print(f"📁 测试图表保存在: {test_output_dir}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization()
