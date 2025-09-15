#!/usr/bin/env python3
"""
测试评估功能增强
验证各类别准确率计算和单独图表生成功能
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.evaluation import ModelMetrics, ReportGenerator

def test_per_class_accuracy():
    """测试各类别准确率计算"""
    print("🧪 测试各类别准确率计算...")
    
    # 创建测试数据
    class_names = ["类别1_120-130", "类别2_131-135", "类别3_136-140"]
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2, 0, 1, 1])  # 故意制造一些错误
    
    # 创建指标计算器
    metrics_calculator = ModelMetrics(class_names)
    
    # 计算指标
    results = metrics_calculator.calculate_basic_metrics(y_true, y_pred)
    
    print("📊 各类别指标:")
    for class_name in class_names:
        precision = results[f'precision_{class_name}']
        recall = results[f'recall_{class_name}']
        f1 = results[f'f1_{class_name}']
        accuracy = results[f'accuracy_{class_name}']
        print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, A={accuracy:.3f}")
    
    print("✅ 各类别准确率计算测试完成")

def test_individual_plots():
    """测试单独图表生成"""
    print("🧪 测试单独图表生成...")
    
    # 创建测试数据
    class_names = ["类别1_120-130", "类别2_131-135", "类别3_136-140"]
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2, 0, 1, 1])
    
    # 创建报告生成器
    report_generator = ReportGenerator(class_names, "TEST_MODEL")
    
    # 计算指标
    metrics_calculator = ModelMetrics(class_names)
    metrics_results = metrics_calculator.calculate_basic_metrics(y_true, y_pred)
    
    # 创建测试目录
    test_dir = "test_evaluation_output"
    os.makedirs(test_dir, exist_ok=True)
    plots_dir = os.path.join(test_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 生成各类别图表
    print("📊 生成各类别精确率图表...")
    precision_path = os.path.join(plots_dir, "class_precision.png")
    report_generator.plot_class_precision(metrics_results, precision_path)
    
    print("📊 生成各类别准确率图表...")
    accuracy_path = os.path.join(plots_dir, "class_accuracy.png")
    report_generator.plot_class_accuracy(metrics_results, accuracy_path)
    
    print("📊 生成各类别召回率图表...")
    recall_path = os.path.join(plots_dir, "class_recall.png")
    report_generator.plot_class_recall(metrics_results, recall_path)
    
    print("📊 生成各类别F1分数图表...")
    f1_path = os.path.join(plots_dir, "class_f1_score.png")
    report_generator.plot_class_f1_score(metrics_results, f1_path)
    
    print(f"✅ 单独图表生成测试完成，结果保存在: {test_dir}")

def main():
    """主测试函数"""
    print("🚀 开始测试评估功能增强...")
    print("="*60)
    
    try:
        # 测试各类别准确率计算
        test_per_class_accuracy()
        print()
        
        # 测试单独图表生成
        test_individual_plots()
        print()
        
        print("="*60)
        print("🎉 所有测试完成！")
        print("📁 测试结果保存在 test_evaluation_output 目录中")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
