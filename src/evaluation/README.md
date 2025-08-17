# 模型评估模块

专门为混凝土坍落度分类模型设计的综合评估系统，提供多维度性能分析和可视化功能。

## 📋 模块概述

本评估模块包含四个核心组件：

- **`metrics.py`** - 核心评估指标计算
- **`confusion_matrix.py`** - 混淆矩阵分析和可视化  
- **`error_analysis.py`** - 错误样本分析
- **`report_generator.py`** - 综合评估报告生成

## 🎯 主要功能

### 1. 多维度评估指标
- **基础指标**: 准确率、精确率、召回率、F1-Score
- **各类别指标**: 每个坍落度范围的详细性能
- **Top-K准确率**: 预测前K个结果的准确性
- **置信度分析**: 模型预测可靠性评估

### 2. 混淆矩阵分析
- **多种视角**: 样本数量、召回率、精确率视角
- **模式识别**: 自动识别易混淆的类别对
- **相邻分析**: 针对坍落度连续性的特殊分析
- **可视化**: 热力图和统计图表

### 3. 错误样本分析
- **错误分类**: 相邻类别错误 vs 严重错误
- **置信度分析**: 高低置信度错误分布
- **样本可视化**: 错误样本图像展示
- **模式总结**: 错误规律和改进建议

### 4. 综合报告生成
- **完整报告**: JSON格式的详细结果
- **文本报告**: 易读的评估总结
- **可视化图表**: 性能总览和分析图
- **文件管理**: 自动组织保存结果

## 🚀 使用方法

### 基础使用

```python
from src.evaluation import ModelMetrics, ConfusionMatrixAnalyzer, ErrorAnalyzer, ReportGenerator

# 初始化评估器
class_names = ["125-175mm", "180-230mm", "233-285mm"]
report_gen = ReportGenerator(class_names, model_name="VGG16")

# 生成综合评估报告
results = report_gen.generate_comprehensive_report(
    y_true=true_labels,
    y_pred=predicted_labels,
    y_prob=prediction_probabilities,  # 可选
    image_paths=image_file_paths,     # 可选
    save_dir="outputs/evaluation"
)
```

### 单独使用各组件

```python
# 1. 计算评估指标
metrics = ModelMetrics(class_names)
metrics_results = metrics.calculate_all_metrics(y_true, y_pred, y_prob)

# 2. 混淆矩阵分析
cm_analyzer = ConfusionMatrixAnalyzer(class_names)
cm_analyzer.plot_multiple_confusion_matrices(y_true, y_pred, "outputs/cm/")

# 3. 错误样本分析
error_analyzer = ErrorAnalyzer(class_names)
error_info = error_analyzer.find_error_samples(y_true, y_pred, y_prob, image_paths)
error_analyzer.plot_error_analysis(error_info, "outputs/error_analysis.png")
```

## 📊 输出结果

### 文件结构
```
outputs/evaluation/
├── reports/
│   ├── evaluation_results.json    # 完整评估结果
│   └── evaluation_report.txt      # 文本格式报告
├── confusion_matrices/
│   ├── confusion_matrix_counts.png     # 样本数量矩阵
│   ├── confusion_matrix_recall.png     # 召回率矩阵
│   ├── confusion_matrix_precision.png  # 精确率矩阵
│   └── confusion_analysis.png          # 混淆分析图
├── error_analysis/
│   ├── error_analysis.png         # 错误分析图表
│   └── error_samples.png          # 错误样本可视化
└── performance_overview.png       # 性能总览图
```

### 关键指标说明

#### 针对混凝土坍落度的特殊指标：
- **相邻类别混淆率**: 相邻坍落度范围间的误分类率
- **严重错误率**: 跨范围的严重误分类率
- **各范围识别率**: 每个坍落度范围的识别准确性

#### 实用性指标：
- **Top-2准确率**: 前两个预测包含正确答案的比例
- **置信度分布**: 模型预测的可靠性分析
- **错误样本统计**: 便于针对性改进的错误分析

## 🎨 可视化特色

### 1. 专业图表
- 彩色热力图显示混淆矩阵
- 多维度柱状图对比性能
- 饼图展示错误类型分布

### 2. 中文支持
- 完整的中文标签和说明
- 适合中文环境的字体设置
- 清晰的图表标题和注释

### 3. 高质量输出
- 300 DPI高分辨率图像
- 专业的配色方案
- 自动布局优化

## 🔧 自定义配置

### 修改类别名称
```python
custom_class_names = ["低坍落度", "中坍落度", "高坍落度"]
report_gen = ReportGenerator(custom_class_names)
```

### 调整可视化参数
```python
# 错误样本可视化数量
error_analyzer.visualize_error_samples(error_info, max_samples=20)

# 混淆矩阵图像大小
cm_analyzer.plot_confusion_matrix(y_true, y_pred, figsize=(12, 10))
```

## 📝 注意事项

1. **数据格式**: 确保标签为整数格式 (0, 1, 2)
2. **概率矩阵**: 如需置信度分析，请提供预测概率
3. **图像路径**: 错误样本可视化需要提供图像文件路径
4. **内存使用**: 大量样本时建议分批处理
5. **中文字体**: 如图表中文显示异常，请安装相应字体

## 🎯 应用场景

- **模型验证**: 训练完成后的性能评估
- **模型对比**: 不同模型或参数的效果比较  
- **错误分析**: 识别模型弱点，指导改进方向
- **报告生成**: 生成专业的评估文档
- **质量控制**: 确保模型达到工程应用标准
