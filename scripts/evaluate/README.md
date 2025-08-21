# 评估模块 (scripts/evaluate)

本模块提供完整的模型评估功能，支持指定训练轮次进行评估，并生成详细的分析报告。

## 📁 模块结构

```
scripts/evaluate/
├── __init__.py              # 模块初始化文件
├── evaluate_model.py        # 主评估脚本
├── evaluate_utils.py        # 评估工具函数
└── README.md               # 本说明文档
```

## 🚀 主要功能

### 1. 指定轮次评估
- 支持指定具体的训练轮次路径进行评估
- 自动提取时间戳并生成对应的评估结果文件夹
- 自动查找对应的模型文件（best_model.pth 或检查点文件）

### 2. 默认评估
- 支持使用 latest 目录中的最新模型进行评估
- 生成带时间戳的评估结果文件夹

### 3. 全面的评估报告
- 生成详细的性能指标分析
- 混淆矩阵和错误样本分析
- 可视化图表和统计报告
- 同时保存到时间戳目录和 latest 目录

## 📋 使用方法

### 方法1：指定训练轮次评估（推荐）

1. **打开 `scripts/evaluate/evaluate_model.py`**

2. **修改 main 函数中的配置**：
   ```python
   # 🎯 指定要评估的训练轮次路径（在这里修改）
   SPECIFIC_TRAIN_PATH = "/var/yjs/zes/vgg/outputs/logs/train_20250820_074155/"
   ```

3. **在 PyCharm 中直接运行**：
   - 右键点击 `evaluate_model.py` → Run
   - 或者在文件中点击 main 函数旁的运行按钮

4. **查看结果**：
   - 评估结果保存在 `outputs/evaluation/eval_20250820_074155/`
   - 同时保存到 `outputs/evaluation/latest/`

### 方法2：使用默认模型评估

1. **修改配置**：
   ```python
   # 使用默认的 latest 模型
   SPECIFIC_TRAIN_PATH = None
   ```

2. **运行脚本**，结果保存在带当前时间戳的文件夹中

## 🎯 路径配置说明

### 支持的训练路径格式

```python
# 完整路径
SPECIFIC_TRAIN_PATH = "/var/yjs/zes/vgg/outputs/logs/train_20250820_074155/"

# 相对路径
SPECIFIC_TRAIN_PATH = "outputs/logs/train_20250820_074155/"

# 只有目录名
SPECIFIC_TRAIN_PATH = "train_20250820_074155"

# 恢复训练的路径
SPECIFIC_TRAIN_PATH = "/var/yjs/zes/vgg/outputs/logs/resume_20250820_074155/"
```

### 自动查找的模型文件

脚本会按以下优先级查找模型文件：

1. `models/checkpoints/train_20250820_074155/best_model.pth`
2. `models/checkpoints/train_20250820_074155/checkpoint_epoch_*.pth`（最新的）
3. `models/checkpoints/resume_20250820_074155/best_model.pth`

### 配置文件使用逻辑

**重要：** 评估脚本会自动使用训练时保存的配置文件，确保评估参数与训练时一致。

**配置文件优先级：**
1. **训练时配置**（优先）：`outputs/logs/train_20250820_074155/config_backup.yaml`
2. **默认配置**（备用）：`configs/training_config.yaml`

**为什么这很重要：**
- 确保模型结构参数（如类别数量）与训练时一致
- 保证数据预处理方式与训练时相同
- 提高评估结果的准确性和可靠性

## 📊 评估结果说明

### 生成的文件结构

```
outputs/evaluation/eval_20250820_074155/
├── comprehensive_report.html        # 综合评估报告（HTML格式）
├── metrics_summary.json            # 性能指标摘要（JSON格式）
├── confusion_matrix.png            # 混淆矩阵图
├── classification_report.txt       # 分类报告（文本格式）
├── error_analysis/                 # 错误分析文件夹
│   ├── error_samples.csv           # 错误样本列表
│   ├── error_distribution.png      # 错误分布图
│   └── misclassified_samples/      # 错误样本图像
└── performance_charts/             # 性能图表文件夹
    ├── precision_recall_curve.png  # PR曲线
    ├── roc_curve.png               # ROC曲线
    └── performance_radar.png       # 性能雷达图
```

### 关键指标说明

- **整体准确率**: 所有样本的预测准确率
- **宏平均F1-Score**: 各类别F1分数的平均值
- **Top-2准确率**: 前两个预测中包含正确答案的比例
- **相邻类别混淆率**: 预测为相邻类别的错误比例
- **严重错误率**: 预测差距较大的错误比例

## 🔧 工具函数说明

### evaluate_utils.py 中的主要函数

```python
# 从路径中提取时间戳
timestamp = extract_timestamp_from_path("/path/to/train_20250820_074155/")

# 验证训练路径是否有效
is_valid = validate_train_path(train_path)

# 查找对应的模型文件
model_path = find_model_file(project_root, timestamp)

# 获取完整的模型路径
model_path = get_corresponding_model_path(train_path, project_root)
```

## ⚠️ 注意事项

1. **路径格式**: 确保训练路径包含正确的时间戳格式 `YYYYMMDD_HHMMSS`

2. **模型文件**: 确保对应的模型文件存在，脚本会自动查找但需要文件确实存在

3. **配置文件**: 脚本会自动使用训练时保存的配置文件，无需手动指定

4. **数据集**: 确保标注文件 `data/annotations/cls_train.txt` 存在且格式正确

5. **设备兼容**: 脚本会自动检测 GPU/CPU，但确保模型文件与当前设备兼容

## 🐛 常见问题

### Q1: 提示"无法从路径中提取时间戳"
**A**: 检查路径是否包含正确的时间戳格式，如 `train_20250820_074155`

### Q2: 提示"未找到对应的模型文件"
**A**: 检查以下位置是否存在模型文件：
- `models/checkpoints/train_YYYYMMDD_HHMMSS/best_model.pth`
- `models/checkpoints/train_YYYYMMDD_HHMMSS/checkpoint_epoch_*.pth`

### Q3: 提示"未找到训练时配置文件"
**A**: 检查以下位置是否存在配置备份：
- `outputs/logs/train_YYYYMMDD_HHMMSS/config_backup.yaml`
- `outputs/logs/resume_YYYYMMDD_HHMMSS/config_backup.yaml`
- 如果没有备份，脚本会使用默认配置并给出警告

### Q4: 评估结果准确率异常
**A**: 检查：
- 是否使用了正确的训练时配置文件
- 配置文件中的类别数量是否与模型一致
- 数据集是否与训练时使用的相同
- 模型文件是否损坏

### Q5: 无法生成可视化图表
**A**: 检查：
- 是否安装了 matplotlib 和相关依赖
- 字体设置是否正确
- 输出目录是否有写入权限

## 💡 使用技巧

1. **批量评估**: 可以修改脚本循环评估多个训练轮次
2. **结果对比**: 使用不同的时间戳文件夹来对比不同模型的性能
3. **错误分析**: 重点关注 `error_analysis` 文件夹中的错误样本分析
4. **性能监控**: 定期评估最新训练的模型，监控性能变化

## 🔄 与其他模块的关系

- **训练模块**: 评估训练产生的模型文件
- **可视化模块**: 使用 `src/utils/visualization.py` 生成图表
- **数据模块**: 使用 `src/data/dataset.py` 加载评估数据
- **评估模块**: 使用 `src/evaluation/` 中的各种评估工具
