# VGG16 图像分类项目

基于 PyTorch 和 VGG16 网络的三分类图像识别项目，具有详细的训练日志和可视化功能。

## 🚀 功能特点

- ✅ 基于预训练 VGG16 网络的迁移学习
- ✅ 支持三分类任务
- ✅ 详细的训练进度显示（类似YOLO训练风格）
- ✅ 实时损失和精度监控
- ✅ 自动保存最佳模型
- ✅ **自动生成6张专业分析图表**（训练历史、混淆矩阵、ROC曲线、PR曲线、雷达图）
- ✅ 单张图像和批量预测
- ✅ 数据增强（翻转、旋转、色域变换等）
- ✅ 完整的模型评估系统（混淆矩阵、错误分析、性能报告）
- ✅ **零训练开销的可视化系统**（只在训练结束后生成图表）
- ✅ **检查点恢复训练功能**（从任意检查点继续训练，避免重新开始）

## 📁 项目结构

```
vgg_test/
├── README.md
├── requirements.txt
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── models/                 # 模型定义
│   │   ├── __init__.py
│   │   └── vgg.py             # VGG16 网络定义
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   └── dataset.py         # 数据集处理模块
│   ├── training/               # 训练相关
│   │   └── __init__.py
│   ├── evaluation/             # 模型评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py         # 评估指标计算
│   │   ├── confusion_matrix.py # 混淆矩阵分析
│   │   ├── error_analysis.py  # 错误样本分析
│   │   └── report_generator.py # 评估报告生成
│   └── utils/                  # 通用工具
│       ├── __init__.py
│       ├── visualization.py   # 可视化工具
│       └── checkpoint_manager.py # 检查点管理工具
│
├── scripts/                    # 执行脚本
│   ├── train.py               # 主训练脚本
│   ├── resume_training.py     # 恢复训练脚本
│   ├── predict.py             # 预测脚本
│   ├── evaluate_model.py      # 模型评估脚本
│   ├── visualize_training.py  # 训练可视化脚本
│   └── generate_annotations.py # 数据集标注生成器
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   │   ├── class1_125-175/    # 类别1图像
│   │   ├── class2_180-230/    # 类别2图像
│   │   └── class3_233-285/    # 类别3图像
│   ├── processed/             # 处理后的数据
│   └── annotations/           # 标注文件
│       └── cls_train.txt      # 训练数据标注文件
│
├── models/                     # 模型文件
│   ├── pretrained/            # 预训练权重
│   │   └── vgg16-397923af.pth
│   └── checkpoints/           # 训练检查点
│       ├── latest/            # 最新训练结果
│       │   └── best_model.pth
│       ├── train_20240815_143022/  # 历史训练记录（时间戳）
│       └── train_20240815_151205/  # 历史训练记录（时间戳）
│
├── outputs/                    # 输出目录
│   ├── logs/                  # 训练日志
│   ├── plots/                 # 可视化图表
│   ├── predictions/           # 预测结果
│   └── evaluation/            # 模型评估结果
│       ├── latest/            # 最新评估结果
│       │   ├── reports/       # 评估报告
│       │   ├── confusion_matrices/ # 混淆矩阵图
│       │   └── error_analysis/ # 错误分析图
│       ├── eval_20240815_143500/  # 历史评估记录（时间戳）
│       └── eval_20240815_151800/  # 历史评估记录（时间戳）
│
├── configs/                    # 配置文件
└── tests/                      # 测试文件
    └── __init__.py
```

## 🛠️ 环境要求

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch torchvision tqdm matplotlib pillow opencv-python numpy PyYAML scikit-learn seaborn
```

### 🔤 中文字体支持

为了正确显示图表中的中文标题，需要安装中文字体：

#### Windows 系统
通常已内置中文字体，无需额外安装。

#### macOS 系统
通常已内置中文字体，无需额外安装。

#### Linux 系统
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-noto-cjk

# CentOS/RHEL
sudo yum install wqy-microhei-fonts

# 或使用conda安装
conda install -c conda-forge fonts-conda-ecosystem
```

#### Docker/服务器环境
如果在无GUI环境中运行，系统会自动使用英文标题，确保图表正常显示。

## 📖 使用方法

### 1. 准备数据集

将图像按类别放入对应文件夹：
- `data/raw/class1_125-175/` - 类别0的图像
- `data/raw/class2_180-230/` - 类别1的图像
- `data/raw/class3_233-285/` - 类别2的图像

### 2. 生成数据标注

```bash
python scripts/generate_annotations.py
```

### 3. 开始训练

```bash
python scripts/train.py
```

训练过程中会显示：
- 📊 数据集统计信息
- 🖥️ 设备和模型信息
- 🚀 实时训练进度条
- 📈 每个epoch的损失和精度
- 🎉 最佳模型自动保存
- 💾 定期保存检查点文件

### 4. 恢复训练（从检查点继续训练）

**🔄 新功能：支持从任意检查点恢复训练！**

如果训练中断或想要继续训练更多轮数，可以使用恢复训练功能：

#### 📋 使用步骤

1. **修改配置文件** `configs/training_config.yaml`：
   ```yaml
   training:
     epochs: 100                    # 设置目标轮数（如从50轮继续到100轮）

   resume:
     enabled: true                  # 启用恢复训练模式
     auto_find_latest: true         # 自动查找最新检查点
   ```

2. **运行恢复训练脚本**：
   ```bash
   python scripts/resume_training.py
   ```

   或在PyCharm中直接运行 `resume_training.py` 的主函数

#### ✨ 恢复训练特性

- ✅ **自动检测检查点**：自动找到最新的检查点文件
- ✅ **完整状态恢复**：恢复模型、优化器、学习率调度器状态
- ✅ **验证机制**：验证检查点文件完整性
- ✅ **灵活配置**：支持指定检查点文件或训练目录
- ✅ **PyCharm友好**：支持在IDE中直接运行
- ✅ **训练历史**：正确处理训练历史记录

#### 🎯 典型使用场景

```yaml
# 场景1：从第50轮继续训练到100轮
training:
  epochs: 100
resume:
  enabled: true
  auto_find_latest: true

# 场景2：指定特定的检查点文件
resume:
  enabled: true
  checkpoint_path: "models/checkpoints/train_20241220_143022/checkpoint_epoch_50.pth"

# 场景3：指定训练目录，自动找最新检查点
resume:
  enabled: true
  train_dir: "train_20241220_143022"
```

### 5. 自动可视化分析

**🎉 功能：训练完成后自动生成完整的分析图表！**

训练脚本会在训练结束后自动生成6张专业分析图表：

#### 📊 自动生成的图表
1. **training_history.png** - 训练历史分析（2x2子图）
   - 训练损失 vs 验证损失
   - 训练准确率 vs 验证准确率
   - 验证准确率详细视图（带最佳点标记）
   - 过拟合分析（训练与验证的差距）

2. **confusion_matrix.png** - 标准混淆矩阵（显示具体数量）
3. **confusion_matrix_normalized.png** - 归一化混淆矩阵（显示百分比）
4. **roc_curves.png** - 三个坍落度类别的ROC曲线 + AUC值
5. **pr_curves.png** - 精确率-召回率曲线 + AP值
6. **class_performance_radar.png** - 类别性能雷达图（精确率/召回率/F1-Score）

#### 📁 图表保存位置
```
outputs/plots/
├── train_YYYYMMDD_HHMMSS/  # 每次训练的时间戳目录
└── latest/                 # 最新训练结果（便于查看）
```

#### 🧪 测试可视化功能
```bash
python tests/test_visualization.py
```

### 6. 模型评估

```bash
python scripts/evaluate_model.py
```

生成完整的评估报告，包括：
- 📊 多维度性能指标（精确率、召回率、F1-Score等）
- 🔍 混淆矩阵分析和可视化
- ❌ 错误样本分析和可视化
- 📈 综合性能报告

### 7. 进行预测

单张图像预测：
```bash
python scripts/predict.py path/to/image.jpg
```

批量预测：
```bash
python scripts/predict.py path/to/image/directory/
```

### 8. 训练历史管理

```bash
python scripts/manage_training_history.py
```

## 🔄 恢复训练详细说明

### 使用场景
- ✅ 训练意外中断，需要从检查点继续
- ✅ 想要增加训练轮数（如从50轮继续到100轮）
- ✅ 调整学习率等参数后继续训练
- ✅ 在不同设备间迁移训练

### 配置说明

在 `configs/training_config.yaml` 中的恢复训练配置：

```yaml
# ==================== 恢复训练配置 ====================
resume:
  enabled: false                        # 是否启用恢复训练模式
  checkpoint_path: ""                   # 指定检查点文件路径（可选）
  train_dir: ""                         # 指定训练目录名（可选）
  auto_find_latest: true                # 自动查找最新检查点
  continue_from_epoch: null             # 从指定epoch继续（可选）
```

### 使用方法

#### 方法1：自动恢复（推荐）
1. 修改配置文件：
   ```yaml
   training:
     epochs: 100                    # 设置目标轮数
   resume:
     enabled: true                  # 启用恢复训练
     auto_find_latest: true         # 自动查找最新检查点
   ```

2. 在PyCharm中运行 `scripts/resume_training.py`

#### 方法2：指定检查点文件
```yaml
resume:
  enabled: true
  checkpoint_path: "models/checkpoints/train_20241220_143022/checkpoint_epoch_50.pth"
```

#### 方法3：指定训练目录
```yaml
resume:
  enabled: true
  train_dir: "train_20241220_143022"
```

### 注意事项
- 🔍 确保数据集和配置与原训练保持一致
- 💾 检查点文件包含完整的训练状态
- 📊 恢复训练会创建新的时间戳目录
- 🎯 最佳模型会在新目录中更新保存

功能包括：
- 📋 查看所有训练和评估历史
- 📊 比较不同训练的性能
- 📄 查看历史训练的配置参数和日志
- 🧹 清理旧的训练结果，节省空间

## 📊 训练输出示例

```
📊 数据集信息:
   总样本数: 23841
   训练样本: 19072
   验证样本: 4769
   训练/验证比例: 80.0%/20.0%
   Batch Size: 4
   训练批次数: 4768
   验证批次数: 1193

🖥️ 设备信息:
   使用设备: cuda
   GPU名称: NVIDIA GeForce RTX 3080
   GPU内存: 10.0 GB

🏗️ 构建网络:
   模型参数总量: 138,357,544
   可训练参数: 138,357,544

⚙️ 训练配置:
   优化器: Adam
   初始学习率: 0.0001
   学习率调度: StepLR (step_size=1)

🚀 开始训练 (共 20 个 epoch)
================================================================================

Epoch 1/20 [Train]: 100%|██████████| 4768/4768 [03:45<00:00, Loss: 0.8234, Acc: 65.23%]
Epoch 1/20 [Val]:   100%|██████████| 1193/1193 [00:32<00:00, Loss: 0.6543, Acc: 72.45%]

Epoch 1/20 完成:
  训练 - Loss: 0.8234, Acc: 65.23%
  验证 - Loss: 0.6543, Acc: 72.45%
  学习率: 0.000100
  用时: 257.3s
  🎉 新的最佳验证精度: 72.45% (模型已保存)
```

## 🎯 模型性能

- **输入尺寸**: 224×224×3
- **类别数量**: 3
- **数据增强**: 随机翻转、旋转、色域变换
- **优化器**: Adam
- **学习率调度**: StepLR

## 🔧 项目特色

### 模块化设计
- **清晰的代码组织**: 按功能模块分离代码
- **可重用组件**: 模型、数据处理、工具函数独立封装
- **易于扩展**: 新增功能只需在对应模块中添加

### 完整的工作流程
- **数据管理**: 原始数据、处理数据、标注文件分离存储
- **模型管理**: 预训练权重、检查点、最佳模型分类保存
- **结果输出**: 日志、图表、预测结果统一管理

### 时间戳历史管理
- **自动时间戳**: 每次训练自动创建时间戳目录，避免结果覆盖
- **完整日志记录**: 保存训练配置、过程日志、最终结果
- **配置备份**: 每次训练自动备份配置文件，便于复现
- **历史对比**: 支持查看和比较不同训练的参数和结果

### 专业的脚本工具
- **训练脚本**: 完整的训练流程，支持断点续训，详细日志记录
- **预测脚本**: 支持单张和批量预测，结果可视化
- **评估脚本**: 全面的模型性能评估和分析
- **可视化脚本**: 训练历史分析，模型性能评估
- **历史管理脚本**: 查看、比较、清理训练历史
- **数据处理脚本**: 自动生成标注文件，数据验证

## 📝 注意事项

1. 确保GPU内存足够（建议8GB以上）
2. 根据数据集大小调整batch_size
3. 可根据需要修改类别名称和数量
4. 训练过程中会自动保存检查点，可随时中断和恢复

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！
