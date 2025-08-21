# 多模型深度学习图像分类项目

基于 PyTorch 的多架构图像分类项目，支持 VGG16、ResNet 等多种模型，具有完整的工厂系统、详细的训练日志和可视化功能。

## ✨ 功能特点

### 🎯 核心功能
- ✅ **多模型架构支持** - VGG16、ResNet18、ResNet50等多种深度学习模型
- ✅ **工厂系统设计** - 统一的模型、优化器、调度器、损失函数创建接口
- ✅ **配置驱动训练** - 通过YAML配置文件灵活切换所有训练组件
- ✅ **多分类坍落度识别** - 精确识别125-285mm范围内的坍落度等级
- ✅ **智能数据增强** - 多种增强策略提升模型泛化能力
- ✅ **自动检查点管理** - 训练过程自动保存，支持断点续训
- ✅ **专业评估体系** - 多维度性能分析和错误样本诊断

### 🛠️ 模块化架构
- ✅ **模型工厂模块** - 统一的多模型创建和管理接口
- ✅ **训练组件工厂** - 优化器、调度器、损失函数的工厂系统
- ✅ **训练管理模块** - 完整的训练和恢复训练功能
- ✅ **评估分析模块** - 支持指定轮次评估和综合性能分析
- ✅ **可视化工具模块** - 数据增强预览、学习率对比、训练历史查看
- ✅ **自动化流程** - 训练完成自动生成专业分析图表

### 📊 技术特性
- ✅ **多模型支持** - VGG16、ResNet18、ResNet50，易于扩展新模型
- ✅ **多组件支持** - 5种优化器、6种调度器、4种损失函数
- ✅ **PyCharm友好** - 所有脚本支持IDE直接运行
- ✅ **配置驱动** - 完全通过YAML配置文件控制训练过程
- ✅ **工厂模式** - 统一的组件创建接口，支持自定义扩展
- ✅ **模块化设计** - 每个功能模块都有详细的使用说明
- ✅ **测试覆盖** - 完整的功能测试确保代码质量

## 📁 项目结构

```
vgg_test/
├── README.md
├── requirements.txt
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── models/                 # 模型定义和工厂
│   │   ├── __init__.py
│   │   ├── vgg.py             # VGG16 网络定义
│   │   ├── resnet.py          # ResNet 网络定义
│   │   ├── model_factory.py   # 模型工厂
│   │   └── README.md          # 模型模块说明
│   ├── training/               # 训练组件工厂
│   │   ├── __init__.py
│   │   ├── optimizer_factory.py # 优化器工厂
│   │   ├── scheduler_factory.py # 学习率调度器工厂
│   │   ├── loss_factory.py    # 损失函数工厂
│   │   └── README.md          # 训练组件说明
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   └── dataset.py         # 数据集处理模块
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
│   ├── predict.py             # 预测脚本
│   ├── training/              # 训练管理模块
│   │   ├── resume_training.py # 恢复训练脚本
│   │   └── README.md          # 训练管理模块详细说明
│   ├── evaluate/              # 评估模块
│   │   ├── evaluate_model.py  # 模型评估脚本
│   │   └── README.md          # 评估模块详细说明
│   ├── visualize/             # 可视化工具模块
│   │   ├── visualize_augmentation.py  # 数据增强效果预览
│   │   ├── visualize_lr_schedule.py   # 学习率调度器对比
│   │   ├── visualize_training.py      # 训练历史信息查看
│   │   └── README.md          # 可视化工具详细说明
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

### 3. 准备预训练权重（推荐）

为了更好的训练效果，建议预先下载预训练权重：

```bash
# 创建权重目录
mkdir -p models/pretrained

# 下载VGG16权重（可选）
wget -O models/pretrained/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth

# 下载ResNet18权重（可选）
wget -O models/pretrained/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth

# 下载ResNet50权重（可选）
wget -O models/pretrained/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

### 4. 选择模型和配置

在 `configs/training_config.yaml` 中配置模型和训练参数：

```yaml
# 模型配置
model:
  name: "resnet50"              # 可选：vgg16, resnet18, resnet50
  pretrained: true              # 使用预训练权重
  dropout: 0.5                  # Dropout概率

  # 指定预训练权重路径（推荐）
  resnet:
    pretrained_weights: "models/pretrained/resnet50-11ad3fa6.pth"

# 训练配置
training:
  optimizer:
    name: "AdamW"               # 可选：Adam, SGD, AdamW, RMSprop
    params:
      lr: 0.001
      weight_decay: 0.01

  scheduler:
    name: "CosineAnnealingLR"   # 可选：StepLR, CosineAnnealingLR等
    params:
      T_max: "auto"
      eta_min: 1e-6

  loss_function:
    name: "WeightedCrossEntropyLoss"  # 可选：CrossEntropyLoss, FocalLoss等
    params:
      auto_weight: true
```

### 5. 开始训练

```bash
python scripts/train.py
```

训练过程中会显示：
- 📊 数据集统计信息
- 🤖 动态模型信息（根据配置显示）
- ⚙️ 优化器和调度器配置
- 🎯 损失函数配置
- 🚀 实时训练进度条
- 📈 每个epoch的损失和精度
- 🎉 最佳模型自动保存
- 💾 定期保存检查点文件

### 6. 恢复训练（从检查点继续训练）

**🔄 智能训练管理**

```bash
# 修改配置启用恢复训练
# configs/training_config.yaml: resume.enabled: true

python scripts/training/resume_training.py
```

恢复训练特性：
- 🔍 **智能检查点查找** - 自动找到最新或指定的检查点
- 🔄 **完整状态恢复** - 恢复模型、优化器、调度器状态
- ✅ **验证机制** - 检查点完整性和配置兼容性验证
- 📊 **训练监控** - 实时进度和预计剩余时间显示

**详细使用说明请查看：** `scripts/training/README.md`

### 5. 自动可视化分析

**� 专业的可视化工具集**

训练完成后自动生成专业分析图表，同时提供独立的可视化工具：

```bash
# 数据增强效果预览
python scripts/visualize/visualize_augmentation.py

# 学习率调度器对比
python scripts/visualize/visualize_lr_schedule.py

# 训练历史信息查看
python scripts/visualize/visualize_training.py
```

自动生成的图表：
- 📈 **训练历史分析** - 损失和准确率变化曲线
- 🎯 **混淆矩阵** - 标准和归一化混淆矩阵
- 📊 **ROC/PR曲线** - 多类别性能评估曲线
- 🕸️ **性能雷达图** - 各类别综合性能对比

**详细使用说明请查看：** `scripts/visualize/README.md`

### 6. 模型评估

**🎯 支持指定训练轮次评估**

```bash
# 方法1：指定训练轮次评估（推荐）
# 修改 scripts/evaluate/evaluate_model.py 中的 SPECIFIC_TRAIN_PATH
# 然后在 PyCharm 中直接运行

# 方法2：使用默认模型评估
python scripts/evaluate/evaluate_model.py
```

评估功能：
- 📊 **多维度性能指标** - 精确率、召回率、F1-Score等
- 🔍 **混淆矩阵分析** - 详细的分类错误分析
- ❌ **错误样本诊断** - 错误样本可视化和分析
- 📈 **综合性能报告** - HTML格式的完整评估报告
- 🔄 **指定轮次评估** - 生成对应时间戳的结果文件夹

**详细使用说明请查看：** `scripts/evaluate/README.md`

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


## 🔧 项目特色

### 🏭 工厂系统设计
- **统一接口**: 所有组件通过工厂模式创建，接口一致
- **配置驱动**: 完全通过YAML配置文件控制所有训练组件
- **易于扩展**: 支持注册自定义模型、优化器、调度器、损失函数
- **类型安全**: 统一的参数验证和错误处理机制

### 🤖 多模型支持
- **VGG16**: 经典架构，适合高精度要求的场景
- **ResNet18**: 轻量级，适合快速实验和资源受限环境
- **ResNet50**: 性能与效率平衡，大多数任务的首选
- **灵活切换**: 仅需修改配置文件即可切换模型

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

## 📁 预训练权重管理

### 权重文件下载

项目支持两种权重管理方式：

**方式1: 预先下载（推荐）**
```bash
# 创建权重目录
mkdir -p models/pretrained

# 根据需要下载对应模型权重
# VGG16 (528MB)
wget -O models/pretrained/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth

# ResNet18 (45MB)
wget -O models/pretrained/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth

# ResNet50 (98MB)
wget -O models/pretrained/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

**方式2: 自动下载**
- 不指定权重路径时，系统会自动下载到PyTorch缓存目录
- 首次使用需要网络连接

### 权重配置

**使用预下载权重**:
```yaml
model:
  name: "resnet50"
  pretrained: true
  resnet:
    pretrained_weights: "models/pretrained/resnet50-11ad3fa6.pth"
```

**使用自动下载**:
```yaml
model:
  name: "resnet50"
  pretrained: true
  resnet:
    pretrained_weights: null  # 或不设置此项
```

### 权重文件对应关系

| 模型 | 权重文件名 | 大小 | 配置路径 |
|------|------------|------|----------|
| VGG16 | vgg16-397923af.pth | 528MB | `vgg.pretrained_weights` |
| ResNet18 | resnet18-f37072fd.pth | 45MB | `resnet.pretrained_weights` |
| ResNet50 | resnet50-11ad3fa6.pth | 98MB | `resnet.pretrained_weights` |

## 📊 模型对比

### 支持的模型架构

| 模型 | 参数量 | 训练速度 | 准确率 | 内存占用 | 推荐场景 |
|------|--------|----------|--------|----------|----------|
| VGG16 | ~138M | 慢 | 高 | 高 | 高精度要求，充足资源 |
| ResNet18 | ~11M | 快 | 中等 | 低 | 快速实验，资源受限 |
| ResNet50 | ~25M | 中等 | 高 | 中等 | 通用场景，性能平衡 |

### 支持的训练组件

| 组件类型 | 支持选项 | 推荐配置 |
|----------|----------|----------|
| **优化器** | Adam, SGD, AdamW, RMSprop, Adagrad | AdamW (大模型), Adam (通用) |
| **调度器** | StepLR, CosineAnnealingLR, ReduceLROnPlateau等 | CosineAnnealingLR |
| **损失函数** | CrossEntropy, WeightedCrossEntropy, FocalLoss等 | WeightedCrossEntropy (不平衡数据) |

### 配置建议

**快速实验**:
```yaml
model:
  name: "resnet18"
  pretrained: true
training:
  optimizer:
    name: "Adam"
  scheduler:
    name: "StepLR"
```

**生产环境**:
```yaml
model:
  name: "resnet50"
  pretrained: true
training:
  optimizer:
    name: "AdamW"
  scheduler:
    name: "CosineAnnealingLR"
  loss_function:
    name: "WeightedCrossEntropyLoss"
```

**高精度要求**:
```yaml
model:
  name: "vgg16"
  pretrained: true
  dropout: 0.5
training:
  optimizer:
    name: "Adam"
  loss_function:
    name: "FocalLoss"
```

## 📝 注意事项

1. **硬件要求**: 确保GPU内存足够（VGG16需要8GB+，ResNet可用4GB+）
2. **批次大小**: 根据GPU内存和模型大小调整batch_size
3. **模型选择**: 根据精度要求和资源限制选择合适的模型
4. **配置文件**: 修改类别数量时需同步更新class_names
5. **预训练权重**:
   - 推荐预先下载权重到`models/pretrained/`目录
   - 在配置文件中指定权重路径可避免网络问题
   - 不指定路径时会自动下载（需要网络连接）
6. **权重文件**: 确保权重文件路径正确且文件完整
7. **检查点管理**: 训练过程中会自动保存检查点，可随时中断和恢复

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！
