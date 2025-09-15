# 训练管理模块 (scripts/training)

本模块提供完整的训练管理功能，专注于恢复训练和训练状态管理。

## 📁 模块结构

```
scripts/training/
├── __init__.py              # 模块初始化文件
├── resume_training.py       # 恢复训练脚本
├── training_utils.py        # 训练工具函数
└── README.md               # 本说明文档
```

## 🔄 主要功能

### 1. 恢复训练管理
- 从检查点恢复训练状态
- 支持多种检查点查找模式
- 完整的训练状态恢复（模型、优化器、调度器）
- 自动验证检查点完整性

### 2. 训练工具函数
- 配置验证和前置条件检查
- 训练目录自动设置
- 日志记录和配置备份
- 训练状态监控和摘要

### 2.1 智能数据集管理 🆕
- **三分法模式**: 自动使用训练集+验证集，测试集保持完全独立
- **科学数据划分**: 60%/20%/20%分层采样确保无偏评估
- **测试集保护**: 测试集从不参与训练过程，确保最终评估可靠性
- **分层采样**: 确保各类别在三个数据集中比例保持一致

### 3. 智能检查点管理
- 自动查找最新检查点
- 支持指定检查点文件或训练目录
- 检查点文件完整性验证
- 灵活的恢复策略

## 🚀 使用方法

### 恢复训练 (resume_training.py)

**功能**: 从检查点恢复训练，避免重新开始训练

**使用步骤**:

1. **修改配置文件** `configs/training_config.yaml`:
   ```yaml
   training:
     epochs: 100                    # 设置目标轮数（如从50轮继续到100轮）

   resume:
     enabled: true                  # 启用恢复训练模式
     auto_find_latest: true         # 自动查找最新检查点
   ```

2. **运行恢复训练脚本**:
   ```bash
   python scripts/training/resume_training.py
   ```
   
   或在PyCharm中直接运行 `resume_training.py`

### 配置选项详解

#### 基础配置
```yaml
resume:
  enabled: true                     # 启用恢复训练
  auto_find_latest: true           # 自动查找最新检查点
```

#### 指定检查点文件
```yaml
resume:
  enabled: true
  checkpoint_path: "models/checkpoints/train_20250820_074155/checkpoint_epoch_50.pth"
```

#### 指定训练目录
```yaml
resume:
  enabled: true
  train_dir: "train_20250820_074155"  # 自动在此目录中找最新检查点
```

#### 高级配置
```yaml
resume:
  enabled: true
  auto_find_latest: false          # 不自动查找
  checkpoint_path: "path/to/specific/checkpoint.pth"
  continue_from_epoch: null        # 从检查点记录的epoch继续
```

## 📊 恢复训练特性

### ✨ 核心特性

1. **🔍 智能检查点查找**
   - 自动查找最新的检查点文件
   - 支持指定检查点文件路径
   - 支持指定训练目录名称
   - 多级查找策略确保找到有效检查点

2. **🔄 完整状态恢复**
   - 恢复模型权重和结构
   - 恢复优化器状态（动量、学习率等）
   - 恢复学习率调度器状态
   - 恢复训练轮数和最佳精度

3. **✅ 验证机制**
   - 检查点文件完整性验证
   - 配置文件有效性检查
   - 训练前置条件验证
   - 数据集和模型兼容性检查

4. **📈 训练监控**
   - 实时训练进度显示
   - 详细的性能指标记录
   - 定期状态摘要输出
   - 预计剩余时间计算

5. **💾 自动化管理**
   - 自动创建时间戳目录
   - 配置文件自动备份
   - 详细日志记录
   - 训练图表自动生成

### 🎯 典型使用场景

#### 场景1: 训练意外中断
```yaml
# 配置：自动恢复最新训练
training:
  epochs: 100
resume:
  enabled: true
  auto_find_latest: true
```

#### 场景2: 继续训练更多轮数
```yaml
# 从50轮继续训练到100轮
training:
  epochs: 100                      # 原来是50，现在改为100
resume:
  enabled: true
  auto_find_latest: true
```

#### 场景3: 从特定检查点恢复
```yaml
# 从指定的检查点文件恢复
resume:
  enabled: true
  checkpoint_path: "models/checkpoints/train_20250820_074155/checkpoint_epoch_30.pth"
```

#### 场景4: 实验对比
```yaml
# 从同一检查点开始不同的实验
resume:
  enabled: true
  train_dir: "train_20250820_074155"
  # 然后修改其他参数（如学习率、调度器等）进行对比实验
```

## 🔧 工具函数说明

### training_utils.py 中的主要函数

```python
# 验证恢复训练配置
valid, error_msg = validate_resume_config(config)

# 设置训练目录
directories = setup_training_directories(project_root, config, "resume")

# 备份配置文件
backup_path = backup_config_file(config, backup_dir)

# 设置训练日志
logger = setup_training_logger(log_dir, timestamp, "resume_training")

# 检查训练前置条件
prerequisites_ok, errors = check_training_prerequisites(project_root, config)

# 获取训练状态摘要
summary = get_training_status_summary(current_epoch, total_epochs, 
                                     train_acc, val_acc, best_acc, elapsed_time)
```

## 📁 输出文件结构

恢复训练会创建新的时间戳目录：

```
outputs/
├── checkpoints/
│   ├── resume_20250820_143022/     # 恢复训练的检查点
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_60.pth
│   │   └── ...
│   └── latest/                     # 最新模型（软链接）
├── logs/
│   ├── resume_20250820_143022/     # 恢复训练的日志
│   │   ├── resume_training.log
│   │   └── config_backup.yaml
│   └── latest/                     # 最新日志
└── plots/
    ├── resume_20250820_143022/     # 恢复训练的图表
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   └── ...
    └── latest/                     # 最新图表
```

## ⚠️ 注意事项

### 重要提醒

1. **数据一致性**: 确保使用与原训练相同的数据集和标注文件
2. **配置兼容性**: 模型结构参数（如类别数量）必须与检查点一致
3. **版本兼容性**: 确保检查点文件与当前代码版本兼容
4. **存储空间**: 恢复训练会创建新的输出目录，注意存储空间

### 配置检查清单

- [ ] `resume.enabled` 设置为 `true`
- [ ] `training.epochs` 设置为目标轮数
- [ ] 检查点文件存在且完整
- [ ] 数据集路径正确
- [ ] 模型配置与检查点匹配

## 🐛 常见问题

### Q1: 提示"恢复训练未启用"
**A**: 检查配置文件中的 `resume.enabled` 是否设置为 `true`

### Q2: 提示"没有找到可用的检查点文件"
**A**: 检查以下项目：
- 是否已经进行过训练并保存了检查点
- 检查点文件路径是否正确
- `models/checkpoints/` 目录是否存在检查点文件

### Q3: 检查点加载失败
**A**: 可能的原因：
- 检查点文件损坏
- 模型结构发生变化
- PyTorch版本不兼容
- 设备类型不匹配（CPU/GPU）

### Q4: 训练轮数已达到目标
**A**: 如果检查点的epoch已达到配置的目标轮数：
- 增加 `training.epochs` 的值
- 或者检查是否选择了错误的检查点

### Q5: 数据加载失败
**A**: 检查：
- 标注文件路径是否正确
- 数据目录是否存在
- 文件权限是否正确

## 💡 最佳实践

### 1. 训练策略
- **定期保存**: 设置合适的保存频率，避免丢失训练进度
- **多检查点**: 保留多个检查点文件，以防单个文件损坏
- **配置备份**: 每次训练都会自动备份配置文件

### 2. 实验管理
- **命名规范**: 使用有意义的训练目录名称
- **参数记录**: 在日志中详细记录实验参数
- **结果对比**: 利用时间戳目录对比不同实验结果

### 3. 资源管理
- **存储清理**: 定期清理旧的检查点和日志文件
- **GPU内存**: 注意批次大小设置，避免内存溢出
- **时间规划**: 合理设置训练轮数，避免过度训练

## 🔗 与其他模块的关系

- **主训练脚本**: `scripts/train.py` 生成的检查点可被本模块使用
- **评估模块**: `scripts/evaluate/` 可评估恢复训练的结果
- **可视化模块**: 自动调用 `src/utils/visualization.py` 生成图表
- **检查点管理**: 使用 `src/utils/checkpoint_manager.py` 管理检查点
- **数据模块**: 使用 `src/data/dataset.py` 加载训练数据
