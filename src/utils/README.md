# 工具模块 (src/utils)

本目录包含项目中使用的各种工具和实用程序模块。

## 📁 模块结构

```
src/utils/
├── __init__.py              # 模块初始化文件
├── visualization.py         # 可视化工具模块
├── checkpoint_manager.py    # 检查点管理工具
└── README.md               # 本说明文档
```

## 🔧 模块说明

### 1. visualization.py
**功能**: 训练过程可视化和结果分析图表生成
- 训练历史图表 (损失、精度曲线)
- 混淆矩阵可视化
- ROC曲线和PR曲线
- 模型性能雷达图
- 错误样本分析图表

**主要函数**:
- `generate_all_plots()`: 生成所有分析图表
- `setup_matplotlib_font()`: 设置中文字体支持
- `get_font_manager()`: 获取字体管理器

### 2. checkpoint_manager.py
**功能**: 训练检查点的管理和恢复功能
- 自动查找最新检查点文件
- 验证检查点文件完整性
- 加载和恢复训练状态
- 检查点信息摘要显示

**主要类**:
- `CheckpointManager`: 检查点管理器主类

**主要方法**:
- `find_latest_checkpoint()`: 查找最新检查点
- `validate_checkpoint()`: 验证检查点文件
- `load_checkpoint()`: 加载检查点并恢复状态
- `list_available_checkpoints()`: 列出所有可用检查点
- `print_checkpoint_summary()`: 打印检查点摘要

## 🚀 使用示例

### 检查点管理器使用
```python
from src.utils.checkpoint_manager import CheckpointManager

# 创建检查点管理器
checkpoint_manager = CheckpointManager(project_root)

# 查找最新检查点
checkpoint_path = checkpoint_manager.find_latest_checkpoint()

# 验证检查点
if checkpoint_manager.validate_checkpoint(checkpoint_path):
    # 加载检查点
    start_epoch, best_acc = checkpoint_manager.load_checkpoint(
        checkpoint_path, model, optimizer, scheduler
    )
    print(f"从第 {start_epoch + 1} 轮开始训练")
```

### 可视化工具使用
```python
from src.utils.visualization import generate_all_plots

# 生成所有分析图表
evaluation_results = generate_all_plots(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    model=model,
    dataloader=val_loader,
    device=device,
    class_names=class_names,
    plot_dir=plot_dir,
    latest_plot_dir=latest_plot_dir
)
```

## 🔄 恢复训练功能

检查点管理器支持完整的训练恢复功能:

1. **自动检测**: 自动查找最新的检查点文件
2. **状态恢复**: 完整恢复模型、优化器、调度器状态
3. **验证机制**: 验证检查点文件的完整性和有效性
4. **灵活配置**: 支持指定检查点文件或训练目录

## 📋 配置说明

恢复训练功能通过配置文件控制，相关配置项:

```yaml
resume:
  enabled: false              # 是否启用恢复训练
  checkpoint_path: ""         # 指定检查点文件路径
  train_dir: ""              # 指定训练目录名
  auto_find_latest: true     # 自动查找最新检查点
  continue_from_epoch: null  # 从指定epoch继续
```

## ⚠️ 注意事项

1. **数据一致性**: 恢复训练时确保使用相同的数据集和配置
2. **路径正确性**: 检查点路径必须相对于项目根目录
3. **版本兼容性**: 确保检查点文件与当前代码版本兼容
4. **存储空间**: 检查点文件较大，注意存储空间管理

## 🛠️ 维护说明

- 定期清理旧的检查点文件以节省存储空间
- 使用 `scripts/manage_training_history.py` 管理训练历史
- 检查点文件包含完整的训练状态，可用于模型部署和进一步分析
