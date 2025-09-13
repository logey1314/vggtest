# 可视化工具模块 (scripts/visualize)

本模块提供各种可视化工具，用于预览和分析训练相关的图表和效果。

## 📁 模块结构

```
scripts/visualize/
├── __init__.py                    # 模块初始化文件
├── visualize_augmentation.py     # 数据增强效果可视化工具
├── visualize_lr_schedule.py      # 学习率调度器可视化工具
├── visualize_training.py         # 训练历史信息查看工具
├── visualize_utils.py            # 可视化工具函数
└── README.md                     # 本说明文档

scripts/data/
├── __init__.py                    # 模块初始化文件
├── visualize_noise.py            # 图像噪声效果可视化工具
└── README.md                     # 数据处理模块说明
```

## 🎨 工具说明

### 1. visualize_augmentation.py - 数据增强效果预览

**功能**: 预览不同数据增强配置对图像的影响效果

**使用方法**:
```bash
python scripts/visualize/visualize_augmentation.py
```

**主要特性**:
- 🖼️ 对比原图与不同强度增强效果
- 📊 生成多样本对比图表
- ⚙️ 支持自定义增强配置
- 💾 自动保存结果到 `outputs/plots/augmentation_effects.png`

**配置选项**:
- **轻度增强**: 保守策略，适合小数据集
- **中度增强**: 推荐配置，平衡效果与稳定性
- **重度增强**: 强泛化能力，适合大数据集

**自定义配置**:
```python
# 修改脚本中的 SAMPLE_IMAGE_PATH
SAMPLE_IMAGE_PATH = 'data/raw/your_class/your_image.jpg'

# 自定义增强配置
custom_config = {
    'enable_flip': True,
    'enable_rotation': True,
    'rotation_range': 20,
    'enable_color_jitter': True,
    'hue_range': 0.15,
    # ... 更多参数
}
```

### 2. visualize_lr_schedule.py - 学习率调度器对比

**功能**: 预览不同学习率调度器在训练过程中的变化曲线

**使用方法**:
```bash
python scripts/visualize/visualize_lr_schedule.py
```

**主要特性**:
- 📈 对比4种主要调度器的变化曲线
- 🔍 使用对数坐标清晰显示变化
- 📝 详细的调度器说明和推荐场景
- 💾 自动保存结果到 `outputs/plots/lr_schedules_comparison.png`

**支持的调度器**:
1. **StepLR**: 阶梯式衰减，每固定epoch降低学习率
2. **MultiStepLR**: 多阶梯衰减，在指定epoch降低学习率
3. **CosineAnnealingLR**: 余弦退火，平滑下降到最小值
4. **ReduceLROnPlateau**: 自适应衰减，监控指标停止改善时降低

**自定义配置**:
```python
# 修改脚本中的参数
EPOCHS = 50          # 训练轮数
LEARNING_RATE = 0.001  # 初始学习率

# 调整调度器参数
'StepLR': {
    'scheduler': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1),
    # ...
}
```

### 3. visualize_training.py - 训练历史信息查看

**功能**: 显示训练图表位置信息和可用图表列表

**使用方法**:
```bash
python scripts/visualize/visualize_training.py
```

**主要特性**:
- 📁 显示图表保存位置
- 📊 列出所有可用的训练图表
- ℹ️ 提供使用说明和提示
- 🔗 引导到其他可视化工具

**显示信息**:
- 最新训练结果位置
- 历史训练结果目录
- 自动生成的图表类型
- 其他可视化工具链接

### 4. 图像噪声效果可视化 (scripts/data/visualize_noise.py)

**功能**: 预览不同噪声类型和强度对图像的影响效果

**使用方法**:
```bash
python scripts/data/visualize_noise.py
```

**主要特性**:
- 🖼️ 对比原图与不同强度噪声效果
- 📊 生成多样本对比图表
- ⚙️ 支持多种噪声类型组合
- 💾 自动保存结果到 `outputs/plots/noise_effects.png`

**噪声类型展示**:
- **原图**: 无噪声处理的原始图像
- **轻度噪声**: 仅高斯噪声，适合轻微降准确率
- **中度噪声**: 高斯噪声 + 椒盐噪声组合
- **重度噪声**: 多种噪声类型组合，显著降低图像质量

**详细使用说明请查看：** `scripts/data/README.md`

### 5. visualize_utils.py - 工具函数库

**功能**: 提供通用的可视化辅助函数

**主要函数**:
```python
# 设置绘图样式和中文字体支持
chinese_support = setup_plot_style()

# 创建输出目录
output_dir = create_output_directory(base_dir, subdir)

# 保存图表并可选择显示
output_path = save_plot_with_timestamp(fig, base_path, filename, show_plot=True)

# 验证图像路径
valid_path = validate_image_path(image_path, project_root)

# 获取项目根目录
project_root = get_project_root()

# 创建颜色调色板
colors = create_color_palette(n_colors)
```

## 🚀 使用示例

### 快速开始

1. **预览数据增强效果**:
   ```bash
   cd F:\project\vgg_test
   python scripts/visualize/visualize_augmentation.py
   ```

2. **对比学习率调度器**:
   ```bash
   python scripts/visualize/visualize_lr_schedule.py
   ```

3. **查看训练图表信息**:
   ```bash
   python scripts/visualize/visualize_training.py
   ```

### 在PyCharm中使用

1. 打开对应的可视化脚本文件
2. 直接点击运行按钮或右键选择"Run"
3. 查看控制台输出和生成的图表文件

### 自定义使用

```python
# 导入工具函数
from scripts.visualize.visualize_utils import setup_plot_style, save_plot_with_timestamp

# 设置绘图样式
chinese_support = setup_plot_style()

# 创建自定义图表
fig, ax = plt.subplots(figsize=(10, 6))
# ... 绘图代码 ...

# 保存图表
output_path = save_plot_with_timestamp(fig, 'outputs/plots', 'my_custom_plot')
```

## 📊 输出文件

所有可视化工具生成的图表都保存在 `outputs/plots/` 目录下：

```
outputs/plots/
├── augmentation_effects.png      # 数据增强效果对比图
├── lr_schedules_comparison.png   # 学习率调度器对比图
├── latest/                       # 最新训练结果
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── ...
└── train_YYYYMMDD_HHMMSS/        # 历史训练结果
    ├── training_history.png
    └── ...
```

## ⚙️ 配置说明

### 图表样式配置

可视化工具会自动：
- 检测并设置中文字体支持
- 使用统一的绘图样式
- 设置高分辨率输出 (300 DPI)
- 添加网格和优化布局

### 自定义配置

1. **修改示例图片路径** (数据增强工具):
   ```python
   SAMPLE_IMAGE_PATH = 'data/raw/your_class/your_image.jpg'
   ```

2. **调整训练参数** (学习率调度器工具):
   ```python
   EPOCHS = 50
   LEARNING_RATE = 0.001
   ```

3. **自定义输出目录**:
   ```python
   output_dir = 'custom/output/path'
   ```

## 🔧 故障排除

### 常见问题

1. **中文字体显示问题**:
   - 工具会自动检测中文字体支持
   - 如果不支持中文，会自动切换到英文显示

2. **图片路径错误**:
   - 检查 `SAMPLE_IMAGE_PATH` 是否指向有效的图片文件
   - 确保图片格式为常见格式 (jpg, png, etc.)

3. **输出目录权限问题**:
   - 确保对 `outputs/plots/` 目录有写入权限
   - 目录会自动创建，无需手动创建

4. **依赖包问题**:
   ```bash
   pip install matplotlib numpy torch torchvision pillow
   ```

### 调试技巧

1. **查看详细错误信息**:
   - 在PyCharm中运行可以看到完整的错误堆栈
   - 检查控制台输出的提示信息

2. **验证环境**:
   - 确保项目根目录正确
   - 检查Python路径设置

## 🔗 与其他模块的关系

- **训练模块**: 训练完成后会自动生成可视化图表
- **数据模块**: 数据增强工具使用 `src/data/dataset.py`
- **配置系统**: 可以扩展支持从配置文件读取参数
- **测试模块**: `tests/test_visualize_module.py` 提供功能测试

## 💡 使用技巧

1. **批量对比**: 修改配置参数批量生成不同设置的对比图
2. **结果分析**: 结合训练结果分析最佳的增强和调度器配置
3. **文档记录**: 将生成的图表用于实验记录和论文撰写
4. **参数调优**: 通过可视化结果指导超参数调整
