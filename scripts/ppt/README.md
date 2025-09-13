# PPT可视化模块 (scripts/ppt)

本模块专门用于生成PPT演示所需的数据处理可视化图表，展示从原始图像到最终处理结果的完整流程。

## 📁 模块结构

```
scripts/ppt/
├── __init__.py                    # 模块初始化文件
├── generate_ppt_visuals.py       # 主要的可视化生成脚本
├── out/                          # 输出图片目录
│   ├── 01_original_image.png     # 原始图像展示
│   ├── 02_rgb_conversion.png     # RGB格式转换对比
│   ├── 03_resize_224.png         # 尺寸调整效果
│   ├── 04_channel_reorder.png    # 通道重排可视化
│   ├── 05_augmentation_effects.png # 数据增强效果对比
│   ├── 06_dataset_distribution.png # 数据集分布统计
│   ├── 07_vgg_architecture.png   # VGG架构简化图
│   └── 08_vgg_vs_human.png       # VGG vs 人眼对比图
└── README.md                     # 本说明文档
```

## 🎨 功能说明

### generate_ppt_visuals.py - PPT可视化生成器

**功能**: 生成完整的数据处理流程可视化图表，适用于PPT演示

**使用方法**:
```bash
# 在项目根目录运行
python scripts/ppt/generate_ppt_visuals.py

# 或在PyCharm中直接运行
```

**生成的图表**:

1. **01_original_image.png** - 原始图像展示
   - 显示原始混凝土图像
   - 包含图像尺寸、格式等信息
   - 用于PPT开场介绍

2. **02_rgb_conversion.png** - RGB格式转换对比
   - 对比原始格式与RGB格式
   - 展示格式标准化的必要性
   - 确保所有图像统一为RGB格式

3. **03_resize_224.png** - 尺寸调整效果
   - 展示原始尺寸 → 等比缩放 → 填充到224×224的过程
   - 说明保持宽高比的智能缩放策略
   - 展示灰色填充区域的作用

4. **04_channel_reorder.png** - 通道重排可视化
   - 展示HWC到CHW格式的转换
   - 显示各个颜色通道的分离效果
   - 说明数值归一化：[0,255] → [-1,1]

5. **05_augmentation_effects.png** - 数据增强效果对比
   - 对比原图、轻度、中度、重度增强效果
   - 每种配置生成3个随机样本
   - 展示数据增强提升泛化能力的作用

6. **06_dataset_distribution.png** - 数据集分布统计
   - 柱状图显示各类别样本数量
   - 饼图显示样本比例分布
   - 包含总体统计信息

7. **07_vgg_architecture.png** - VGG架构3D立体图
   - 3D立体方块展示VGG16的完整架构
   - 精确标注每层的尺寸变化（如224×224×64）
   - 颜色编码：卷积层（粉色）、池化层（红色）、全连接层（紫色）、输出层（黄色）
   - 专业的学术风格，类似论文中的架构图
   - 清晰展示从输入到输出的数据流和维度变化

8. **08_vgg_vs_human.png** - VGG vs 人眼对比图
   - 左右对比人眼识别和VGG识别的完整过程
   - 展示处理时间、准确性、一致性等关键差异
   - 突出VGG模型的技术优势
   - 用通俗易懂的语言和图标说明

## 🚀 使用示例

### 快速生成所有图表
```bash
cd F:\project\vgg_test
python scripts/ppt/generate_ppt_visuals.py
```

### 在PyCharm中使用
1. 打开 `scripts/ppt/generate_ppt_visuals.py` 文件
2. 直接点击运行按钮或右键选择"Run"
3. 查看控制台输出和生成的图表文件

## 📊 输出特性

### 图片质量
- **分辨率**: 300 DPI高分辨率，适合投影展示
- **格式**: PNG格式，支持透明背景
- **尺寸**: 根据内容自动调整，保证清晰度

### 视觉风格
- **中文支持**: 自动检测中文字体，确保中文显示正常
- **统一样式**: 使用项目统一的绘图样式
- **专业布局**: 适合学术和技术演示的布局设计

### 文件命名
- **有序命名**: 按照数据处理流程顺序编号
- **描述性名称**: 文件名清楚说明图表内容
- **便于使用**: 可以直接按顺序插入PPT

## 🔧 技术实现

### 依赖模块
- 复用 `src.data.dataset` 模块的预处理功能
- 使用 `scripts.visualize.visualize_utils` 的绘图工具
- 读取项目配置文件获取数据路径和参数

### 数据来源
- **示例图像**: 自动从 `data/raw/` 目录选择第一张可用图像
- **配置参数**: 从 `configs/training_config.yaml` 读取
- **标注数据**: 从 `data/annotations/cls_train.txt` 读取

### 错误处理
- **文件检查**: 自动验证图像文件和配置文件存在性
- **异常捕获**: 完整的错误信息和堆栈跟踪
- **目录创建**: 自动创建输出目录

## 💡 使用建议

### PPT演示顺序
1. **数据概览**: 使用 `06_dataset_distribution.png` 介绍数据集
2. **预处理流程**: 按顺序使用 `01-04` 的图表
3. **数据增强**: 使用 `05_augmentation_effects.png` 说明增强策略
4. **技术优势**: 结合图表说明技术特点

### 自定义修改
- **示例图像**: 修改脚本中的 `find_sample_image()` 函数选择特定图像
- **增强配置**: 调整 `generate_augmentation_effects()` 中的配置参数
- **图表样式**: 修改绘图参数调整视觉效果

## 🔍 故障排除

### 常见问题
1. **找不到图像**: 确保 `data/raw/` 目录下有图像文件
2. **配置文件错误**: 检查 `configs/training_config.yaml` 格式
3. **权限问题**: 确保对输出目录有写入权限

### 调试技巧
- 在PyCharm中运行可以看到详细的错误信息
- 检查控制台输出的进度信息
- 验证项目根目录路径设置正确

## 📝 注意事项

1. **运行环境**: 需要在项目根目录或正确设置Python路径
2. **依赖包**: 确保安装了matplotlib、PIL、numpy等依赖
3. **数据完整性**: 确保数据文件和配置文件完整
4. **输出目录**: 脚本会自动创建 `scripts/ppt/out/` 目录
