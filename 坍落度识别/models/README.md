# 模型模块 (src/models)

本模块提供多种深度学习模型架构和统一的模型工厂接口，支持灵活的模型切换和配置。

## 📁 模块结构

```
src/models/
├── __init__.py              # 模块初始化和导入
├── vgg.py                  # VGG模型实现
├── resnet.py               # ResNet模型实现
├── model_factory.py        # 模型工厂核心
└── README.md               # 本说明文档
```

## 🤖 支持的模型

### 1. VGG16
- **文件**: `vgg.py`
- **特点**: 经典的VGG16架构，适合图像分类任务
- **优势**: 结构简单，易于理解和调试
- **适用场景**: 中小型数据集，对准确率要求较高的场景

### 2. ResNet18
- **文件**: `resnet.py`
- **特点**: 轻量级ResNet架构，训练速度快
- **优势**: 参数量少，训练快速，适合快速实验
- **适用场景**: 资源受限环境，快速原型验证

### 3. ResNet50
- **文件**: `resnet.py`
- **特点**: 标准ResNet架构，性能与效率平衡
- **优势**: 残差连接解决梯度消失，性能优秀
- **适用场景**: 大多数图像分类任务的首选

## 🏭 模型工厂使用

### 基础使用

```python
from src.models import create_model

# 配置字典
config = {
    'name': 'resnet50',
    'pretrained': True,
    'num_classes': 4,
    'dropout': 0.5
}

# 创建模型
model = create_model(config)
```

### 配置文件使用

```yaml
model:
  name: "resnet50"                     # 模型类型
  pretrained: true                     # 使用预训练权重
  dropout: 0.5                        # Dropout概率

  # VGG特定配置
  vgg:
    progress: true                     # 显示下载进度
    model_dir: "models/pretrained"     # 权重保存目录
    pretrained_weights: "models/pretrained/vgg16-397923af.pth"  # 自定义权重路径

  # ResNet特定配置
  resnet:
    replace_stride_with_dilation: [false, false, false]
    zero_init_residual: false
    pretrained_weights: "models/pretrained/resnet50-11ad3fa6.pth"  # 自定义权重路径
```

### 高级配置

```python
# VGG配置
vgg_config = {
    'name': 'vgg16',
    'pretrained': True,
    'num_classes': 4,
    'dropout': 0.5,
    'vgg': {
        'progress': True,
        'model_dir': 'models/pretrained'
    }
}

# ResNet配置
resnet_config = {
    'name': 'resnet50',
    'pretrained': True,
    'num_classes': 4,
    'dropout': 0.3,
    'resnet': {
        'replace_stride_with_dilation': [False, False, False],
        'zero_init_residual': False
    }
}
```

## 📁 预训练权重管理

### 权重文件获取

**VGG16权重**:
- 文件名: `vgg16-397923af.pth`
- 下载地址: https://download.pytorch.org/models/vgg16-397923af.pth
- 大小: ~528MB

**ResNet18权重**:
- 文件名: `resnet18-f37072fd.pth`
- 下载地址: https://download.pytorch.org/models/resnet18-f37072fd.pth
- 大小: ~45MB

**ResNet50权重**:
- 文件名: `resnet50-11ad3fa6.pth`
- 下载地址: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
- 大小: ~98MB

### 权重文件放置

推荐的目录结构：
```
models/
└── pretrained/
    ├── vgg16-397923af.pth
    ├── resnet18-f37072fd.pth
    └── resnet50-11ad3fa6.pth
```

### 配置权重路径

**方式1: 使用自定义权重**
```yaml
model:
  name: "resnet50"
  pretrained: true
  resnet:
    pretrained_weights: "models/pretrained/resnet50-11ad3fa6.pth"
```

**方式2: 使用默认下载**
```yaml
model:
  name: "resnet50"
  pretrained: true
  resnet:
    pretrained_weights: null  # 或不设置此项
```

### 权重加载优先级

1. **自定义权重文件** - 如果指定了`pretrained_weights`且文件存在
2. **默认下载** - 如果未指定或文件不存在，使用PyTorch默认下载
3. **无预训练** - 如果`pretrained: false`，从头开始训练

## 🔧 模型工厂API

### ModelFactory类

```python
from src.models import ModelFactory

# 创建模型
model = ModelFactory.create_model(config)

# 获取可用模型
available_models = ModelFactory.get_available_models()

# 获取模型信息
model_info = ModelFactory.get_model_info('resnet50')

# 验证配置
is_valid = ModelFactory.validate_config(config)

# 注册自定义模型
ModelFactory.register_model(name, model_class, display_name, description)
```

### 便捷函数

```python
from src.models import create_model, get_available_models, register_custom_model

# 创建模型
model = create_model(config)

# 获取可用模型列表
models = get_available_models()

# 注册自定义模型
register_custom_model(name, model_class, display_name, description)
```

## 📊 模型对比

| 模型 | 参数量 | 训练速度 | 准确率 | 内存占用 | 推荐场景 |
|------|--------|----------|--------|----------|----------|
| VGG16 | ~138M | 慢 | 高 | 高 | 高精度要求 |
| ResNet18 | ~11M | 快 | 中等 | 低 | 快速实验 |
| ResNet50 | ~25M | 中等 | 高 | 中等 | 通用场景 |

## 🎯 使用建议

### 模型选择指南

1. **快速实验**: 使用ResNet18
   ```yaml
   model:
     name: "resnet18"
     pretrained: true
   ```

2. **生产环境**: 使用ResNet50
   ```yaml
   model:
     name: "resnet50"
     pretrained: true
     dropout: 0.3
   ```

3. **高精度要求**: 使用VGG16
   ```yaml
   model:
     name: "vgg16"
     pretrained: true
     dropout: 0.5
   ```

### 配置优化建议

1. **预训练权重**: 始终使用预训练权重，除非有特殊需求
2. **Dropout设置**: 
   - 小数据集: 0.5-0.7
   - 大数据集: 0.3-0.5
3. **类别数量**: 确保与数据集标签数量一致

## 🔄 扩展自定义模型

### 添加新模型

1. **创建模型文件**:
   ```python
   # src/models/custom_model.py
   import torch.nn as nn
   
   class CustomModel(nn.Module):
       def __init__(self, num_classes=3, dropout=0.5, **kwargs):
           super().__init__()
           # 模型定义
           
       def forward(self, x):
           # 前向传播
           return x
       
       def get_name(self):
           return "CustomModel"
       
       def get_info(self):
           return {'name': 'CustomModel', 'num_classes': self.num_classes}
   ```

2. **注册模型**:
   ```python
   from src.models import register_custom_model
   from .custom_model import CustomModel
   
   register_custom_model(
       name="custom",
       model_class=CustomModel,
       display_name="Custom Model",
       description="自定义模型描述"
   )
   ```

3. **配置使用**:
   ```yaml
   model:
     name: "custom"
     num_classes: 4
     dropout: 0.5
   ```

## ⚠️ 注意事项

1. **模型兼容性**: 确保模型输出维度与类别数量匹配
2. **预训练权重**: 不同模型的预训练权重不通用
3. **内存管理**: 大模型需要更多GPU内存
4. **批次大小**: 根据模型大小调整批次大小

## 🐛 常见问题

### Q1: 模型创建失败
**A**: 检查配置文件中的模型名称是否正确，支持的模型: vgg16, resnet18, resnet50

### Q2: 预训练权重下载失败
**A**: 检查网络连接，或手动下载权重文件到指定目录

### Q3: GPU内存不足
**A**: 减少批次大小或使用更小的模型（如ResNet18）

### Q4: 模型精度不理想
**A**: 尝试不同的模型架构，调整Dropout参数，或使用数据增强

## 🔗 与其他模块的关系

- **训练模块**: 使用模型工厂创建训练模型
- **评估模块**: 加载训练好的模型进行评估
- **预测模块**: 使用训练好的模型进行推理
- **配置系统**: 通过YAML配置文件控制模型选择
