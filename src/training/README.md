# 训练组件模块 (src/training)

本模块提供优化器、学习率调度器、损失函数的工厂接口，支持灵活的训练组件切换和配置。

## 📁 模块结构

```
src/training/
├── __init__.py              # 模块初始化和导入
├── optimizer_factory.py    # 优化器工厂
├── scheduler_factory.py    # 学习率调度器工厂
├── loss_factory.py         # 损失函数工厂
└── README.md               # 本说明文档
```

## ⚙️ 优化器工厂

### 支持的优化器

| 优化器 | 名称 | 特点 | 适用场景 |
|--------|------|------|----------|
| Adam | Adam | 自适应学习率，收敛快 | 大多数任务的首选 |
| SGD | SGD | 经典梯度下降，稳定 | 需要精确控制的场景 |
| AdamW | AdamW | Adam的改进版，更好的权重衰减 | 大模型训练 |
| RMSprop | RMSprop | 适合非平稳目标 | RNN等循环网络 |
| Adagrad | Adagrad | 自适应梯度算法 | 稀疏数据 |

### 使用示例

```python
from src.training import create_optimizer

# 基础配置
optimizer_config = {
    'name': 'Adam',
    'params': {
        'lr': 0.001,
        'weight_decay': 0.0001
    }
}

optimizer = create_optimizer(model, optimizer_config)
```

### 配置文件示例

```yaml
training:
  optimizer:
    name: "AdamW"
    params:
      lr: 0.001
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1e-8
```

## 📈 学习率调度器工厂

### 支持的调度器

| 调度器 | 名称 | 特点 | 适用场景 |
|--------|------|------|----------|
| StepLR | 阶梯式衰减 | 固定间隔降低学习率 | 简单稳定的训练 |
| MultiStepLR | 多阶梯衰减 | 指定轮次降低学习率 | 精确控制学习率变化 |
| CosineAnnealingLR | 余弦退火 | 平滑的学习率变化 | 大多数深度学习任务 |
| ReduceLROnPlateau | 自适应调度 | 基于指标自动调整 | 不确定最优调度策略时 |
| ExponentialLR | 指数衰减 | 指数形式衰减 | 需要快速衰减的场景 |

### 使用示例

```python
from src.training import create_scheduler

scheduler_config = {
    'name': 'CosineAnnealingLR',
    'params': {
        'T_max': 100,  # 或 'auto'
        'eta_min': 1e-6
    }
}

scheduler = create_scheduler(optimizer, scheduler_config, total_epochs=100)
```

### 配置文件示例

```yaml
training:
  scheduler:
    name: "CosineAnnealingLR"
    params:
      T_max: "auto"  # 自动使用总轮数
      eta_min: 1e-6
```

## 🎯 损失函数工厂

### 支持的损失函数

| 损失函数 | 名称 | 特点 | 适用场景 |
|----------|------|------|----------|
| CrossEntropyLoss | 标准交叉熵 | 经典分类损失 | 平衡数据集 |
| WeightedCrossEntropyLoss | 加权交叉熵 | 自动处理类别不平衡 | 不平衡数据集 |
| FocalLoss | Focal Loss | 专门处理类别不平衡 | 极度不平衡数据 |
| LabelSmoothingCrossEntropy | 标签平滑 | 减少过拟合 | 提高泛化能力 |

### 使用示例

```python
from src.training import create_loss_function

loss_config = {
    'name': 'WeightedCrossEntropyLoss',
    'params': {
        'auto_weight': True
    }
}

criterion = create_loss_function(loss_config, train_lines, num_classes)
```

### 配置文件示例

```yaml
training:
  loss_function:
    name: "FocalLoss"
    params:
      focal_alpha: 1.0
      focal_gamma: 2.0
```

## 🔧 工厂API详解

### OptimizerFactory

```python
from src.training import OptimizerFactory

# 创建优化器
optimizer = OptimizerFactory.create_optimizer(model, config)

# 获取可用优化器
available = OptimizerFactory.get_available_optimizers()

# 验证配置
is_valid = OptimizerFactory.validate_config(config)

# 注册自定义优化器
OptimizerFactory.register_optimizer(name, optimizer_class, display_name, description, default_params)
```

### SchedulerFactory

```python
from src.training import SchedulerFactory

# 创建调度器
scheduler = SchedulerFactory.create_scheduler(optimizer, config, total_epochs)

# 检查是否需要指标
needs_metric = SchedulerFactory.requires_metric('ReduceLROnPlateau')

# 获取可用调度器
available = SchedulerFactory.get_available_schedulers()
```

### LossFactory

```python
from src.training import LossFactory

# 创建损失函数
criterion = LossFactory.create_loss_function(config, train_lines, num_classes)

# 获取可用损失函数
available = LossFactory.get_available_losses()

# 注册自定义损失函数
LossFactory.register_loss(name, loss_class, display_name, description, default_params)
```

## 🎯 最佳实践

### 优化器选择

1. **通用场景**: 使用Adam
   ```yaml
   optimizer:
     name: "Adam"
     params:
       lr: 0.001
       weight_decay: 0.0001
   ```

2. **大模型训练**: 使用AdamW
   ```yaml
   optimizer:
     name: "AdamW"
     params:
       lr: 0.001
       weight_decay: 0.01
   ```

3. **精确控制**: 使用SGD
   ```yaml
   optimizer:
     name: "SGD"
     params:
       lr: 0.01
       momentum: 0.9
       nesterov: true
   ```

### 调度器选择

1. **推荐配置**: 余弦退火
   ```yaml
   scheduler:
     name: "CosineAnnealingLR"
     params:
       T_max: "auto"
       eta_min: 1e-6
   ```

2. **简单稳定**: 阶梯衰减
   ```yaml
   scheduler:
     name: "StepLR"
     params:
       step_size: 10
       gamma: 0.5
   ```

3. **自适应**: 基于指标调整
   ```yaml
   scheduler:
     name: "ReduceLROnPlateau"
     params:
       mode: "min"
       factor: 0.5
       patience: 5
   ```

### 损失函数选择

1. **平衡数据**: 标准交叉熵
   ```yaml
   loss_function:
     name: "CrossEntropyLoss"
   ```

2. **不平衡数据**: 加权交叉熵
   ```yaml
   loss_function:
     name: "WeightedCrossEntropyLoss"
     params:
       auto_weight: true
   ```

3. **极度不平衡**: Focal Loss
   ```yaml
   loss_function:
     name: "FocalLoss"
     params:
       focal_alpha: 1.0
       focal_gamma: 2.0
   ```

## 🔄 扩展自定义组件

### 添加自定义优化器

```python
import torch.optim as optim
from src.training import register_custom_optimizer

class CustomOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.001, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        # 自定义优化步骤
        pass

# 注册优化器
register_custom_optimizer(
    name="custom",
    optimizer_class=CustomOptimizer,
    display_name="Custom Optimizer",
    description="自定义优化器",
    default_params={'lr': 0.001}
)
```

### 添加自定义损失函数

```python
import torch.nn as nn
from src.training import register_custom_loss

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        # 自定义损失计算
        return loss

# 注册损失函数
register_custom_loss(
    name="custom_loss",
    loss_class=CustomLoss,
    display_name="Custom Loss",
    description="自定义损失函数",
    default_params={'alpha': 1.0}
)
```

## ⚠️ 注意事项

1. **参数兼容性**: 确保参数与PyTorch官方API兼容
2. **设备管理**: 损失函数需要手动移动到GPU
3. **调度器类型**: ReduceLROnPlateau需要在验证后调用step()
4. **权重计算**: WeightedCrossEntropyLoss会自动计算类别权重

## 🐛 常见问题

### Q1: 优化器参数无效
**A**: 检查参数名称是否正确，参考PyTorch官方文档

### Q2: 调度器不生效
**A**: 确保在正确的时机调用scheduler.step()

### Q3: 损失函数报错
**A**: 检查输入张量的形状和设备是否匹配

### Q4: 自定义组件注册失败
**A**: 确保组件类继承了正确的基类
