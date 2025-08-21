"""
损失函数工厂
统一的损失函数创建接口，支持多种损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    用于处理类别不平衡问题
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失函数
    减少过拟合，提高泛化能力
    """
    
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LossFactory:
    """损失函数工厂类"""
    
    # 损失函数注册表
    LOSS_REGISTRY = {
        'crossentropyloss': {
            'class': nn.CrossEntropyLoss,
            'name': 'CrossEntropyLoss',
            'description': '标准交叉熵损失函数',
            'default_params': {
                'reduction': 'mean',
                'label_smoothing': 0.0
            },
            'supports_weights': True
        },
        'weightedcrossentropyloss': {
            'class': nn.CrossEntropyLoss,
            'name': 'WeightedCrossEntropyLoss',
            'description': '加权交叉熵损失函数，自动处理类别不平衡',
            'default_params': {
                'reduction': 'mean',
                'label_smoothing': 0.0
            },
            'supports_weights': True,
            'auto_weight': True
        },
        'focalloss': {
            'class': FocalLoss,
            'name': 'FocalLoss',
            'description': 'Focal Loss，专门处理类别不平衡问题',
            'default_params': {
                'alpha': 1.0,
                'gamma': 2.0,
                'reduction': 'mean'
            },
            'supports_weights': False
        },
        'labelsmoothingcrossentropy': {
            'class': LabelSmoothingLoss,
            'name': 'LabelSmoothingCrossEntropy',
            'description': '标签平滑交叉熵损失函数',
            'default_params': {
                'smoothing': 0.1,
                'reduction': 'mean'
            },
            'supports_weights': False
        }
    }
    
    @classmethod
    def create_loss_function(cls, config: Dict[str, Any], train_lines: List[str] = None, 
                           num_classes: int = None) -> nn.Module:
        """
        根据配置创建损失函数
        
        Args:
            config (Dict): 损失函数配置
            train_lines (List[str]): 训练数据行（用于计算类别权重）
            num_classes (int): 类别数量
            
        Returns:
            nn.Module: 创建的损失函数实例
            
        Example:
            config = {
                'name': 'FocalLoss',
                'params': {
                    'alpha': 1.0,
                    'gamma': 2.0
                }
            }
            loss_fn = LossFactory.create_loss_function(config)
        """
        # 兼容旧配置格式
        if isinstance(config, str):
            loss_name = config.lower()
            loss_params = {}
        else:
            loss_name = config.get('name', 'crossentropyloss').lower()
            loss_params = config.get('params', {})
        
        if loss_name not in cls.LOSS_REGISTRY:
            available_losses = list(cls.LOSS_REGISTRY.keys())
            raise ValueError(f"不支持的损失函数类型: {loss_name}. "
                           f"支持的损失函数: {available_losses}")
        
        # 获取损失函数信息
        loss_info = cls.LOSS_REGISTRY[loss_name]
        loss_class = loss_info['class']
        default_params = loss_info['default_params'].copy()
        
        # 合并默认参数和用户参数
        final_params = {**default_params, **loss_params}
        
        # 处理特殊参数
        final_params = cls._process_special_params(
            loss_name, final_params, loss_info, train_lines, num_classes
        )
        
        # 创建损失函数
        try:
            loss_function = loss_class(**final_params)
            
            # 打印损失函数信息
            cls._print_loss_info(loss_name, final_params, loss_info)
            
            return loss_function
            
        except Exception as e:
            raise RuntimeError(f"创建损失函数 {loss_name} 失败: {e}")
    
    @classmethod
    def _process_special_params(cls, loss_name: str, params: Dict[str, Any], 
                               loss_info: Dict[str, Any], train_lines: List[str] = None, 
                               num_classes: int = None) -> Dict[str, Any]:
        """处理特殊参数"""
        final_params = params.copy()
        
        # 处理加权交叉熵的权重计算
        if loss_info.get('auto_weight', False) and train_lines and num_classes:
            if params.get('auto_weight', True):
                weights = cls._calculate_class_weights(train_lines, num_classes)
                final_params['weight'] = weights
                print(f"🔢 自动计算类别权重: {weights.tolist()}")
                # 移除auto_weight参数，因为PyTorch的CrossEntropyLoss不接受这个参数
                final_params.pop('auto_weight', None)
        
        # 处理标签平滑损失的类别数量
        if loss_name == 'labelsmoothingcrossentropy' and num_classes:
            final_params['num_classes'] = num_classes
        
        return final_params
    
    @classmethod
    def _calculate_class_weights(cls, train_lines: List[str], num_classes: int) -> torch.FloatTensor:
        """计算类别权重"""
        class_counts = [0] * num_classes
        
        for line in train_lines:
            try:
                class_id = int(line.split(';')[0])
                if 0 <= class_id < num_classes:
                    class_counts[class_id] += 1
            except (ValueError, IndexError):
                continue
        
        total_samples = sum(class_counts)
        if total_samples == 0:
            return torch.ones(num_classes, dtype=torch.float32)
        
        # 计算权重：总样本数 / (类别数 * 该类别样本数)
        class_weights = []
        for count in class_counts:
            if count > 0:
                weight = total_samples / (num_classes * count)
            else:
                weight = 1.0  # 如果某个类别没有样本，设置权重为1
            class_weights.append(weight)
        
        return torch.FloatTensor(class_weights)
    
    @classmethod
    def _print_loss_info(cls, loss_name: str, params: Dict[str, Any], loss_info: Dict[str, Any]):
        """打印损失函数信息"""
        print(f"\n🎯 损失函数配置:")
        print(f"   损失函数类型: {loss_info['name']}")
        
        # 打印关键参数
        if loss_name == 'crossentropyloss' or loss_name == 'weightedcrossentropyloss':
            print(f"   标签平滑: {params.get('label_smoothing', 0.0)}")
            if 'weight' in params:
                print(f"   使用类别权重: 是")
            else:
                print(f"   使用类别权重: 否")
        elif loss_name == 'focalloss':
            print(f"   Alpha: {params.get('alpha', 1.0)}")
            print(f"   Gamma: {params.get('gamma', 2.0)}")
        elif loss_name == 'labelsmoothingcrossentropy':
            print(f"   平滑参数: {params.get('smoothing', 0.1)}")
            print(f"   类别数量: {params.get('num_classes', 'N/A')}")
        
        print(f"   归约方式: {params.get('reduction', 'mean')}")
    
    @classmethod
    def get_available_losses(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有可用的损失函数信息
        
        Returns:
            Dict: 损失函数信息字典
        """
        return cls.LOSS_REGISTRY.copy()
    
    @classmethod
    def get_loss_info(cls, loss_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定损失函数的信息
        
        Args:
            loss_name (str): 损失函数名称
            
        Returns:
            Dict: 损失函数信息，如果损失函数不存在返回None
        """
        return cls.LOSS_REGISTRY.get(loss_name.lower())
    
    @classmethod
    def register_loss(cls, name: str, loss_class, display_name: str, 
                     description: str, default_params: Dict[str, Any], 
                     supports_weights: bool = False, auto_weight: bool = False):
        """
        注册新的损失函数类型
        
        Args:
            name (str): 损失函数名称（用于配置文件）
            loss_class: 损失函数类
            display_name (str): 显示名称
            description (str): 损失函数描述
            default_params (Dict): 默认参数
            supports_weights (bool): 是否支持类别权重
            auto_weight (bool): 是否自动计算权重
        """
        cls.LOSS_REGISTRY[name.lower()] = {
            'class': loss_class,
            'name': display_name,
            'description': description,
            'default_params': default_params,
            'supports_weights': supports_weights,
            'auto_weight': auto_weight
        }
        print(f"✅ 已注册新损失函数: {display_name} ({name})")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证损失函数配置
        
        Args:
            config (Dict): 损失函数配置
            
        Returns:
            bool: 配置是否有效
        """
        if isinstance(config, str):
            return config.lower() in cls.LOSS_REGISTRY
        
        if not isinstance(config, dict):
            print("❌ 损失函数配置必须是字符串或字典")
            return False
        
        if 'name' not in config:
            print("❌ 损失函数配置缺少 'name' 字段")
            return False
        
        loss_name = config['name'].lower()
        if loss_name not in cls.LOSS_REGISTRY:
            available_losses = list(cls.LOSS_REGISTRY.keys())
            print(f"❌ 不支持的损失函数类型: {loss_name}")
            print(f"   支持的损失函数: {available_losses}")
            return False
        
        return True


# 便捷函数
def create_loss_function(config: Dict[str, Any], train_lines: List[str] = None, 
                        num_classes: int = None) -> nn.Module:
    """
    便捷的损失函数创建函数
    
    Args:
        config (Dict): 损失函数配置
        train_lines (List[str]): 训练数据行
        num_classes (int): 类别数量
        
    Returns:
        nn.Module: 创建的损失函数
    """
    return LossFactory.create_loss_function(config, train_lines, num_classes)


def get_available_losses() -> Dict[str, Dict[str, Any]]:
    """获取所有可用损失函数"""
    return LossFactory.get_available_losses()


def register_custom_loss(name: str, loss_class, display_name: str, description: str, 
                        default_params: Dict[str, Any], supports_weights: bool = False, 
                        auto_weight: bool = False):
    """注册自定义损失函数"""
    LossFactory.register_loss(name, loss_class, display_name, description, 
                             default_params, supports_weights, auto_weight)
