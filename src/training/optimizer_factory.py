"""
优化器工厂
统一的优化器创建接口，支持多种优化算法
"""

import torch.optim as optim
from typing import Dict, Any, Optional
import torch.nn as nn


class OptimizerFactory:
    """优化器工厂类"""
    
    # 优化器注册表
    OPTIMIZER_REGISTRY = {
        'adam': {
            'class': optim.Adam,
            'name': 'Adam',
            'description': '自适应学习率优化器，适合大多数任务',
            'default_params': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0,
                'amsgrad': False
            }
        },
        'sgd': {
            'class': optim.SGD,
            'name': 'SGD',
            'description': '随机梯度下降，经典优化算法',
            'default_params': {
                'lr': 0.01,
                'momentum': 0.9,
                'dampening': 0,
                'weight_decay': 0,
                'nesterov': False
            }
        },
        'adamw': {
            'class': optim.AdamW,
            'name': 'AdamW',
            'description': 'Adam的改进版本，更好的权重衰减',
            'default_params': {
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0.01,
                'amsgrad': False
            }
        },
        'rmsprop': {
            'class': optim.RMSprop,
            'name': 'RMSprop',
            'description': '适合处理非平稳目标的优化器',
            'default_params': {
                'lr': 0.01,
                'alpha': 0.99,
                'eps': 1e-08,
                'weight_decay': 0,
                'momentum': 0,
                'centered': False
            }
        },
        'adagrad': {
            'class': optim.Adagrad,
            'name': 'Adagrad',
            'description': '自适应梯度算法，适合稀疏数据',
            'default_params': {
                'lr': 0.01,
                'lr_decay': 0,
                'weight_decay': 0,
                'initial_accumulator_value': 0,
                'eps': 1e-10
            }
        }
    }
    
    @classmethod
    def create_optimizer(cls, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """
        根据配置创建优化器
        
        Args:
            model (nn.Module): 要优化的模型
            config (Dict): 优化器配置
            
        Returns:
            optim.Optimizer: 创建的优化器实例
            
        Example:
            config = {
                'name': 'Adam',
                'params': {
                    'lr': 0.001,
                    'weight_decay': 0.0001
                }
            }
            optimizer = OptimizerFactory.create_optimizer(model, config)
        """
        # 兼容旧配置格式
        if isinstance(config, str):
            optimizer_name = config.lower()
            optimizer_params = {}
        else:
            optimizer_name = config.get('name', 'adam').lower()
            optimizer_params = config.get('params', {})
        
        if optimizer_name not in cls.OPTIMIZER_REGISTRY:
            available_optimizers = list(cls.OPTIMIZER_REGISTRY.keys())
            raise ValueError(f"不支持的优化器类型: {optimizer_name}. "
                           f"支持的优化器: {available_optimizers}")
        
        # 获取优化器信息
        optimizer_info = cls.OPTIMIZER_REGISTRY[optimizer_name]
        optimizer_class = optimizer_info['class']
        default_params = optimizer_info['default_params'].copy()
        
        # 合并默认参数和用户参数
        final_params = {**default_params, **optimizer_params}
        
        # 创建优化器
        try:
            optimizer = optimizer_class(model.parameters(), **final_params)
            
            # 打印优化器信息
            cls._print_optimizer_info(optimizer_name, final_params)
            
            return optimizer
            
        except Exception as e:
            raise RuntimeError(f"创建优化器 {optimizer_name} 失败: {e}")
    
    @classmethod
    def _print_optimizer_info(cls, optimizer_name: str, params: Dict[str, Any]):
        """打印优化器信息"""
        optimizer_info = cls.OPTIMIZER_REGISTRY[optimizer_name]
        
        print(f"\n⚙️  优化器配置:")
        print(f"   优化器类型: {optimizer_info['name']}")
        print(f"   学习率: {params.get('lr', 'N/A')}")
        
        # 打印关键参数
        if optimizer_name == 'adam' or optimizer_name == 'adamw':
            print(f"   Beta1: {params.get('betas', (0.9, 0.999))[0]}")
            print(f"   Beta2: {params.get('betas', (0.9, 0.999))[1]}")
            print(f"   权重衰减: {params.get('weight_decay', 0)}")
        elif optimizer_name == 'sgd':
            print(f"   动量: {params.get('momentum', 0)}")
            print(f"   Nesterov: {params.get('nesterov', False)}")
            print(f"   权重衰减: {params.get('weight_decay', 0)}")
        elif optimizer_name == 'rmsprop':
            print(f"   Alpha: {params.get('alpha', 0.99)}")
            print(f"   动量: {params.get('momentum', 0)}")
            print(f"   权重衰减: {params.get('weight_decay', 0)}")
    
    @classmethod
    def get_available_optimizers(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有可用的优化器信息
        
        Returns:
            Dict: 优化器信息字典
        """
        return cls.OPTIMIZER_REGISTRY.copy()
    
    @classmethod
    def get_optimizer_info(cls, optimizer_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定优化器的信息
        
        Args:
            optimizer_name (str): 优化器名称
            
        Returns:
            Dict: 优化器信息，如果优化器不存在返回None
        """
        return cls.OPTIMIZER_REGISTRY.get(optimizer_name.lower())
    
    @classmethod
    def register_optimizer(cls, name: str, optimizer_class, display_name: str, 
                          description: str, default_params: Dict[str, Any]):
        """
        注册新的优化器类型
        
        Args:
            name (str): 优化器名称（用于配置文件）
            optimizer_class: 优化器类
            display_name (str): 显示名称
            description (str): 优化器描述
            default_params (Dict): 默认参数
        """
        cls.OPTIMIZER_REGISTRY[name.lower()] = {
            'class': optimizer_class,
            'name': display_name,
            'description': description,
            'default_params': default_params
        }
        print(f"✅ 已注册新优化器: {display_name} ({name})")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证优化器配置
        
        Args:
            config (Dict): 优化器配置
            
        Returns:
            bool: 配置是否有效
        """
        if isinstance(config, str):
            return config.lower() in cls.OPTIMIZER_REGISTRY
        
        if not isinstance(config, dict):
            print("❌ 优化器配置必须是字符串或字典")
            return False
        
        if 'name' not in config:
            print("❌ 优化器配置缺少 'name' 字段")
            return False
        
        optimizer_name = config['name'].lower()
        if optimizer_name not in cls.OPTIMIZER_REGISTRY:
            available_optimizers = list(cls.OPTIMIZER_REGISTRY.keys())
            print(f"❌ 不支持的优化器类型: {optimizer_name}")
            print(f"   支持的优化器: {available_optimizers}")
            return False
        
        # 验证学习率
        params = config.get('params', {})
        if 'lr' in params and params['lr'] <= 0:
            print("❌ 学习率必须大于0")
            return False
        
        return True


# 便捷函数
def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    便捷的优化器创建函数
    
    Args:
        model (nn.Module): 要优化的模型
        config (Dict): 优化器配置
        
    Returns:
        optim.Optimizer: 创建的优化器
    """
    return OptimizerFactory.create_optimizer(model, config)


def get_available_optimizers() -> Dict[str, Dict[str, Any]]:
    """获取所有可用优化器"""
    return OptimizerFactory.get_available_optimizers()


def register_custom_optimizer(name: str, optimizer_class, display_name: str, 
                             description: str, default_params: Dict[str, Any]):
    """注册自定义优化器"""
    OptimizerFactory.register_optimizer(name, optimizer_class, display_name, description, default_params)
