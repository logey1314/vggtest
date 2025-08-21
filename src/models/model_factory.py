"""
模型工厂
统一的模型创建接口，支持多种深度学习模型架构
"""

import torch.nn as nn
from typing import Dict, Any, Optional
from .vgg import vgg16
from .resnet import resnet18_concrete, resnet50_concrete


class ModelFactory:
    """模型工厂类"""
    
    # 模型注册表
    MODEL_REGISTRY = {
        'vgg16': {
            'class': vgg16,
            'name': 'VGG16',
            'description': '经典的VGG16架构，适合图像分类任务'
        },
        'resnet18': {
            'class': resnet18_concrete,
            'name': 'ResNet18',
            'description': '轻量级ResNet架构，训练速度快'
        },
        'resnet50': {
            'class': resnet50_concrete,
            'name': 'ResNet50',
            'description': '标准ResNet架构，性能与效率平衡'
        }
    }
    
    @classmethod
    def create_model(cls, config: Dict[str, Any]) -> nn.Module:
        """
        根据配置创建模型
        
        Args:
            config (Dict): 模型配置字典
            
        Returns:
            nn.Module: 创建的模型实例
            
        Raises:
            ValueError: 当模型类型不支持时
            
        Example:
            config = {
                'name': 'resnet50',
                'pretrained': True,
                'dropout': 0.5,
                'num_classes': 4,
                'resnet': {
                    'replace_stride_with_dilation': [False, False, False],
                    'zero_init_residual': False
                }
            }
            model = ModelFactory.create_model(config)
        """
        model_name = config.get('name', '').lower()
        
        if model_name not in cls.MODEL_REGISTRY:
            available_models = list(cls.MODEL_REGISTRY.keys())
            raise ValueError(f"不支持的模型类型: {model_name}. "
                           f"支持的模型: {available_models}")
        
        # 获取模型创建函数
        model_class = cls.MODEL_REGISTRY[model_name]['class']
        
        # 准备基础参数
        base_params = {
            'pretrained': config.get('pretrained', True),
            'num_classes': config.get('num_classes', 3),
            'dropout': config.get('dropout', 0.5)
        }
        
        # 添加模型特定参数
        model_specific_params = {}
        
        if model_name.startswith('vgg'):
            # VGG特定参数
            vgg_config = config.get('vgg', {})
            model_specific_params.update({
                'progress': vgg_config.get('progress', True),
                'model_dir': vgg_config.get('model_dir', None)
            })
            
        elif model_name.startswith('resnet'):
            # ResNet特定参数
            resnet_config = config.get('resnet', {})
            model_specific_params.update({
                'replace_stride_with_dilation': resnet_config.get('replace_stride_with_dilation', None),
                'zero_init_residual': resnet_config.get('zero_init_residual', False)
            })
        
        # 合并所有参数
        all_params = {**base_params, **model_specific_params}
        
        # 创建模型
        try:
            model = model_class(**all_params)
            
            # 打印模型信息
            cls._print_model_info(model, config)
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"创建模型 {model_name} 失败: {e}")
    
    @classmethod
    def _print_model_info(cls, model: nn.Module, config: Dict[str, Any]):
        """打印模型信息"""
        model_name = config.get('name', 'Unknown').upper()
        
        print(f"\n🤖 模型配置:")
        print(f"   模型类型: {cls.MODEL_REGISTRY[config['name'].lower()]['name']}")
        print(f"   预训练: {'是' if config.get('pretrained', True) else '否'}")
        print(f"   分类数量: {config.get('num_classes', 3)}")
        print(f"   Dropout: {config.get('dropout', 0.5)}")
        
        # 获取参数数量
        if hasattr(model, 'get_parameter_count'):
            param_info = model.get_parameter_count()
            total_params = param_info['total']
            trainable_params = param_info['trainable']
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 打印模型特定信息
        if hasattr(model, 'get_info'):
            model_info = model.get_info()
            if 'variant' in model_info:
                print(f"   模型变体: {model_info['variant'].upper()}")
            if 'feature_dim' in model_info:
                print(f"   特征维度: {model_info['feature_dim']}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, str]]:
        """
        获取所有可用的模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        return cls.MODEL_REGISTRY.copy()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, str]]:
        """
        获取指定模型的信息
        
        Args:
            model_name (str): 模型名称
            
        Returns:
            Dict: 模型信息，如果模型不存在返回None
        """
        return cls.MODEL_REGISTRY.get(model_name.lower())
    
    @classmethod
    def register_model(cls, name: str, model_class, display_name: str, description: str):
        """
        注册新的模型类型
        
        Args:
            name (str): 模型名称（用于配置文件）
            model_class: 模型类或创建函数
            display_name (str): 显示名称
            description (str): 模型描述
        """
        cls.MODEL_REGISTRY[name.lower()] = {
            'class': model_class,
            'name': display_name,
            'description': description
        }
        print(f"✅ 已注册新模型: {display_name} ({name})")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证模型配置
        
        Args:
            config (Dict): 模型配置
            
        Returns:
            bool: 配置是否有效
        """
        if 'name' not in config:
            print("❌ 模型配置缺少 'name' 字段")
            return False
        
        model_name = config['name'].lower()
        if model_name not in cls.MODEL_REGISTRY:
            available_models = list(cls.MODEL_REGISTRY.keys())
            print(f"❌ 不支持的模型类型: {model_name}")
            print(f"   支持的模型: {available_models}")
            return False
        
        # 验证基础参数
        if 'num_classes' in config and config['num_classes'] <= 0:
            print("❌ num_classes 必须大于0")
            return False
        
        if 'dropout' in config and not (0 <= config['dropout'] <= 1):
            print("❌ dropout 必须在0-1之间")
            return False
        
        return True


# 便捷函数
def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    便捷的模型创建函数
    
    Args:
        config (Dict): 模型配置
        
    Returns:
        nn.Module: 创建的模型
    """
    return ModelFactory.create_model(config)


def get_available_models() -> Dict[str, Dict[str, str]]:
    """获取所有可用模型"""
    return ModelFactory.get_available_models()


def register_custom_model(name: str, model_class, display_name: str, description: str):
    """注册自定义模型"""
    ModelFactory.register_model(name, model_class, display_name, description)
