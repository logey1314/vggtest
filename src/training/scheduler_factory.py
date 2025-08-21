"""
学习率调度器工厂
统一的学习率调度器创建接口，支持多种调度策略
"""

import torch.optim as optim
from typing import Dict, Any, Optional, Union


class SchedulerFactory:
    """学习率调度器工厂类"""
    
    # 调度器注册表
    SCHEDULER_REGISTRY = {
        'steplr': {
            'class': optim.lr_scheduler.StepLR,
            'name': 'StepLR',
            'description': '阶梯式学习率衰减',
            'default_params': {
                'step_size': 7,
                'gamma': 0.5,
                'last_epoch': -1
            },
            'requires_metric': False,
            'supported_params': ['step_size', 'gamma', 'last_epoch']
        },
        'multisteplr': {
            'class': optim.lr_scheduler.MultiStepLR,
            'name': 'MultiStepLR',
            'description': '多阶梯学习率衰减',
            'default_params': {
                'milestones': [10, 15],
                'gamma': 0.1,
                'last_epoch': -1
            },
            'requires_metric': False,
            'supported_params': ['milestones', 'gamma', 'last_epoch']
        },
        'cosineannealinglr': {
            'class': optim.lr_scheduler.CosineAnnealingLR,
            'name': 'CosineAnnealingLR',
            'description': '余弦退火学习率调度',
            'default_params': {
                'T_max': 20,
                'eta_min': 0,
                'last_epoch': -1
            },
            'requires_metric': False,
            'supported_params': ['T_max', 'eta_min', 'last_epoch']
        },
        'reducelronplateau': {
            'class': optim.lr_scheduler.ReduceLROnPlateau,
            'name': 'ReduceLROnPlateau',
            'description': '基于指标的自适应学习率调度',
            'default_params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': 3,
                'threshold': 1e-4,
                'threshold_mode': 'rel',
                'cooldown': 0,
                'min_lr': 0,
                'eps': 1e-8
            },
            'requires_metric': True,
            'supported_params': ['mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lr', 'eps']
        },
        'exponentiallr': {
            'class': optim.lr_scheduler.ExponentialLR,
            'name': 'ExponentialLR',
            'description': '指数衰减学习率调度',
            'default_params': {
                'gamma': 0.95,
                'last_epoch': -1
            },
            'requires_metric': False,
            'supported_params': ['gamma', 'last_epoch']
        },
        'cosineannealingwarmrestarts': {
            'class': optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'name': 'CosineAnnealingWarmRestarts',
            'description': '带热重启的余弦退火',
            'default_params': {
                'T_0': 10,
                'T_mult': 1,
                'eta_min': 0,
                'last_epoch': -1
            },
            'requires_metric': False,
            'supported_params': ['T_0', 'T_mult', 'eta_min', 'last_epoch']
        }
    }
    
    @classmethod
    def create_scheduler(cls, optimizer: optim.Optimizer, config: Dict[str, Any], 
                        total_epochs: int = None) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        根据配置创建学习率调度器
        
        Args:
            optimizer (optim.Optimizer): 优化器
            config (Dict): 调度器配置
            total_epochs (int): 总训练轮数（用于自动设置T_max）
            
        Returns:
            optim.lr_scheduler._LRScheduler: 创建的调度器实例，如果为None则不使用调度器
            
        Example:
            config = {
                'name': 'CosineAnnealingLR',
                'params': {
                    'T_max': 'auto',
                    'eta_min': 1e-6
                }
            }
            scheduler = SchedulerFactory.create_scheduler(optimizer, config, total_epochs=100)
        """
        # 兼容旧配置格式
        if isinstance(config, str):
            if config.lower() == 'none' or config.lower() == 'null':
                return None
            scheduler_name = config.lower()
            scheduler_params = {}
        else:
            scheduler_name = config.get('name', '').lower()
            if scheduler_name == 'none' or scheduler_name == 'null' or scheduler_name == '':
                return None
            scheduler_params = config.get('params', {})
        
        if scheduler_name not in cls.SCHEDULER_REGISTRY:
            available_schedulers = list(cls.SCHEDULER_REGISTRY.keys())
            raise ValueError(f"不支持的调度器类型: {scheduler_name}. "
                           f"支持的调度器: {available_schedulers}")
        
        # 获取调度器信息
        scheduler_info = cls.SCHEDULER_REGISTRY[scheduler_name]
        scheduler_class = scheduler_info['class']
        default_params = scheduler_info['default_params'].copy()
        supported_params = scheduler_info['supported_params']

        # 处理特殊参数
        processed_params = cls._process_special_params(
            scheduler_name, default_params, scheduler_params, total_epochs
        )

        # 只保留该调度器支持的参数
        filtered_params = {k: v for k, v in processed_params.items() if k in supported_params}

        # 转换参数类型
        final_params = cls._convert_param_types(filtered_params)
        
        # 创建调度器
        try:
            scheduler = scheduler_class(optimizer, **final_params)
            
            # 打印调度器信息
            cls._print_scheduler_info(scheduler_name, final_params, scheduler_info['requires_metric'], supported_params)
            
            return scheduler
            
        except Exception as e:
            raise RuntimeError(f"创建调度器 {scheduler_name} 失败: {e}")
    
    @classmethod
    def _process_special_params(cls, scheduler_name: str, default_params: Dict[str, Any], 
                               user_params: Dict[str, Any], total_epochs: int = None) -> Dict[str, Any]:
        """处理特殊参数"""
        final_params = {**default_params, **user_params}
        
        # 处理CosineAnnealingLR的T_max自动设置
        if scheduler_name == 'cosineannealinglr':
            if final_params.get('T_max') == 'auto' and total_epochs:
                final_params['T_max'] = total_epochs
                print(f"🔄 自动设置 T_max = {total_epochs}")
            elif final_params.get('T_max') == 'auto':
                final_params['T_max'] = 20  # 默认值
                print(f"⚠️  未提供总轮数，使用默认 T_max = 20")
        
        # 处理CosineAnnealingWarmRestarts的T_0自动设置
        if scheduler_name == 'cosineannealingwarmrestarts':
            if final_params.get('T_0') == 'auto' and total_epochs:
                final_params['T_0'] = max(1, total_epochs // 10)  # 总轮数的1/10
                print(f"🔄 自动设置 T_0 = {final_params['T_0']}")
        
        # 确保eta_min是数值类型
        if 'eta_min' in final_params and isinstance(final_params['eta_min'], str):
            try:
                final_params['eta_min'] = float(final_params['eta_min'])
            except ValueError:
                final_params['eta_min'] = 0
        
        return final_params

    @classmethod
    def _convert_param_types(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """转换参数类型，确保数值参数是正确的类型"""
        converted_params = {}

        for key, value in params.items():
            try:
                if key in ['step_size', 'T_max', 'T_0', 'T_mult', 'patience', 'cooldown', 'last_epoch']:
                    # 整数参数
                    converted_params[key] = int(value)
                elif key in ['gamma', 'eta_min', 'factor', 'threshold', 'min_lr', 'eps']:
                    # 浮点数参数
                    converted_params[key] = float(value)
                elif key == 'milestones':
                    # 里程碑列表转换为整数列表
                    if isinstance(value, (list, tuple)):
                        converted_params[key] = [int(x) for x in value]
                    else:
                        converted_params[key] = value
                elif key in ['mode', 'threshold_mode']:
                    # 字符串参数
                    converted_params[key] = str(value)
                else:
                    # 其他参数保持原样
                    converted_params[key] = value

            except (ValueError, TypeError) as e:
                print(f"⚠️  调度器参数 {key} 类型转换失败，使用原值: {value} (错误: {e})")
                converted_params[key] = value

        return converted_params

    @classmethod
    def _print_scheduler_info(cls, scheduler_name: str, params: Dict[str, Any], requires_metric: bool, supported_params: list):
        """打印调度器信息"""
        scheduler_info = cls.SCHEDULER_REGISTRY[scheduler_name]
        
        print(f"\n📈 学习率调度器配置:")
        print(f"   调度器类型: {scheduler_info['name']}")
        print(f"   需要指标: {'是' if requires_metric else '否'}")
        print(f"   使用参数: {list(params.keys())}")

        # 打印关键参数
        if scheduler_name == 'steplr':
            if 'step_size' in params:
                print(f"   步长: {params['step_size']}")
            if 'gamma' in params:
                print(f"   衰减因子: {params['gamma']}")
        elif scheduler_name == 'multisteplr':
            if 'milestones' in params:
                print(f"   里程碑: {params['milestones']}")
            if 'gamma' in params:
                print(f"   衰减因子: {params['gamma']}")
        elif scheduler_name == 'cosineannealinglr':
            if 'T_max' in params:
                print(f"   T_max: {params['T_max']}")
            if 'eta_min' in params:
                print(f"   最小学习率: {params['eta_min']}")
        elif scheduler_name == 'reducelronplateau':
            if 'mode' in params:
                print(f"   模式: {params['mode']}")
            if 'factor' in params:
                print(f"   衰减因子: {params['factor']}")
            if 'patience' in params:
                print(f"   耐心值: {params['patience']}")
    
    @classmethod
    def get_available_schedulers(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有可用的调度器信息
        
        Returns:
            Dict: 调度器信息字典
        """
        return cls.SCHEDULER_REGISTRY.copy()
    
    @classmethod
    def get_scheduler_info(cls, scheduler_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定调度器的信息
        
        Args:
            scheduler_name (str): 调度器名称
            
        Returns:
            Dict: 调度器信息，如果调度器不存在返回None
        """
        return cls.SCHEDULER_REGISTRY.get(scheduler_name.lower())
    
    @classmethod
    def requires_metric(cls, scheduler_name: str) -> bool:
        """
        检查调度器是否需要指标
        
        Args:
            scheduler_name (str): 调度器名称
            
        Returns:
            bool: 是否需要指标
        """
        scheduler_info = cls.SCHEDULER_REGISTRY.get(scheduler_name.lower())
        return scheduler_info['requires_metric'] if scheduler_info else False
    
    @classmethod
    def register_scheduler(cls, name: str, scheduler_class, display_name: str, 
                          description: str, default_params: Dict[str, Any], requires_metric: bool = False):
        """
        注册新的调度器类型
        
        Args:
            name (str): 调度器名称（用于配置文件）
            scheduler_class: 调度器类
            display_name (str): 显示名称
            description (str): 调度器描述
            default_params (Dict): 默认参数
            requires_metric (bool): 是否需要指标
        """
        cls.SCHEDULER_REGISTRY[name.lower()] = {
            'class': scheduler_class,
            'name': display_name,
            'description': description,
            'default_params': default_params,
            'requires_metric': requires_metric
        }
        print(f"✅ 已注册新调度器: {display_name} ({name})")


# 便捷函数
def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], 
                    total_epochs: int = None) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    便捷的调度器创建函数
    
    Args:
        optimizer (optim.Optimizer): 优化器
        config (Dict): 调度器配置
        total_epochs (int): 总训练轮数
        
    Returns:
        optim.lr_scheduler._LRScheduler: 创建的调度器
    """
    return SchedulerFactory.create_scheduler(optimizer, config, total_epochs)


def get_available_schedulers() -> Dict[str, Dict[str, Any]]:
    """获取所有可用调度器"""
    return SchedulerFactory.get_available_schedulers()


def requires_metric(scheduler_name: str) -> bool:
    """检查调度器是否需要指标"""
    return SchedulerFactory.requires_metric(scheduler_name)
