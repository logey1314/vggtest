"""
训练工具函数
提供训练管理相关的通用功能
"""

import os
import yaml
import datetime
import logging
from typing import Dict, Tuple, Optional, List


def validate_resume_config(config: Dict) -> Tuple[bool, str]:
    """
    验证恢复训练配置
    
    Args:
        config (Dict): 配置字典
        
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    """
    resume_config = config.get('resume', {})
    
    # 检查是否启用恢复训练
    if not resume_config.get('enabled', False):
        return False, "恢复训练未启用，请在配置文件中设置 resume.enabled: true"
    
    # 检查必要的配置项
    required_keys = ['training', 'data', 'model', 'dataloader']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        return False, f"配置文件缺少必要项: {missing_keys}"
    
    # 检查训练配置
    training_config = config['training']
    if 'epochs' not in training_config or training_config['epochs'] <= 0:
        return False, "训练轮数配置无效，请设置 training.epochs 为正整数"
    
    # 检查数据配置
    data_config = config['data']
    required_data_keys = ['num_classes', 'class_names', 'annotation_file']
    missing_data_keys = [key for key in required_data_keys if key not in data_config]
    if missing_data_keys:
        return False, f"数据配置缺少必要项: {missing_data_keys}"
    
    # 检查类别数量与类别名称是否匹配
    num_classes = data_config['num_classes']
    class_names = data_config['class_names']
    if len(class_names) != num_classes:
        return False, f"类别数量({num_classes})与类别名称数量({len(class_names)})不匹配"
    
    return True, "配置验证通过"


def setup_training_directories(project_root: str, config: Dict, 
                              prefix: str = "resume") -> Dict[str, str]:
    """
    设置训练相关目录
    
    Args:
        project_root (str): 项目根目录
        config (Dict): 配置字典
        prefix (str): 目录前缀
        
    Returns:
        Dict[str, str]: 目录路径字典
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 基础目录
    base_checkpoint_dir = os.path.join(project_root, config['save']['checkpoint_dir'])
    base_log_dir = os.path.join(project_root, config['logging']['log_dir'])
    base_plot_dir = os.path.join(project_root, config['logging']['plot_dir'])
    
    # 时间戳目录
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{prefix}_{timestamp}")
    log_dir = os.path.join(base_log_dir, f"{prefix}_{timestamp}")
    plot_dir = os.path.join(base_plot_dir, f"{prefix}_{timestamp}")
    
    # latest目录
    latest_checkpoint_dir = os.path.join(base_checkpoint_dir, "latest")
    latest_log_dir = os.path.join(base_log_dir, "latest")
    latest_plot_dir = os.path.join(base_plot_dir, "latest")
    
    # 创建所有目录
    directories = {
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'plot_dir': plot_dir,
        'latest_checkpoint_dir': latest_checkpoint_dir,
        'latest_log_dir': latest_log_dir,
        'latest_plot_dir': latest_plot_dir,
        'timestamp': timestamp
    }
    
    for dir_path in [checkpoint_dir, log_dir, plot_dir, 
                     latest_checkpoint_dir, latest_log_dir, latest_plot_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def backup_config_file(config: Dict, backup_dir: str, 
                      filename: str = "config_backup.yaml") -> str:
    """
    备份配置文件
    
    Args:
        config (Dict): 配置字典
        backup_dir (str): 备份目录
        filename (str): 备份文件名
        
    Returns:
        str: 备份文件路径
    """
    backup_path = os.path.join(backup_dir, filename)
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return backup_path


def setup_training_logger(log_dir: str, timestamp: str, 
                         script_name: str = "training") -> logging.Logger:
    """
    设置训练日志记录器
    
    Args:
        log_dir (str): 日志目录
        timestamp (str): 时间戳
        script_name (str): 脚本名称
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    log_file = os.path.join(log_dir, f'{script_name}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(script_name)
    logger.info("="*80)
    logger.info(f"🔄 {script_name} 开始")
    logger.info("="*80)
    logger.info(f"训练时间戳: {timestamp}")
    logger.info(f"日志文件: {log_file}")
    
    return logger


def print_training_summary(config: Dict, directories: Dict, 
                          checkpoint_path: str = None):
    """
    打印训练摘要信息
    
    Args:
        config (Dict): 配置字典
        directories (Dict): 目录字典
        checkpoint_path (str): 检查点路径
    """
    print(f"📊 训练配置摘要:")
    print(f"   目标轮数: {config['training']['epochs']}")
    print(f"   学习率: {config['training']['learning_rate']}")
    print(f"   批次大小: {config['dataloader']['batch_size']}")
    print(f"   类别数量: {config['data']['num_classes']}")
    print(f"   优化器: {config['training']['optimizer']}")
    print(f"   调度器: {config['training']['scheduler']['type']}")
    
    if checkpoint_path:
        print(f"\n🎯 检查点信息:")
        print(f"   检查点文件: {checkpoint_path}")
    
    print(f"\n💾 输出目录:")
    print(f"   检查点: {directories['checkpoint_dir']}")
    print(f"   日志: {directories['log_dir']}")
    print(f"   图表: {directories['plot_dir']}")


def check_training_prerequisites(project_root: str, config: Dict) -> Tuple[bool, List[str]]:
    """
    检查训练前置条件
    
    Args:
        project_root (str): 项目根目录
        config (Dict): 配置字典
        
    Returns:
        Tuple[bool, List[str]]: (是否满足条件, 错误信息列表)
    """
    errors = []
    
    # 检查标注文件
    annotation_path = os.path.join(project_root, config['data']['annotation_file'])
    if not os.path.exists(annotation_path):
        errors.append(f"标注文件不存在: {annotation_path}")
    
    # 检查数据目录
    data_dir = os.path.join(project_root, config['data']['data_dir'])
    if not os.path.exists(data_dir):
        errors.append(f"数据目录不存在: {data_dir}")
    
    # 检查配置文件必要项
    required_sections = ['training', 'data', 'model', 'dataloader', 'save', 'logging']
    for section in required_sections:
        if section not in config:
            errors.append(f"配置文件缺少 {section} 部分")
    
    # 检查保存目录权限
    try:
        checkpoint_base = os.path.join(project_root, config['save']['checkpoint_dir'])
        os.makedirs(checkpoint_base, exist_ok=True)
        
        log_base = os.path.join(project_root, config['logging']['log_dir'])
        os.makedirs(log_base, exist_ok=True)
        
        plot_base = os.path.join(project_root, config['logging']['plot_dir'])
        os.makedirs(plot_base, exist_ok=True)
        
    except PermissionError as e:
        errors.append(f"目录创建权限不足: {e}")
    except Exception as e:
        errors.append(f"目录创建失败: {e}")
    
    return len(errors) == 0, errors


def format_time_duration(seconds: float) -> str:
    """
    格式化时间持续时间
    
    Args:
        seconds (float): 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def calculate_eta(current_epoch: int, total_epochs: int, 
                 elapsed_time: float) -> str:
    """
    计算预计剩余时间
    
    Args:
        current_epoch (int): 当前轮数
        total_epochs (int): 总轮数
        elapsed_time (float): 已用时间（秒）
        
    Returns:
        str: 预计剩余时间字符串
    """
    if current_epoch == 0:
        return "计算中..."
    
    avg_time_per_epoch = elapsed_time / current_epoch
    remaining_epochs = total_epochs - current_epoch
    eta_seconds = avg_time_per_epoch * remaining_epochs
    
    return format_time_duration(eta_seconds)


def get_training_status_summary(current_epoch: int, total_epochs: int,
                               train_acc: float, val_acc: float,
                               best_acc: float, elapsed_time: float) -> str:
    """
    获取训练状态摘要
    
    Args:
        current_epoch (int): 当前轮数
        total_epochs (int): 总轮数
        train_acc (float): 训练准确率
        val_acc (float): 验证准确率
        best_acc (float): 最佳准确率
        elapsed_time (float): 已用时间
        
    Returns:
        str: 状态摘要字符串
    """
    progress = (current_epoch / total_epochs) * 100
    eta = calculate_eta(current_epoch, total_epochs, elapsed_time)
    elapsed_str = format_time_duration(elapsed_time)
    
    summary = f"""
📊 训练状态摘要 (Epoch {current_epoch}/{total_epochs})
{'='*50}
进度: {progress:.1f}% 
训练准确率: {train_acc:.2f}%
验证准确率: {val_acc:.2f}%
最佳准确率: {best_acc:.2f}%
已用时间: {elapsed_str}
预计剩余: {eta}
{'='*50}
"""
    return summary
