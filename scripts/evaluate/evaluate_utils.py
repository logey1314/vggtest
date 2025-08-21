"""
评估工具函数
用于处理路径、时间戳提取和模型文件查找等功能
"""

import os
import re
import glob
import yaml
from typing import Optional, Tuple, Dict


def extract_timestamp_from_path(path: str) -> Optional[str]:
    """
    从路径中提取时间戳

    Args:
        path (str): 包含时间戳的路径

    Returns:
        str: 提取的时间戳，格式为 YYYYMMDD_HHMMSS，如果未找到返回None

    Examples:
        >>> extract_timestamp_from_path("/var/yjs/zes/vgg/outputs/logs/train_20250820_074155/")
        "20250820_074155"
        >>> extract_timestamp_from_path("train_20250820_074155")
        "20250820_074155"
    """
    # 匹配时间戳格式：YYYYMMDD_HHMMSS
    timestamp_pattern = r'(\d{8}_\d{6})'
    match = re.search(timestamp_pattern, path)

    if match:
        return match.group(1)
    else:
        print(f"⚠️  无法从路径中提取时间戳: {path}")
        return None


def validate_train_path(train_path: str) -> bool:
    """
    验证训练路径是否有效

    Args:
        train_path (str): 训练路径

    Returns:
        bool: 路径是否有效
    """
    if not train_path:
        return False

    # 检查路径是否存在
    if not os.path.exists(train_path):
        print(f"❌ 指定的训练路径不存在: {train_path}")
        return False

    # 检查是否包含时间戳
    timestamp = extract_timestamp_from_path(train_path)
    if not timestamp:
        print(f"❌ 路径中未找到有效的时间戳格式: {train_path}")
        return False

    return True


def find_model_file(project_root: str, timestamp: str) -> Optional[str]:
    """
    根据时间戳查找对应的模型文件

    Args:
        project_root (str): 项目根目录
        timestamp (str): 时间戳，格式为 YYYYMMDD_HHMMSS

    Returns:
        str: 模型文件路径，如果未找到返回None
    """
    # 可能的模型文件位置
    possible_paths = [
        # 1. 时间戳目录中的最佳模型
        os.path.join(project_root, 'models', 'checkpoints', f'train_{timestamp}', 'best_model.pth'),
        # 2. 时间戳目录中的检查点文件（查找最新的）
        os.path.join(project_root, 'models', 'checkpoints', f'train_{timestamp}'),
        # 3. 恢复训练目录中的模型
        os.path.join(project_root, 'models', 'checkpoints', f'resume_{timestamp}', 'best_model.pth'),
    ]

    # 检查直接的最佳模型文件
    for path in possible_paths[:1] + possible_paths[2:]:
        if os.path.exists(path):
            print(f"✅ 找到模型文件: {path}")
            return path

    # 检查检查点目录中的文件
    checkpoint_dir = possible_paths[1]
    if os.path.exists(checkpoint_dir):
        # 查找最佳模型
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"✅ 找到模型文件: {best_model_path}")
            return best_model_path

        # 查找检查点文件
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
        if checkpoint_files:
            # 按epoch数排序，选择最新的
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            print(f"✅ 找到检查点文件: {latest_checkpoint}")
            return latest_checkpoint

    print(f"❌ 未找到时间戳 {timestamp} 对应的模型文件")
    print(f"   检查的路径包括:")
    for path in possible_paths:
        print(f"   - {path}")

    return None


def get_corresponding_model_path(train_path: str, project_root: str) -> Optional[str]:
    """
    根据训练路径获取对应的模型文件路径

    Args:
        train_path (str): 训练路径
        project_root (str): 项目根目录

    Returns:
        str: 模型文件路径，如果未找到返回None
    """
    # 验证训练路径
    if not validate_train_path(train_path):
        return None

    # 提取时间戳
    timestamp = extract_timestamp_from_path(train_path)
    if not timestamp:
        return None

    # 查找模型文件
    model_path = find_model_file(project_root, timestamp)
    return model_path


def generate_eval_timestamp(train_timestamp: str) -> str:
    """
    根据训练时间戳生成评估时间戳
    
    Args:
        train_timestamp (str): 训练时间戳，格式为 YYYYMMDD_HHMMSS
        
    Returns:
        str: 评估时间戳，格式为 eval_YYYYMMDD_HHMMSS
    """
    return f"eval_{train_timestamp}"


def find_training_config(train_path: str, project_root: str) -> Optional[str]:
    """
    查找训练时保存的配置文件

    Args:
        train_path (str): 训练路径
        project_root (str): 项目根目录

    Returns:
        str: 配置文件路径，如果未找到返回None
    """
    # 提取时间戳
    timestamp = extract_timestamp_from_path(train_path)
    if not timestamp:
        return None

    # 可能的配置文件位置
    possible_config_paths = [
        # 1. 日志目录中的配置备份
        os.path.join(project_root, 'outputs', 'logs', f'train_{timestamp}', 'config_backup.yaml'),
        # 2. 恢复训练的日志目录
        os.path.join(project_root, 'outputs', 'logs', f'resume_{timestamp}', 'config_backup.yaml'),
        # 3. 检查点目录中的配置备份
        os.path.join(project_root, 'models', 'checkpoints', f'train_{timestamp}', 'config_backup.yaml'),
    ]

    for config_path in possible_config_paths:
        if os.path.exists(config_path):
            print(f"✅ 找到训练时配置文件: {config_path}")
            return config_path

    print(f"⚠️  未找到训练时配置文件，将使用默认配置")
    print(f"   检查的路径包括:")
    for path in possible_config_paths:
        print(f"   - {path}")

    return None


def load_training_config(train_path: str, project_root: str) -> Tuple[Optional[Dict], str]:
    """
    加载训练时的配置文件

    Args:
        train_path (str): 训练路径
        project_root (str): 项目根目录

    Returns:
        Tuple[Dict, str]: (配置字典, 配置文件来源说明)
    """
    config = None
    config_source = ""

    if train_path:
        # 优先查找训练时保存的配置
        training_config_path = find_training_config(train_path, project_root)
        if training_config_path:
            try:
                with open(training_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                config_source = f"训练时配置: {training_config_path}"
                print(f"📄 使用训练时保存的配置文件")
            except Exception as e:
                print(f"❌ 训练时配置文件读取失败: {e}")

    # 如果没有找到训练时配置，使用默认配置
    if config is None:
        default_config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
        try:
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            config_source = f"默认配置: {default_config_path}"
            print(f"📄 使用默认配置文件")
            print(f"⚠️  注意: 默认配置可能与训练时配置不同，可能影响评估准确性")
        except Exception as e:
            print(f"❌ 默认配置文件读取失败: {e}")
            return None, "配置文件读取失败"

    return config, config_source


def print_path_info(train_path: str, model_path: str, eval_dir: str, config_source: str = ""):
    """
    打印路径信息摘要

    Args:
        train_path (str): 训练路径
        model_path (str): 模型文件路径
        eval_dir (str): 评估结果目录
        config_source (str): 配置文件来源
    """
    print("📋 路径信息摘要:")
    print("=" * 60)
    print(f"🎯 指定训练路径: {train_path}")
    print(f"🤖 对应模型文件: {model_path}")
    print(f"📄 配置文件来源: {config_source}")
    print(f"💾 评估结果目录: {eval_dir}")
    print("=" * 60)
