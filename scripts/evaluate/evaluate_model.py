"""
模型评估主脚本
对训练好的混凝土坍落度分类模型进行全面评估
支持指定训练轮次路径进行评估
"""

import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入工厂系统
from src.models import create_model
from src.data.dataset import DataGenerator
from src.evaluation import ReportGenerator
from src.utils.font_manager import setup_matplotlib_font, get_font_manager
from scripts.evaluate.evaluate_utils import (
    get_corresponding_model_path,
    extract_timestamp_from_path,
    generate_eval_timestamp,
    print_path_info,
    load_training_config
)



def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None


def load_model(model_path, config, device='cpu'):
    """
    根据配置文件加载训练好的模型

    Args:
        model_path (str): 模型文件路径
        config (dict): 训练配置文件
        device (str): 设备类型

    Returns:
        torch.nn.Module: 加载的模型
    """
    print(f"📥 加载模型: {model_path}")

    # 从配置文件获取模型信息
    model_config = config['model'].copy()
    model_config['num_classes'] = config['data']['num_classes']
    model_config['pretrained'] = False  # 评估时不需要预训练权重

    print(f"🤖 模型类型: {model_config['name']}")
    print(f"📊 分类数量: {model_config['num_classes']}")

    # 使用工厂系统创建模型
    try:
        model = create_model(model_config)
        print(f"✅ 模型创建成功: {model.get_name() if hasattr(model, 'get_name') else model_config['name']}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

    # 加载权重
    try:
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
        else:
            state_dict = torch.load(model_path)

        # 处理不同的保存格式
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            # 检查点格式
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"📋 加载检查点格式权重")
        else:
            # 直接模型权重格式
            model.load_state_dict(state_dict)
            print(f"📋 加载直接权重格式")

        model.to(device)
        model.eval()

        print(f"✅ 模型权重加载成功")
        return model

    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return None


def create_evaluation_dataloader(config, project_root):
    """
    创建评估数据加载器 - 使用测试集进行最终评估
    
    Args:
        config (dict): 配置字典
        project_root (str): 项目根目录
        
    Returns:
        DataLoader: 评估数据加载器
    """
    # 使用测试集标注文件进行评估
    # 兼容性处理：如果配置中没有test_annotation_file，使用默认路径
    if 'test_annotation_file' in config['data']:
        annotation_file = config['data']['test_annotation_file']
        dataset_type = "测试集"
        print(f"📊 评估模式: 使用测试集进行最终性能评估")
    else:
        # 兼容旧配置：回退到默认测试集路径
        annotation_file = "data/annotations/cls_test.txt"
        dataset_type = "测试集（默认路径）"
        print(f"📊 评估模式: 使用测试集进行最终性能评估（兼容模式）")
        print(f"⚠️  配置文件中未找到test_annotation_file，使用默认路径")
    
    annotation_path = os.path.join(project_root, annotation_file)
    print(f"📁 评估数据集: {dataset_type}")
    print(f"📄 标注文件: {annotation_path}")
    
    if not os.path.exists(annotation_path):
        print(f"❌ 标注文件不存在: {annotation_path}")
        return None
    
    # 读取标注文件
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    if len(lines) == 0:
        print(f"❌ 标注文件为空: {annotation_path}")
        return None
    
    print(f"📊 加载标注文件: {annotation_path}")
    print(f"   总样本数: {len(lines)}")
    
    # 创建数据集（评估时不使用数据增强）
    input_shape = config['data']['input_shape']
    dataset = DataGenerator(lines, input_shape, random=False)
    
    # 创建数据加载器
    batch_size = config['dataloader']['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 评估时不打乱
        num_workers=0,  # 评估时使用单线程
        drop_last=False
    )
    
    return dataloader


def evaluate_model(model, dataloader, device='cpu'):
    """
    评估模型性能
    
    Args:
        model (torch.nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        device (str): 设备类型
        
    Returns:
        tuple: (y_true, y_pred, y_prob, image_paths)
    """
    model.eval()
    
    y_true = []
    y_pred = []
    y_prob = []
    image_paths = []
    
    print("🔍 开始模型评估...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, paths) in enumerate(tqdm(dataloader, desc="评估进度")):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            
            # 收集结果
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())
            image_paths.extend(paths)
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob), image_paths


def load_training_config(train_path, project_root):
    """
    加载训练时的配置文件

    Args:
        train_path (str): 训练路径
        project_root (str): 项目根目录

    Returns:
        tuple: (config_dict, config_source)
    """
    config = None
    config_source = None

    if train_path:
        # 尝试从训练路径加载配置备份
        config_backup_path = os.path.join(train_path, 'config_backup.yaml')
        if os.path.exists(config_backup_path):
            try:
                with open(config_backup_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                config_source = f"训练配置备份: {config_backup_path}"
                print(f"✅ 从训练配置备份加载: {config_backup_path}")
            except Exception as e:
                print(f"⚠️  配置备份加载失败: {e}")

    # 如果没有找到配置备份，使用默认配置文件
    if config is None:
        default_config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
        if os.path.exists(default_config_path):
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                config_source = f"默认配置文件: {default_config_path}"
                print(f"✅ 从默认配置文件加载: {default_config_path}")
            except Exception as e:
                print(f"❌ 默认配置文件加载失败: {e}")
                return None, None

    if config:
        print(f"📋 配置来源: {config_source}")
        print(f"🤖 模型类型: {config['model']['name']}")
        print(f"📊 分类数量: {config['data']['num_classes']}")
        print(f"📏 输入尺寸: {config['data']['input_shape']}")

    return config, config_source


def main():
    """主评估函数"""

    # ==================== 配置参数 ====================
    # 🎯 指定要评估的训练轮次路径（在这里修改）
    # 设置为具体路径来评估指定轮次，设置为 None 使用默认的 latest 模型
    SPECIFIC_TRAIN_PATH = "/var/yjs/zes/vgg/outputs/logs/train_20250914_112502"

    # SPECIFIC_TRAIN_PATH = None  # 使用默认的 latest 模型

    # ==================== 字体设置 ====================
    print("🔤 设置字体...")
    setup_matplotlib_font()
    font_manager = get_font_manager()
    font_manager.print_font_status()

    # ==================== 路径设置 ====================
    # 获取项目路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # 上两级目录
    
    # 修复路径问题：如果路径以 /root 开头，去掉 /root 前缀
    if project_root.startswith('/root/'):
        project_root = project_root[6:]  # 去掉 '/root' 前缀
        print(f"🔧 修复项目根目录路径: {project_root}")

    # 根据配置确定模型路径和评估目录
    if SPECIFIC_TRAIN_PATH:
        print(f"🎯 指定训练轮次模式")
        print(f"   训练路径: {SPECIFIC_TRAIN_PATH}")

        # 获取对应的模型文件路径
        model_path = get_corresponding_model_path(SPECIFIC_TRAIN_PATH, project_root)
        if not model_path:
            print("❌ 无法找到对应的模型文件，程序退出")
            return

        # 提取时间戳并生成评估目录名
        timestamp = extract_timestamp_from_path(SPECIFIC_TRAIN_PATH)
        eval_timestamp = generate_eval_timestamp(timestamp)

    else:
        print(f"📋 默认模式：使用 latest 模型")
        # 使用默认的 latest 模型
        model_path = os.path.join(project_root, 'models', 'checkpoints', 'latest', 'best_model.pth')
        eval_timestamp = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")

    # ==================== 确定评估类型和保存目录 ====================
    # 使用测试集评估，但保存到evaluation目录（保持兼容性）
    base_eval_dir = os.path.join(project_root, 'outputs', 'evaluation')
    eval_type = "测试集"
    
    save_dir = os.path.join(base_eval_dir, eval_timestamp)
    latest_eval_dir = os.path.join(base_eval_dir, "latest")

    # ==================== 加载配置 ====================
    print("📄 加载配置文件...")
    config, config_source = load_training_config(SPECIFIC_TRAIN_PATH, project_root)
    if config is None:
        print("❌ 无法加载配置文件，程序退出")
        return

    # 打印路径信息
    if SPECIFIC_TRAIN_PATH:
        print_path_info(SPECIFIC_TRAIN_PATH, model_path, save_dir, config_source)

    print("🚀 开始模型评估")
    print("="*80)
    print(f"📊 评估类型: {eval_type}最终评估")
    print(f"📁 项目根目录: {project_root}")
    print(f"📄 配置文件来源: {config_source}")
    print(f"🤖 模型文件: {model_path}")
    print(f"💾 保存目录: {save_dir}")
    print("="*80)

    # ==================== 设备检查 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    # ==================== 检查模型文件 ====================
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先完成模型训练或检查模型路径")
        return

    # ==================== 加载模型 ====================
    model = load_model(model_path, config, device)
    if model is None:
        return

    # ==================== 创建数据加载器 ====================
    dataloader = create_evaluation_dataloader(config, project_root)
    if dataloader is None:
        return

    print(f"📊 数据集信息:")
    print(f"   总样本数: {len(dataloader.dataset)}")
    print(f"   批次大小: {dataloader.batch_size}")
    print(f"   批次数量: {len(dataloader)}")

    # ==================== 模型评估 ====================
    y_true, y_pred, y_prob, image_paths = evaluate_model(model, dataloader, device)

    print(f"✅ 评估完成")
    print(f"   总样本数: {len(y_true)}")
    print(f"   预测准确率: {np.mean(y_true == y_pred):.4f}")

    # ==================== 生成评估报告 ====================
    print("📋 准备生成评估报告...")
    class_names = config['data']['class_names']
    model_type = config['model']['name'].upper()
    model_name = f"{model_type}-{config['data']['num_classes']}类"

    print(f"📊 类别数量: {len(class_names)}")
    print(f"🤖 模型名称: {model_name}")

    # 初始化报告生成器
    print("🔧 初始化报告生成器...")
    report_generator = ReportGenerator(class_names, model_name)

    # 生成综合评估报告（保存到时间戳目录）
    print(f"📋 开始生成综合评估报告...")
    print(f"💾 保存目录: {save_dir}")
    
    try:
        results = report_generator.generate_comprehensive_report(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            image_paths=image_paths,
            save_dir=save_dir
        )
        print("✅ 综合评估报告生成完成")
    except Exception as e:
        print(f"❌ 生成综合评估报告时出错: {e}")
        print("🔍 尝试生成简化报告...")
        # 如果综合报告失败，至少输出基本指标
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_true, y_pred)
        print(f"整体准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        return

    # 同时保存到latest目录
    print(f"\n📋 同时保存到latest目录: {latest_eval_dir}")
    report_generator.generate_comprehensive_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        image_paths=image_paths,
        save_dir=latest_eval_dir
    )

    # 特征可视化分析已移至独立脚本 scripts/evaluate/feature_visualization.py
    # 如需进行特征分析，请单独运行该脚本

    # ==================== 输出关键结果 ====================
    print("\n" + "="*80)
    print(f"📊 {eval_type}评估结果摘要")
    print("="*80)
    print(f"整体准确率: {results['metrics']['accuracy']:.4f}")
    print(f"宏平均F1-Score: {results['metrics']['f1_macro']:.4f}")
    print(f"错误样本数: {results['error_info']['total_errors']}")
    print(f"错误率: {results['error_info']['error_rate']:.4f}")

    if 'top2_accuracy' in results['metrics']:
        print(f"Top-2准确率: {results['metrics']['top2_accuracy']:.4f}")

    print(f"相邻类别混淆率: {results['metrics']['adjacent_confusion_rate']:.4f}")
    print(f"严重错误率: {results['metrics']['severe_error_rate']:.4f}")

    print("\n各类别性能:")
    for class_name in class_names:
        precision = results['metrics'][f'precision_{class_name}']
        recall = results['metrics'][f'recall_{class_name}']
        f1 = results['metrics'][f'f1_{class_name}']
        accuracy = results['metrics'][f'accuracy_{class_name}']
        print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, A={accuracy:.3f}")

    print("="*80)
    print("🎉 评估完成！详细结果请查看保存的文件。")


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行

    使用方法:
    1. 修改 main 函数中的 SPECIFIC_TRAIN_PATH 变量
    2. 设置为具体的训练路径来评估指定轮次
    3. 设置为 None 来使用默认的 latest 模型
    4. 在 PyCharm 中直接运行此脚本
    """
    main()
