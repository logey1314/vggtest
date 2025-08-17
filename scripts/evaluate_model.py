"""
模型评估主脚本
对训练好的混凝土坍落度分类模型进行全面评估
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.vgg import vgg16
from src.data.dataset import DataGenerator
from src.evaluation import ReportGenerator


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


def load_model(model_path, num_classes=3, device='cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型文件路径
        num_classes (int): 分类数量
        device (str): 设备类型
        
    Returns:
        torch.nn.Module: 加载的模型
    """
    print(f"📥 加载模型: {model_path}")
    
    # 创建模型（评估时dropout不影响结果，使用默认值即可）
    model = vgg16(pretrained=False, num_classes=num_classes)
    
    # 加载权重
    try:
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
        else:
            state_dict = torch.load(model_path)
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"✅ 模型加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def evaluate_model(model, dataloader, device='cpu'):
    """
    评估模型性能
    
    Args:
        model (torch.nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        device (str): 设备类型
        
    Returns:
        tuple: (真实标签, 预测标签, 预测概率, 图像路径)
    """
    print("🔍 开始模型评估...")
    
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    all_image_paths = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, image_paths) in enumerate(tqdm(dataloader, desc="评估进度")):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)
            
            # 收集结果
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
            all_pred_probs.extend(probabilities.cpu().numpy())
            all_image_paths.extend(image_paths)
    
    return (np.array(all_true_labels), 
            np.array(all_pred_labels), 
            np.array(all_pred_probs),
            all_image_paths)


def create_evaluation_dataloader(config, project_root):
    """
    创建评估数据加载器
    
    Args:
        config (dict): 配置字典
        project_root (str): 项目根目录
        
    Returns:
        DataLoader: 评估数据加载器
    """
    # 读取标注文件
    annotation_path = os.path.join(project_root, config['data']['annotation_file'])
    
    if not os.path.exists(annotation_path):
        print(f"❌ 标注文件不存在: {annotation_path}")
        return None
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析标注
    annotation_lines = []
    for line in lines:
        line = line.strip()
        if line:
            annotation_lines.append(line)
    
    print(f"📊 加载标注文件: {len(annotation_lines)} 个样本")
    
    # 创建数据集（评估时不使用数据增强）
    dataset = DataGenerator(
        annotation_lines=annotation_lines,
        input_shape=config['data']['input_shape'],
        random=False,  # 评估时不使用随机增强
        augmentation_config=None
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,  # 评估时不打乱
        num_workers=config['dataloader']['num_workers']
    )
    
    return dataloader


def main():
    """主评估函数"""
    
    # ==================== 配置参数 ====================
    # 获取项目路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 配置文件路径
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    
    # 模型文件路径（使用latest目录中的最新模型）
    model_path = os.path.join(project_root, 'models', 'checkpoints', 'latest', 'best_model.pth')
    
    # 评估结果保存目录（添加时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_eval_dir = os.path.join(project_root, 'outputs', 'evaluation')
    save_dir = os.path.join(base_eval_dir, f"eval_{timestamp}")

    # 同时保存到latest目录
    latest_eval_dir = os.path.join(base_eval_dir, "latest")
    
    print("🚀 开始模型评估")
    print("="*80)
    print(f"📁 项目根目录: {project_root}")
    print(f"📄 配置文件: {config_path}")
    print(f"🤖 模型文件: {model_path}")
    print(f"💾 保存目录: {save_dir}")
    print("="*80)
    
    # ==================== 加载配置 ====================
    config = load_config(config_path)
    if config is None:
        print("❌ 无法加载配置文件，程序退出")
        return
    
    # ==================== 设备检查 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # ==================== 检查模型文件 ====================
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先完成模型训练或检查模型路径")
        return
    
    # ==================== 加载模型 ====================
    num_classes = config['data']['num_classes']
    model = load_model(model_path, num_classes, device)
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
    class_names = config['data']['class_names']
    model_name = f"VGG16-{config['data']['num_classes']}类"
    
    # 初始化报告生成器
    report_generator = ReportGenerator(class_names, model_name)
    
    # 生成综合评估报告（保存到时间戳目录）
    results = report_generator.generate_comprehensive_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        image_paths=image_paths,
        save_dir=save_dir
    )

    # 同时保存到latest目录
    print(f"\n📋 同时保存到latest目录: {latest_eval_dir}")
    report_generator.generate_comprehensive_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        image_paths=image_paths,
        save_dir=latest_eval_dir
    )
    
    # ==================== 输出关键结果 ====================
    print("\n" + "="*80)
    print("📊 评估结果摘要")
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
        print(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    
    print("="*80)
    print("🎉 评估完成！详细结果请查看保存的文件。")


if __name__ == "__main__":
    main()
