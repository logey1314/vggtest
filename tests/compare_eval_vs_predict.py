#!/usr/bin/env python3
"""
对比评估函数和预测函数在类别9上的表现
使用相同的数据和模型，找出差异所在
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_class9_images():
    """加载所有类别9的图片路径"""
    print("📊 加载类别9图片路径...")
    
    # 查找测试集文件
    possible_paths = [
        "data/annotations/cls_test.txt",
        "/var/yjs/zes/vgg/data/annotations/cls_test.txt",
        "../data/annotations/cls_test.txt",
        "../../data/annotations/cls_test.txt"
    ]
    
    test_file = None
    for path in possible_paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        print("❌ 测试集文件不存在")
        return None
    
    print(f"✅ 找到测试集文件: {test_file}")
    
    # 读取测试集数据
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    # 筛选类别9的图片
    class9_paths = []
    class9_annotations = []
    
    for line in lines:
        parts = line.split(';')
        if len(parts) >= 2:
            class_id = int(parts[0])
            image_path = parts[1]
            
            if class_id == 8:  # 类别9对应索引8
                class9_paths.append(image_path)
                class9_annotations.append(line)
    
    print(f"找到 {len(class9_paths)} 个类别9图片")
    print("示例路径:")
    for i, path in enumerate(class9_paths[:3]):
        print(f"   {i+1}: {path}")
    
    return class9_paths, class9_annotations

def load_config():
    """加载配置文件"""
    config_paths = [
        "configs/training_config.yaml",
        "/var/yjs/zes/vgg/configs/training_config.yaml",
        "../configs/training_config.yaml"
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 加载配置文件: {path}")
            return config
    
    print("❌ 无法加载配置文件")
    return None

def load_model():
    """加载训练好的模型"""
    print("🤖 加载模型...")
    
    config = load_config()
    if config is None:
        return None, None
    
    # 导入模型工厂
    from src.models import create_model
    
    # 查找模型文件
    model_paths = [
        "models/checkpoints/latest/best_model.pth",
        "/var/yjs/zes/vgg/models/checkpoints/latest/best_model.pth",
        "../models/checkpoints/latest/best_model.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 无法找到模型文件")
        return None, None
    
    print(f"✅ 找到模型文件: {model_path}")
    
    # 创建模型
    model_config = config['model'].copy()
    model_config['num_classes'] = config['data']['num_classes']
    model_config['pretrained'] = False
    
    model = create_model(model_config)
    print(f"✅ 模型创建成功: {model_config['name']}")
    
    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("✅ 模型权重加载成功")
    
    return model, config

def test_with_evaluation_function():
    """使用评估函数测试类别9"""
    print("\n🧪 使用评估函数测试类别9...")
    
    try:
        # 导入评估相关模块
        from src.data.dataset import DataGenerator
        from torch.utils.data import DataLoader
        
        # 加载数据和模型
        class9_paths, class9_annotations = load_class9_images()
        model, config = load_model()
        
        if class9_paths is None or model is None:
            return None
        
        # 创建数据生成器
        input_shape = config['data']['input_shape']
        dataset = DataGenerator(class9_annotations, input_shape, random=False)
        
        # 创建数据加载器
        batch_size = config['dataloader']['batch_size']
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        print(f"评估函数数据加载器:")
        print(f"   样本数: {len(dataset)}")
        print(f"   批次大小: {batch_size}")
        
        # 运行评估
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        y_true = []
        y_pred = []
        y_prob = []
        image_paths = []
        
        print("开始评估...")
        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(dataloader):
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
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # 计算准确率
        accuracy = np.mean(y_true == y_pred)
        correct_count = np.sum(y_true == y_pred)
        error_count = len(y_true) - correct_count
        
        print(f"✅ 评估函数结果:")
        print(f"   总样本数: {len(y_true)}")
        print(f"   正确预测: {correct_count}")
        print(f"   错误预测: {error_count}")
        print(f"   准确率: {accuracy:.4f}")
        
        # 分析预测分布
        pred_counter = Counter(y_pred)
        print(f"   预测分布: {dict(pred_counter)}")
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'image_paths': image_paths,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'error_count': error_count
        }
        
    except Exception as e:
        print(f"❌ 评估函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_input(x):
    """
    训练时使用的预处理函数
    将像素值从 [0, 255] 范围转换到 [-1, 1] 范围
    """
    x /= 127.5
    x -= 1.
    return x

def get_transform(config):
    """获取与训练时一致的数据预处理"""
    input_shape = config['data']['input_shape']
    
    # 根据训练时的预处理创建transform
    # 注意：不使用ImageNet归一化，而是使用自定义的preprocess_input
    transform = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor(),  # 转换到[0,1]范围
        transforms.Lambda(lambda x: preprocess_input(x * 255.0))  # 转换到[-1,1]范围
    ])
    
    return transform

def test_with_prediction_function():
    """使用预测函数测试类别9"""
    print("\n🔍 使用预测函数测试类别9...")
    
    try:
        # 加载数据和模型
        class9_paths, class9_annotations = load_class9_images()
        model, config = load_model()
        
        if class9_paths is None or model is None:
            return None
        
        # 获取数据预处理
        transform = get_transform(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        y_true = []
        y_pred = []
        y_prob = []
        successful_paths = []
        
        print(f"预测函数处理 {len(class9_paths)} 个图片...")
        
        for i, image_path in enumerate(class9_paths):
            try:
                # 检查图片文件是否存在
                if not os.path.exists(image_path):
                    print(f"⚠️  图片不存在: {image_path}")
                    continue
                
                # 加载和预处理图片
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # 预测
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                
                # 收集结果
                y_true.append(8)  # 真实标签是类别9（索引8）
                y_pred.append(predicted.cpu().item())
                y_prob.append(probabilities.cpu().numpy()[0])
                successful_paths.append(image_path)
                
                if (i + 1) % 20 == 0:
                    print(f"   已处理: {i + 1}/{len(class9_paths)}")
                
            except Exception as e:
                print(f"⚠️  处理图片失败 {image_path}: {e}")
                continue
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # 计算准确率
        accuracy = np.mean(y_true == y_pred)
        correct_count = np.sum(y_true == y_pred)
        error_count = len(y_true) - correct_count
        
        print(f"✅ 预测函数结果:")
        print(f"   成功处理: {len(successful_paths)}")
        print(f"   正确预测: {correct_count}")
        print(f"   错误预测: {error_count}")
        print(f"   准确率: {accuracy:.4f}")
        
        # 分析预测分布
        pred_counter = Counter(y_pred)
        print(f"   预测分布: {dict(pred_counter)}")
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'image_paths': successful_paths,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'error_count': error_count
        }
        
    except Exception as e:
        print(f"❌ 预测函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(eval_result, pred_result):
    """对比两种方法的结果"""
    print("\n📊 对比结果...")
    
    if eval_result is None or pred_result is None:
        print("❌ 无法对比，某个测试失败")
        return
    
    print("="*60)
    print(f"📋 结果对比:")
    print(f"   评估函数 - 样本数: {len(eval_result['y_true'])}, 准确率: {eval_result['accuracy']:.4f}")
    print(f"   预测函数 - 样本数: {len(pred_result['y_true'])}, 准确率: {pred_result['accuracy']:.4f}")
    print()
    print(f"   准确率差异: {abs(eval_result['accuracy'] - pred_result['accuracy']):.4f}")
    
    if abs(eval_result['accuracy'] - pred_result['accuracy']) > 0.1:
        print("❌ 发现显著差异！")
        print("   可能的原因:")
        print("   1. 数据预处理不一致")
        print("   2. 数据加载方式不同")
        print("   3. 模型推理环境不同")
    else:
        print("✅ 结果基本一致")
    
    # 详细对比预测分布
    print(f"\n预测分布对比:")
    eval_counter = Counter(eval_result['y_pred'])
    pred_counter = Counter(pred_result['y_pred'])
    
    all_classes = set(eval_counter.keys()) | set(pred_counter.keys())
    for class_id in sorted(all_classes):
        eval_count = eval_counter.get(class_id, 0)
        pred_count = pred_counter.get(class_id, 0)
        print(f"   类别{class_id}: 评估函数={eval_count}, 预测函数={pred_count}, 差异={abs(eval_count-pred_count)}")

def main():
    """主函数"""
    print("🚀 对比评估函数和预测函数在类别9上的表现...")
    print("="*60)
    
    try:
        # 测试评估函数
        eval_result = test_with_evaluation_function()
        
        # 测试预测函数
        pred_result = test_with_prediction_function()
        
        # 对比结果
        compare_results(eval_result, pred_result)
        
        print("="*60)
        print("🎯 测试完成")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
