#!/usr/bin/env python3
"""
工厂系统测试脚本
验证模型工厂、优化器工厂、调度器工厂、损失函数工厂是否正常工作
"""

import sys
import os
sys.path.append('..')

import torch
import yaml

def test_model_factory():
    """测试模型工厂"""
    print("🤖 测试模型工厂...")
    
    try:
        from src.models import create_model, get_available_models
        
        # 获取可用模型
        available_models = get_available_models()
        print(f"   可用模型: {list(available_models.keys())}")
        
        # 测试VGG16
        vgg_config = {
            'name': 'vgg16',
            'pretrained': False,  # 避免下载预训练权重
            'num_classes': 4,
            'dropout': 0.5
        }
        
        vgg_model = create_model(vgg_config)
        print(f"   ✅ VGG16创建成功: {vgg_model.get_name() if hasattr(vgg_model, 'get_name') else 'VGG16'}")
        
        # 测试ResNet50
        resnet_config = {
            'name': 'resnet50',
            'pretrained': False,  # 避免下载预训练权重
            'num_classes': 4,
            'dropout': 0.3,
            'resnet': {
                'replace_stride_with_dilation': [False, False, False],
                'zero_init_residual': False
            }
        }
        
        resnet_model = create_model(resnet_config)
        print(f"   ✅ ResNet50创建成功: {resnet_model.get_name() if hasattr(resnet_model, 'get_name') else 'ResNet50'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型工厂测试失败: {e}")
        return False


def test_optimizer_factory():
    """测试优化器工厂"""
    print("\n⚙️ 测试优化器工厂...")
    
    try:
        from src.training import create_optimizer, get_available_optimizers
        from src.models import create_model
        
        # 创建一个简单模型用于测试
        model_config = {'name': 'vgg16', 'pretrained': False, 'num_classes': 3}
        model = create_model(model_config)
        
        # 获取可用优化器
        available_optimizers = get_available_optimizers()
        print(f"   可用优化器: {list(available_optimizers.keys())}")
        
        # 测试Adam优化器
        adam_config = {
            'name': 'Adam',
            'params': {
                'lr': 0.001,
                'weight_decay': 0.0001
            }
        }
        
        adam_optimizer = create_optimizer(model, adam_config)
        print(f"   ✅ Adam优化器创建成功: {type(adam_optimizer).__name__}")
        
        # 测试SGD优化器
        sgd_config = {
            'name': 'SGD',
            'params': {
                'lr': 0.01,
                'momentum': 0.9
            }
        }
        
        sgd_optimizer = create_optimizer(model, sgd_config)
        print(f"   ✅ SGD优化器创建成功: {type(sgd_optimizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 优化器工厂测试失败: {e}")
        return False


def test_scheduler_factory():
    """测试调度器工厂"""
    print("\n📈 测试调度器工厂...")
    
    try:
        from src.training import create_scheduler, get_available_schedulers
        from src.training import create_optimizer
        from src.models import create_model
        
        # 创建模型和优化器用于测试
        model_config = {'name': 'vgg16', 'pretrained': False, 'num_classes': 3}
        model = create_model(model_config)
        
        optimizer_config = {'name': 'Adam', 'params': {'lr': 0.001}}
        optimizer = create_optimizer(model, optimizer_config)
        
        # 获取可用调度器
        available_schedulers = get_available_schedulers()
        print(f"   可用调度器: {list(available_schedulers.keys())}")
        
        # 测试CosineAnnealingLR
        cosine_config = {
            'name': 'CosineAnnealingLR',
            'params': {
                'T_max': 'auto',
                'eta_min': 1e-6
            }
        }
        
        cosine_scheduler = create_scheduler(optimizer, cosine_config, total_epochs=100)
        print(f"   ✅ CosineAnnealingLR创建成功: {type(cosine_scheduler).__name__}")
        
        # 测试StepLR
        step_config = {
            'name': 'StepLR',
            'params': {
                'step_size': 10,
                'gamma': 0.5
            }
        }
        
        step_scheduler = create_scheduler(optimizer, step_config)
        print(f"   ✅ StepLR创建成功: {type(step_scheduler).__name__}")
        
        # 测试None调度器
        none_config = {'name': 'None'}
        none_scheduler = create_scheduler(optimizer, none_config)
        print(f"   ✅ None调度器创建成功: {none_scheduler}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 调度器工厂测试失败: {e}")
        return False


def test_loss_factory():
    """测试损失函数工厂"""
    print("\n🎯 测试损失函数工厂...")
    
    try:
        from src.training import create_loss_function, get_available_losses
        
        # 获取可用损失函数
        available_losses = get_available_losses()
        print(f"   可用损失函数: {list(available_losses.keys())}")
        
        # 模拟训练数据
        train_lines = [
            "0;path1.jpg",
            "0;path2.jpg", 
            "1;path3.jpg",
            "2;path4.jpg",
            "2;path5.jpg"
        ]
        
        # 测试CrossEntropyLoss
        ce_config = {
            'name': 'CrossEntropyLoss'
        }
        
        ce_loss = create_loss_function(ce_config)
        print(f"   ✅ CrossEntropyLoss创建成功: {type(ce_loss).__name__}")
        
        # 测试WeightedCrossEntropyLoss
        weighted_config = {
            'name': 'WeightedCrossEntropyLoss',
            'params': {
                'auto_weight': True
            }
        }
        
        weighted_loss = create_loss_function(weighted_config, train_lines, 3)
        print(f"   ✅ WeightedCrossEntropyLoss创建成功: {type(weighted_loss).__name__}")
        
        # 测试FocalLoss
        focal_config = {
            'name': 'FocalLoss',
            'params': {
                'alpha': 1.0,
                'gamma': 2.0
            }
        }
        
        focal_loss = create_loss_function(focal_config)
        print(f"   ✅ FocalLoss创建成功: {type(focal_loss).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 损失函数工厂测试失败: {e}")
        return False


def test_custom_weights():
    """测试自定义权重路径功能"""
    print("\n📁 测试自定义权重路径...")

    try:
        from src.models import create_model

        # 测试VGG16自定义权重路径（文件不存在，应该回退到默认）
        vgg_config = {
            'name': 'vgg16',
            'pretrained': True,
            'num_classes': 3,
            'vgg': {
                'pretrained_weights': 'models/pretrained/vgg16-397923af.pth'  # 假设不存在
            }
        }

        vgg_model = create_model(vgg_config)
        print(f"   ✅ VGG16自定义权重路径测试成功（回退到默认下载）")

        # 测试ResNet50自定义权重路径（文件不存在，应该回退到默认）
        resnet_config = {
            'name': 'resnet50',
            'pretrained': True,
            'num_classes': 3,
            'resnet': {
                'pretrained_weights': 'models/pretrained/resnet50-11ad3fa6.pth'  # 假设不存在
            }
        }

        resnet_model = create_model(resnet_config)
        print(f"   ✅ ResNet50自定义权重路径测试成功（回退到默认下载）")

        return True

    except Exception as e:
        print(f"   ❌ 自定义权重路径测试失败: {e}")
        return False


def test_config_integration():
    """测试配置文件集成"""
    print("\n📄 测试配置文件集成...")

    try:
        # 加载配置文件
        config_path = '../configs/training_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"   ✅ 配置文件加载成功")

        # 测试模型配置
        from src.models import create_model
        model_config = config['model'].copy()
        model_config['num_classes'] = 3
        model_config['pretrained'] = False  # 避免下载

        model = create_model(model_config)
        print(f"   ✅ 从配置创建模型成功: {model_config['name']}")

        # 测试优化器配置
        from src.training import create_optimizer
        optimizer_config = config['training']['optimizer']
        optimizer = create_optimizer(model, optimizer_config)
        print(f"   ✅ 从配置创建优化器成功: {optimizer_config['name']}")

        # 测试调度器配置
        from src.training import create_scheduler
        scheduler_config = config['training']['scheduler']
        scheduler = create_scheduler(optimizer, scheduler_config, total_epochs=50)
        print(f"   ✅ 从配置创建调度器成功: {scheduler_config['name']}")

        # 测试损失函数配置
        from src.training import create_loss_function
        loss_config = config['training']['loss_function']
        criterion = create_loss_function(loss_config)
        print(f"   ✅ 从配置创建损失函数成功: {loss_config['name']}")

        return True

    except Exception as e:
        print(f"   ❌ 配置文件集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 开始工厂系统测试")
    print("=" * 50)
    
    tests = [
        test_model_factory,
        test_optimizer_factory,
        test_scheduler_factory,
        test_loss_factory,
        test_custom_weights,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有工厂系统测试通过！")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
