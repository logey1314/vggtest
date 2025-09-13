#!/usr/bin/env python3
"""
测试调度器工厂修复是否成功
"""

import sys
import os
sys.path.append('..')

def test_scheduler_creation():
    """测试调度器创建"""
    print("🧪 测试调度器工厂修复...")
    
    try:
        from src.models import create_model
        from src.training import create_optimizer, create_scheduler
        
        # 创建模型和优化器用于测试
        model_config = {
            'name': 'resnet50',
            'pretrained': False,
            'num_classes': 4,
            'dropout': 0.5
        }
        
        print("   创建模型...")
        model = create_model(model_config)
        
        optimizer_config = {
            'name': 'Adam',
            'params': {
                'lr': 0.001,
                'weight_decay': 0.0001
            }
        }
        
        print("   创建优化器...")
        optimizer = create_optimizer(model, optimizer_config)
        print("   ✅ 模型和优化器创建成功")
        
        # 测试CosineAnnealingLR（包含所有调度器参数，应该过滤掉不支持的）
        cosine_config = {
            'name': 'CosineAnnealingLR',
            'params': {
                'T_max': '100',        # 字符串，应该转换为int
                'eta_min': '1e-6',     # 字符串，应该转换为float
                'step_size': 10,       # CosineAnnealingLR不支持，应该被过滤掉
                'gamma': 0.5,          # CosineAnnealingLR不支持，应该被过滤掉
                'milestones': [10, 20] # CosineAnnealingLR不支持，应该被过滤掉
            }
        }
        
        print("   创建CosineAnnealingLR调度器（包含不支持的参数）...")
        cosine_scheduler = create_scheduler(optimizer, cosine_config, total_epochs=100)
        print(f"   ✅ CosineAnnealingLR创建成功: {type(cosine_scheduler).__name__}")
        
        # 测试StepLR
        step_config = {
            'name': 'StepLR',
            'params': {
                'step_size': '10',     # 字符串，应该转换为int
                'gamma': '0.5',        # 字符串，应该转换为float
                'T_max': 100,          # StepLR不支持，应该被过滤掉
                'eta_min': 1e-6        # StepLR不支持，应该被过滤掉
            }
        }
        
        print("   创建StepLR调度器（包含不支持的参数）...")
        step_scheduler = create_scheduler(optimizer, step_config)
        print(f"   ✅ StepLR创建成功: {type(step_scheduler).__name__}")
        
        # 测试ReduceLROnPlateau
        plateau_config = {
            'name': 'ReduceLROnPlateau',
            'params': {
                'mode': 'min',
                'factor': '0.5',       # 字符串，应该转换为float
                'patience': '3',       # 字符串，应该转换为int
                'step_size': 10,       # ReduceLROnPlateau不支持，应该被过滤掉
                'T_max': 100           # ReduceLROnPlateau不支持，应该被过滤掉
            }
        }
        
        print("   创建ReduceLROnPlateau调度器（包含不支持的参数）...")
        plateau_scheduler = create_scheduler(optimizer, plateau_config)
        print(f"   ✅ ReduceLROnPlateau创建成功: {type(plateau_scheduler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 调度器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 调度器工厂修复测试")
    print("=" * 50)
    
    if test_scheduler_creation():
        print("\n🎉 调度器工厂修复测试通过！")
        print("现在可以正常使用调度器工厂了，参数过滤和类型转换功能正常工作。")
        return True
    else:
        print("\n❌ 调度器工厂修复测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
