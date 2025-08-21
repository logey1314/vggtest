#!/usr/bin/env python3
"""
测试优化器工厂修复是否成功
"""

import sys
import os
sys.path.append('.')

def test_optimizer_creation():
    """测试优化器创建"""
    print("🧪 测试优化器工厂修复...")
    
    try:
        from src.models import create_model
        from src.training import create_optimizer
        
        # 创建一个简单模型用于测试
        model_config = {
            'name': 'resnet50',
            'pretrained': False,
            'num_classes': 4,
            'dropout': 0.5
        }
        
        print("   创建ResNet50模型...")
        model = create_model(model_config)
        print("   ✅ 模型创建成功")
        
        # 测试Adam优化器（包含字符串类型的参数，应该被转换）
        adam_config = {
            'name': 'Adam',
            'params': {
                'lr': '0.001',        # 字符串，应该转换为float
                'weight_decay': '0.0001',  # 字符串，应该转换为float
                'eps': '1e-8',        # 字符串，应该转换为float
                'betas': ['0.9', '0.999'],  # 字符串列表，应该转换为float元组
                'amsgrad': 'false',   # 字符串，应该转换为bool
                'momentum': 0.9,      # Adam不支持，应该被过滤掉
                'nesterov': False     # Adam不支持，应该被过滤掉
            }
        }
        
        print("   创建Adam优化器（包含字符串类型参数）...")
        adam_optimizer = create_optimizer(model, adam_config)
        print(f"   ✅ Adam优化器创建成功: {type(adam_optimizer).__name__}")
        
        # 测试SGD优化器
        sgd_config = {
            'name': 'SGD',
            'params': {
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0001,
                'betas': [0.9, 0.999],  # SGD不支持，应该被过滤掉
            }
        }
        
        print("   创建SGD优化器（包含不支持的参数）...")
        sgd_optimizer = create_optimizer(model, sgd_config)
        print(f"   ✅ SGD优化器创建成功: {type(sgd_optimizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 优化器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 优化器工厂修复测试")
    print("=" * 50)
    
    if test_optimizer_creation():
        print("\n🎉 优化器工厂修复测试通过！")
        print("现在可以正常使用优化器工厂了，参数过滤功能正常工作。")
        return True
    else:
        print("\n❌ 优化器工厂修复测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
