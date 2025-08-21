#!/usr/bin/env python3
"""
测试ResNet修复是否成功
"""

import sys
import os
sys.path.append('.')

def test_resnet_creation():
    """测试ResNet模型创建"""
    print("🧪 测试ResNet模型创建...")
    
    try:
        from src.models import create_model
        
        # 测试ResNet50创建（不使用预训练权重）
        resnet50_config = {
            'name': 'resnet50',
            'pretrained': False,  # 避免下载
            'num_classes': 4,
            'dropout': 0.5
        }
        
        print("   创建ResNet50模型...")
        resnet50_model = create_model(resnet50_config)
        print(f"   ✅ ResNet50创建成功: {resnet50_model.get_name() if hasattr(resnet50_model, 'get_name') else 'ResNet50'}")
        
        # 测试ResNet18创建（不使用预训练权重）
        resnet18_config = {
            'name': 'resnet18',
            'pretrained': False,  # 避免下载
            'num_classes': 4,
            'dropout': 0.5
        }
        
        print("   创建ResNet18模型...")
        resnet18_model = create_model(resnet18_config)
        print(f"   ✅ ResNet18创建成功: {resnet18_model.get_name() if hasattr(resnet18_model, 'get_name') else 'ResNet18'}")
        
        # 测试模型前向传播
        import torch
        x = torch.randn(1, 3, 224, 224)
        
        print("   测试ResNet50前向传播...")
        output50 = resnet50_model(x)
        print(f"   ✅ ResNet50输出形状: {output50.shape}")
        
        print("   测试ResNet18前向传播...")
        output18 = resnet18_model(x)
        print(f"   ✅ ResNet18输出形状: {output18.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ResNet模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 ResNet修复测试")
    print("=" * 50)
    
    if test_resnet_creation():
        print("\n🎉 ResNet修复测试通过！")
        print("现在可以正常使用ResNet模型进行训练了。")
        return True
    else:
        print("\n❌ ResNet修复测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
