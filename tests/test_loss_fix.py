#!/usr/bin/env python3
"""
测试损失函数工厂修复是否成功
"""

import sys
import os
sys.path.append('..')

def test_loss_creation():
    """测试损失函数创建"""
    print("🧪 测试损失函数工厂修复...")
    
    try:
        from src.training import create_loss_function
        
        # 模拟训练数据
        train_lines = [
            "0;path1.jpg",
            "0;path2.jpg", 
            "1;path3.jpg",
            "2;path4.jpg",
            "2;path5.jpg"
        ]
        
        # 测试CrossEntropyLoss（包含所有损失函数参数，应该过滤掉不支持的）
        ce_config = {
            'name': 'CrossEntropyLoss',
            'params': {
                'reduction': 'mean',
                'label_smoothing': 0.0,
                'auto_weight': True,        # CrossEntropyLoss不支持，应该被过滤掉
                'focal_alpha': 1.0,         # CrossEntropyLoss不支持，应该被过滤掉
                'focal_gamma': 2.0,         # CrossEntropyLoss不支持，应该被过滤掉
                'smoothing': 0.1            # CrossEntropyLoss不支持，应该被过滤掉
            }
        }
        
        print("   创建CrossEntropyLoss（包含不支持的参数）...")
        ce_loss = create_loss_function(ce_config)
        print(f"   ✅ CrossEntropyLoss创建成功: {type(ce_loss).__name__}")
        
        # 测试WeightedCrossEntropyLoss（包含auto_weight参数，应该被处理）
        weighted_config = {
            'name': 'WeightedCrossEntropyLoss',
            'params': {
                'auto_weight': True,        # 自定义参数，应该被处理但不传递给PyTorch
                'reduction': 'mean',
                'label_smoothing': 0.0,
                'focal_alpha': 1.0,         # WeightedCrossEntropyLoss不支持，应该被过滤掉
                'focal_gamma': 2.0          # WeightedCrossEntropyLoss不支持，应该被过滤掉
            }
        }
        
        print("   创建WeightedCrossEntropyLoss（包含自定义参数）...")
        weighted_loss = create_loss_function(weighted_config, train_lines, 3)
        print(f"   ✅ WeightedCrossEntropyLoss创建成功: {type(weighted_loss).__name__}")
        
        # 测试FocalLoss（包含不支持的参数，应该被过滤）
        focal_config = {
            'name': 'FocalLoss',
            'params': {
                'alpha': 1.0,
                'gamma': 2.0,
                'reduction': 'mean',
                'auto_weight': True,        # FocalLoss不支持，应该被过滤掉
                'label_smoothing': 0.1,     # FocalLoss不支持，应该被过滤掉
                'smoothing': 0.1            # FocalLoss不支持，应该被过滤掉
            }
        }
        
        print("   创建FocalLoss（包含不支持的参数）...")
        focal_loss = create_loss_function(focal_config)
        print(f"   ✅ FocalLoss创建成功: {type(focal_loss).__name__}")
        
        # 测试LabelSmoothingCrossEntropy
        smooth_config = {
            'name': 'LabelSmoothingCrossEntropy',
            'params': {
                'smoothing': 0.1,
                'reduction': 'mean',
                'auto_weight': True,        # LabelSmoothingCrossEntropy不支持，应该被过滤掉
                'focal_alpha': 1.0,         # LabelSmoothingCrossEntropy不支持，应该被过滤掉
                'focal_gamma': 2.0          # LabelSmoothingCrossEntropy不支持，应该被过滤掉
            }
        }
        
        print("   创建LabelSmoothingCrossEntropy（包含不支持的参数）...")
        smooth_loss = create_loss_function(smooth_config, num_classes=3)
        print(f"   ✅ LabelSmoothingCrossEntropy创建成功: {type(smooth_loss).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 损失函数创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 损失函数工厂修复测试")
    print("=" * 50)
    
    if test_loss_creation():
        print("\n🎉 损失函数工厂修复测试通过！")
        print("现在可以正常使用损失函数工厂了，参数过滤功能正常工作。")
        return True
    else:
        print("\n❌ 损失函数工厂修复测试失败！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
