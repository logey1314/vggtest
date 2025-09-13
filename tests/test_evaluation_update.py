#!/usr/bin/env python3
"""
测试更新后的评估脚本是否正常工作
"""

import sys
import os
sys.path.append('.')

def test_config_loading():
    """测试配置文件加载功能"""
    print("🧪 测试配置文件加载...")
    
    try:
        # 导入更新后的评估脚本
        sys.path.append(os.path.join(os.getcwd(), 'scripts', 'evaluate'))
        from scripts.evaluate.evaluate_model import load_training_config
        
        # 测试项目根目录
        project_root = os.getcwd()
        
        # 测试1: 加载默认配置文件
        print("   测试默认配置文件加载...")
        config, config_source = load_training_config(None, project_root)
        
        if config:
            print(f"   ✅ 配置加载成功")
            print(f"   📋 配置来源: {config_source}")
            print(f"   🤖 模型类型: {config['model']['name']}")
            print(f"   📊 分类数量: {config['data']['num_classes']}")
        else:
            print(f"   ❌ 配置加载失败")
            print(f"   📁 项目根目录: {project_root}")
            print(f"   📄 配置文件路径: {os.path.join(project_root, 'configs', 'training_config.yaml')}")
            print(f"   📄 配置文件是否存在: {os.path.exists(os.path.join(project_root, 'configs', 'training_config.yaml'))}")
            return False
        
        # 测试2: 测试模型工厂导入
        print("   测试模型工厂导入...")
        from src.models import create_model
        
        # 创建测试模型配置
        test_model_config = config['model'].copy()
        test_model_config['num_classes'] = config['data']['num_classes']
        test_model_config['pretrained'] = False
        
        print(f"   创建测试模型: {test_model_config['name']}")
        model = create_model(test_model_config)
        
        if model:
            model_name = model.get_name() if hasattr(model, 'get_name') else test_model_config['name']
            print(f"   ✅ 模型创建成功: {model_name}")
        else:
            print(f"   ❌ 模型创建失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_imports():
    """测试评估脚本的导入"""
    print("\n🧪 测试评估脚本导入...")
    
    try:
        # 测试所有必要的导入
        sys.path.append(os.path.join(os.getcwd(), 'scripts', 'evaluate'))

        print("   测试基础导入...")
        from scripts.evaluate.evaluate_model import load_model, load_training_config, create_evaluation_dataloader
        print("   ✅ 基础函数导入成功")
        
        print("   测试工厂系统导入...")
        from src.models import create_model
        print("   ✅ 模型工厂导入成功")
        
        print("   测试评估模块导入...")
        from src.evaluation.metrics import ModelMetrics
        from src.evaluation.confusion_matrix import ConfusionMatrixAnalyzer
        print("   ✅ 评估模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🔧 评估脚本更新测试")
    print("=" * 50)
    
    tests = [
        test_evaluation_imports,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 评估脚本更新成功！")
        print("\n📋 使用方法:")
        print("1. 打开 scripts/evaluate/evaluate_model.py")
        print("2. 修改 SPECIFIC_TRAIN_PATH 为您的训练路径")
        print("3. 运行脚本进行评估")
        print("\n🎯 新功能:")
        print("- ✅ 支持多模型架构（VGG16、ResNet18、ResNet50）")
        print("- ✅ 配置驱动评估，自动读取训练配置")
        print("- ✅ 智能模型加载，根据配置选择正确架构")
        print("- ✅ 保持一键式评估的便利性")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
