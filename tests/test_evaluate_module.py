#!/usr/bin/env python3
"""
评估模块功能测试
测试 scripts/evaluate/ 模块的各项功能
"""

import sys
import os
import tempfile
import shutil
import torch
import yaml
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from scripts.evaluate.evaluate_utils import (
    extract_timestamp_from_path,
    validate_train_path,
    find_model_file,
    get_corresponding_model_path,
    generate_eval_timestamp,
    print_path_info
)


class TestEvaluateModule:
    """评估模块测试类"""
    
    def __init__(self):
        self.test_results = []
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def log_test(self, test_name, passed, message=""):
        """记录测试结果"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
        print(f"{status} {test_name}")
        if message:
            print(f"    {message}")
    
    def test_timestamp_extraction(self):
        """测试时间戳提取功能"""
        print("\n🧪 测试1: 时间戳提取功能")
        
        test_cases = [
            {
                'input': "/var/yjs/zes/vgg/outputs/logs/train_20250820_074155/",
                'expected': "20250820_074155",
                'description': "完整路径"
            },
            {
                'input': "train_20250820_074155",
                'expected': "20250820_074155", 
                'description': "目录名"
            },
            {
                'input': "resume_20250820_074155",
                'expected': "20250820_074155",
                'description': "恢复训练目录"
            },
            {
                'input': "invalid_path",
                'expected': None,
                'description': "无效路径"
            }
        ]
        
        for case in test_cases:
            try:
                result = extract_timestamp_from_path(case['input'])
                passed = result == case['expected']
                message = f"{case['description']}: '{case['input']}' -> '{result}'"
                self.log_test(f"时间戳提取 - {case['description']}", passed, message)
            except Exception as e:
                self.log_test(f"时间戳提取 - {case['description']}", False, f"异常: {e}")
    
    def test_eval_timestamp_generation(self):
        """测试评估时间戳生成"""
        print("\n🧪 测试2: 评估时间戳生成")
        
        test_cases = [
            {
                'input': "20250820_074155",
                'expected': "eval_20250820_074155"
            },
            {
                'input': "20241225_120000",
                'expected': "eval_20241225_120000"
            }
        ]
        
        for case in test_cases:
            try:
                result = generate_eval_timestamp(case['input'])
                passed = result == case['expected']
                message = f"'{case['input']}' -> '{result}'"
                self.log_test("评估时间戳生成", passed, message)
            except Exception as e:
                self.log_test("评估时间戳生成", False, f"异常: {e}")
    
    def test_path_validation(self):
        """测试路径验证功能"""
        print("\n🧪 测试3: 路径验证功能")
        
        # 创建临时测试目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建有效的测试路径
            valid_path = os.path.join(temp_dir, "train_20250820_074155")
            os.makedirs(valid_path, exist_ok=True)
            
            test_cases = [
                {
                    'input': valid_path,
                    'expected': True,
                    'description': "有效路径"
                },
                {
                    'input': "/nonexistent/path/train_20250820_074155",
                    'expected': False,
                    'description': "不存在的路径"
                },
                {
                    'input': temp_dir,  # 存在但无时间戳
                    'expected': False,
                    'description': "无时间戳的路径"
                },
                {
                    'input': "",
                    'expected': False,
                    'description': "空路径"
                }
            ]
            
            for case in test_cases:
                try:
                    result = validate_train_path(case['input'])
                    passed = result == case['expected']
                    message = f"{case['description']}: {passed}"
                    self.log_test(f"路径验证 - {case['description']}", passed, message)
                except Exception as e:
                    self.log_test(f"路径验证 - {case['description']}", False, f"异常: {e}")
    
    def test_model_file_finding(self):
        """测试模型文件查找功能"""
        print("\n🧪 测试4: 模型文件查找功能")
        
        # 创建临时测试环境
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = "20250820_074155"
            
            # 创建模型目录结构
            checkpoint_dir = os.path.join(temp_dir, 'models', 'checkpoints', f'train_{timestamp}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 创建测试模型文件
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_50.pth')
            
            # 创建空文件
            with open(best_model_path, 'w') as f:
                f.write("dummy model")
            with open(checkpoint_path, 'w') as f:
                f.write("dummy checkpoint")
            
            # 测试查找功能
            try:
                result = find_model_file(temp_dir, timestamp)
                passed = result == best_model_path
                message = f"找到模型文件: {result}"
                self.log_test("模型文件查找 - 最佳模型", passed, message)
            except Exception as e:
                self.log_test("模型文件查找 - 最佳模型", False, f"异常: {e}")
            
            # 测试不存在的时间戳
            try:
                result = find_model_file(temp_dir, "20990101_000000")
                passed = result is None
                self.log_test("模型文件查找 - 不存在的时间戳", passed, "正确返回None")
            except Exception as e:
                self.log_test("模型文件查找 - 不存在的时间戳", False, f"异常: {e}")
    
    def test_config_loading(self):
        """测试配置文件加载"""
        print("\n🧪 测试5: 配置文件加载")
        
        config_path = os.path.join(self.project_root, 'configs', 'training_config.yaml')
        
        try:
            # 导入配置加载函数
            from scripts.evaluate.evaluate_model import load_config
            
            config = load_config(config_path)
            
            if config is None:
                self.log_test("配置文件加载", False, "配置文件加载失败")
            else:
                # 检查关键配置项
                required_keys = ['data', 'model', 'training']
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    self.log_test("配置文件加载", False, f"缺少关键配置项: {missing_keys}")
                else:
                    self.log_test("配置文件加载", True, "配置文件加载成功，包含所有必需项")
                    
                    # 检查数据配置
                    if 'num_classes' in config['data'] and 'class_names' in config['data']:
                        num_classes = config['data']['num_classes']
                        class_names = config['data']['class_names']
                        classes_match = len(class_names) == num_classes
                        self.log_test("配置一致性检查", classes_match, 
                                    f"类别数量: {num_classes}, 类别名称数量: {len(class_names)}")
        
        except Exception as e:
            self.log_test("配置文件加载", False, f"异常: {e}")
    
    def test_integration(self):
        """集成测试"""
        print("\n🧪 测试6: 集成测试")
        
        # 创建完整的测试环境
        with tempfile.TemporaryDirectory() as temp_dir:
            timestamp = "20250820_074155"
            train_path = os.path.join(temp_dir, "outputs", "logs", f"train_{timestamp}")
            os.makedirs(train_path, exist_ok=True)
            
            # 创建模型文件
            model_dir = os.path.join(temp_dir, 'models', 'checkpoints', f'train_{timestamp}')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_model.pth')
            with open(model_path, 'w') as f:
                f.write("dummy model")
            
            try:
                # 测试完整流程
                result_model_path = get_corresponding_model_path(train_path, temp_dir)
                
                if result_model_path == model_path:
                    self.log_test("集成测试 - 完整流程", True, "成功找到对应的模型文件")
                else:
                    self.log_test("集成测试 - 完整流程", False, 
                                f"期望: {model_path}, 实际: {result_model_path}")
            
            except Exception as e:
                self.log_test("集成测试 - 完整流程", False, f"异常: {e}")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🧪 测试7: 错误处理")
        
        error_cases = [
            {
                'func': lambda: extract_timestamp_from_path(None),
                'description': "None输入处理"
            },
            {
                'func': lambda: validate_train_path(None),
                'description': "None路径验证"
            },
            {
                'func': lambda: find_model_file("", ""),
                'description': "空参数处理"
            }
        ]
        
        for case in error_cases:
            try:
                result = case['func']()
                # 如果没有抛出异常，检查是否返回了合理的默认值
                passed = result is None or result is False
                self.log_test(f"错误处理 - {case['description']}", passed, 
                            f"返回值: {result}")
            except Exception as e:
                # 如果抛出异常，这也是可以接受的错误处理方式
                self.log_test(f"错误处理 - {case['description']}", True, 
                            f"正确抛出异常: {type(e).__name__}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始评估模块功能测试")
        print("=" * 60)
        
        # 运行所有测试
        self.test_timestamp_extraction()
        self.test_eval_timestamp_generation()
        self.test_path_validation()
        self.test_model_file_finding()
        self.test_config_loading()
        self.test_integration()
        self.test_error_handling()
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 60)
        print("📊 测试结果摘要")
        print("=" * 60)
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"通过率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['name']}: {result['message']}")
        
        print("=" * 60)
        
        if failed_tests == 0:
            print("🎉 所有测试通过！评估模块功能正常。")
        else:
            print("⚠️  部分测试失败，请检查相关功能。")
        
        return failed_tests == 0


def main():
    """主测试函数"""
    tester = TestEvaluateModule()
    success = tester.run_all_tests()
    
    if success:
        print("\n💡 使用提示:")
        print("1. 评估模块功能测试通过")
        print("2. 可以安全使用 scripts/evaluate/evaluate_model.py")
        print("3. 修改 SPECIFIC_TRAIN_PATH 变量来指定要评估的训练轮次")
        print("4. 详细使用说明请查看 scripts/evaluate/README.md")
    
    return success


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    测试评估模块的各项功能是否正常工作
    """
    success = main()
    sys.exit(0 if success else 1)
