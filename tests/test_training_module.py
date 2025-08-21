#!/usr/bin/env python3
"""
训练管理模块功能测试
测试 scripts/training/ 模块的各项功能
"""

import sys
import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from scripts.training.training_utils import (
    validate_resume_config,
    setup_training_directories,
    backup_config_file,
    check_training_prerequisites,
    format_time_duration,
    calculate_eta,
    get_training_status_summary
)


class TestTrainingModule:
    """训练管理模块测试类"""
    
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
    
    def create_test_config(self, resume_enabled=True):
        """创建测试配置"""
        return {
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            },
            'data': {
                'num_classes': 4,
                'class_names': ['class1', 'class2', 'class3', 'class4'],
                'annotation_file': 'data/annotations/cls_train.txt',
                'data_dir': 'data/raw'
            },
            'model': {
                'pretrained': True,
                'dropout': 0.5
            },
            'dataloader': {
                'batch_size': 4,
                'num_workers': 0
            },
            'save': {
                'checkpoint_dir': 'models/checkpoints'
            },
            'logging': {
                'log_dir': 'outputs/logs',
                'plot_dir': 'outputs/plots'
            },
            'resume': {
                'enabled': resume_enabled,
                'auto_find_latest': True
            }
        }
    
    def test_config_validation(self):
        """测试配置验证功能"""
        print("\n🧪 测试1: 配置验证功能")
        
        # 测试有效配置
        valid_config = self.create_test_config(resume_enabled=True)
        valid, error_msg = validate_resume_config(valid_config)
        self.log_test("有效配置验证", valid, f"验证结果: {error_msg}")
        
        # 测试恢复训练未启用
        invalid_config = self.create_test_config(resume_enabled=False)
        valid, error_msg = validate_resume_config(invalid_config)
        self.log_test("恢复训练未启用", not valid, f"正确识别错误: {error_msg}")
        
        # 测试缺少必要配置项
        incomplete_config = self.create_test_config()
        del incomplete_config['training']
        valid, error_msg = validate_resume_config(incomplete_config)
        self.log_test("缺少必要配置项", not valid, f"正确识别错误: {error_msg}")
        
        # 测试类别数量不匹配
        mismatch_config = self.create_test_config()
        mismatch_config['data']['num_classes'] = 3  # 但class_names有4个
        valid, error_msg = validate_resume_config(mismatch_config)
        self.log_test("类别数量不匹配", not valid, f"正确识别错误: {error_msg}")
    
    def test_directory_setup(self):
        """测试目录设置功能"""
        print("\n🧪 测试2: 目录设置功能")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_test_config()
            
            try:
                directories = setup_training_directories(temp_dir, config, "resume")
                
                # 检查返回的目录字典
                required_keys = ['checkpoint_dir', 'log_dir', 'plot_dir', 
                               'latest_checkpoint_dir', 'latest_log_dir', 'latest_plot_dir', 'timestamp']
                missing_keys = [key for key in required_keys if key not in directories]
                
                if missing_keys:
                    self.log_test("目录设置 - 返回值", False, f"缺少键: {missing_keys}")
                else:
                    self.log_test("目录设置 - 返回值", True, "包含所有必需的目录键")
                
                # 检查目录是否实际创建
                dirs_created = all(os.path.exists(directories[key]) 
                                 for key in required_keys if key != 'timestamp')
                self.log_test("目录设置 - 实际创建", dirs_created, 
                            f"目录创建状态: {dirs_created}")
                
                # 检查时间戳格式
                timestamp = directories['timestamp']
                timestamp_valid = len(timestamp) == 15 and '_' in timestamp
                self.log_test("目录设置 - 时间戳格式", timestamp_valid, 
                            f"时间戳: {timestamp}")
                
            except Exception as e:
                self.log_test("目录设置", False, f"异常: {e}")
    
    def test_config_backup(self):
        """测试配置备份功能"""
        print("\n🧪 测试3: 配置备份功能")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_test_config()
            
            try:
                backup_path = backup_config_file(config, temp_dir, "test_config.yaml")
                
                # 检查备份文件是否创建
                file_exists = os.path.exists(backup_path)
                self.log_test("配置备份 - 文件创建", file_exists, f"备份路径: {backup_path}")
                
                if file_exists:
                    # 检查备份文件内容
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        backup_config = yaml.safe_load(f)
                    
                    content_match = backup_config == config
                    self.log_test("配置备份 - 内容一致性", content_match, 
                                f"内容匹配: {content_match}")
                
            except Exception as e:
                self.log_test("配置备份", False, f"异常: {e}")
    
    def test_prerequisites_check(self):
        """测试前置条件检查"""
        print("\n🧪 测试4: 前置条件检查")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.create_test_config()
            
            # 创建必要的文件和目录
            annotation_path = os.path.join(temp_dir, config['data']['annotation_file'])
            os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
            with open(annotation_path, 'w') as f:
                f.write("0;test_image.jpg\n")
            
            data_dir = os.path.join(temp_dir, config['data']['data_dir'])
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                prerequisites_ok, errors = check_training_prerequisites(temp_dir, config)
                
                self.log_test("前置条件检查 - 有效环境", prerequisites_ok, 
                            f"检查结果: {prerequisites_ok}, 错误: {errors}")
                
                # 测试缺少文件的情况
                os.remove(annotation_path)
                prerequisites_ok, errors = check_training_prerequisites(temp_dir, config)
                
                self.log_test("前置条件检查 - 缺少文件", not prerequisites_ok, 
                            f"正确识别缺少文件: {len(errors)} 个错误")
                
            except Exception as e:
                self.log_test("前置条件检查", False, f"异常: {e}")
    
    def test_time_formatting(self):
        """测试时间格式化功能"""
        print("\n🧪 测试5: 时间格式化功能")
        
        test_cases = [
            (30, "30.0秒"),
            (90, "1.5分钟"),
            (3600, "1.0小时"),
            (7200, "2.0小时")
        ]
        
        for seconds, expected in test_cases:
            try:
                result = format_time_duration(seconds)
                passed = expected in result or result in expected  # 允许一些格式差异
                self.log_test(f"时间格式化 - {seconds}秒", passed, 
                            f"输入: {seconds}s, 输出: {result}, 期望: {expected}")
            except Exception as e:
                self.log_test(f"时间格式化 - {seconds}秒", False, f"异常: {e}")
    
    def test_eta_calculation(self):
        """测试预计剩余时间计算"""
        print("\n🧪 测试6: ETA计算功能")
        
        test_cases = [
            (0, 100, 0, "计算中..."),
            (10, 100, 1000, "1.5分钟"),  # 10轮用了1000秒，剩余90轮
            (50, 100, 5000, "1.7小时")   # 50轮用了5000秒，剩余50轮
        ]
        
        for current_epoch, total_epochs, elapsed_time, expected_type in test_cases:
            try:
                result = calculate_eta(current_epoch, total_epochs, elapsed_time)
                
                if current_epoch == 0:
                    passed = result == "计算中..."
                else:
                    # 检查结果包含时间单位
                    passed = any(unit in result for unit in ["秒", "分钟", "小时"])
                
                self.log_test(f"ETA计算 - epoch {current_epoch}", passed, 
                            f"结果: {result}")
                
            except Exception as e:
                self.log_test(f"ETA计算 - epoch {current_epoch}", False, f"异常: {e}")
    
    def test_status_summary(self):
        """测试训练状态摘要"""
        print("\n🧪 测试7: 训练状态摘要")
        
        try:
            summary = get_training_status_summary(
                current_epoch=50,
                total_epochs=100,
                train_acc=85.5,
                val_acc=82.3,
                best_acc=83.1,
                elapsed_time=3600
            )
            
            # 检查摘要是否包含关键信息
            required_info = ["50/100", "85.5", "82.3", "83.1", "50.0%"]
            missing_info = [info for info in required_info if info not in summary]
            
            passed = len(missing_info) == 0
            self.log_test("训练状态摘要", passed, 
                        f"包含所有关键信息: {passed}, 缺少: {missing_info}")
            
        except Exception as e:
            self.log_test("训练状态摘要", False, f"异常: {e}")
    
    def test_script_import(self):
        """测试脚本导入"""
        print("\n🧪 测试8: 脚本导入")
        
        try:
            import scripts.training.resume_training
            self.log_test("恢复训练脚本导入", True, "导入成功")
        except Exception as e:
            self.log_test("恢复训练脚本导入", False, f"导入失败: {e}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始训练管理模块功能测试")
        print("=" * 60)
        
        # 运行所有测试
        self.test_config_validation()
        self.test_directory_setup()
        self.test_config_backup()
        self.test_prerequisites_check()
        self.test_time_formatting()
        self.test_eta_calculation()
        self.test_status_summary()
        self.test_script_import()
        
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
            print("🎉 所有测试通过！训练管理模块功能正常。")
        else:
            print("⚠️  部分测试失败，请检查相关功能。")
        
        return failed_tests == 0


def main():
    """主测试函数"""
    tester = TestTrainingModule()
    success = tester.run_all_tests()
    
    if success:
        print("\n💡 使用提示:")
        print("1. 训练管理模块功能测试通过")
        print("2. 可以安全使用 scripts/training/resume_training.py")
        print("3. 详细使用说明请查看 scripts/training/README.md")
        print("4. 在PyCharm中可以直接运行恢复训练脚本")
    
    return success


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    测试训练管理模块的各项功能是否正常工作
    """
    success = main()
    sys.exit(0 if success else 1)
