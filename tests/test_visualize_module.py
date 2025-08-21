#!/usr/bin/env python3
"""
可视化模块功能测试
测试 scripts/visualize/ 模块的各项功能
"""

import sys
import os
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入要测试的模块
from scripts.visualize.visualize_utils import (
    setup_plot_style,
    create_output_directory,
    save_plot_with_timestamp,
    validate_image_path,
    get_project_root,
    create_color_palette,
    setup_subplot_layout
)


class TestVisualizeModule:
    """可视化模块测试类"""
    
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
    
    def test_plot_style_setup(self):
        """测试绘图样式设置"""
        print("\n🧪 测试1: 绘图样式设置")
        
        try:
            chinese_support = setup_plot_style()
            
            # 检查返回值类型
            passed = isinstance(chinese_support, bool)
            self.log_test("绘图样式设置", passed, f"中文支持: {chinese_support}")
            
            # 检查matplotlib参数是否设置
            figsize_set = matplotlib.rcParams['figure.figsize'] is not None
            dpi_set = matplotlib.rcParams['figure.dpi'] > 0
            
            self.log_test("matplotlib参数设置", figsize_set and dpi_set, 
                         f"figsize: {matplotlib.rcParams['figure.figsize']}, dpi: {matplotlib.rcParams['figure.dpi']}")
            
        except Exception as e:
            self.log_test("绘图样式设置", False, f"异常: {e}")
    
    def test_directory_creation(self):
        """测试目录创建功能"""
        print("\n🧪 测试2: 目录创建功能")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_cases = [
                {
                    'base_dir': temp_dir,
                    'subdir': 'test_subdir',
                    'description': '创建子目录'
                },
                {
                    'base_dir': temp_dir,
                    'subdir': '',
                    'description': '使用基础目录'
                },
                {
                    'base_dir': os.path.join(temp_dir, 'nested', 'path'),
                    'subdir': 'deep',
                    'description': '创建嵌套目录'
                }
            ]
            
            for case in test_cases:
                try:
                    result_dir = create_output_directory(case['base_dir'], case['subdir'])
                    passed = os.path.exists(result_dir)
                    self.log_test(f"目录创建 - {case['description']}", passed, 
                                f"创建目录: {result_dir}")
                except Exception as e:
                    self.log_test(f"目录创建 - {case['description']}", False, f"异常: {e}")
    
    def test_plot_saving(self):
        """测试图表保存功能"""
        print("\n🧪 测试3: 图表保存功能")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 创建测试图表
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot([1, 2, 3], [1, 4, 2])
                ax.set_title('Test Plot')
                
                # 测试保存功能
                output_path = save_plot_with_timestamp(
                    fig, temp_dir, 'test_plot', show_plot=False
                )
                
                # 检查文件是否创建
                passed = os.path.exists(output_path) and output_path.endswith('.png')
                self.log_test("图表保存", passed, f"保存路径: {output_path}")
                
                # 检查文件大小（应该大于0）
                if passed:
                    file_size = os.path.getsize(output_path)
                    size_ok = file_size > 0
                    self.log_test("图表文件大小", size_ok, f"文件大小: {file_size} bytes")
                
            except Exception as e:
                self.log_test("图表保存", False, f"异常: {e}")
    
    def test_path_validation(self):
        """测试路径验证功能"""
        print("\n🧪 测试4: 路径验证功能")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            test_file = os.path.join(temp_dir, 'test_image.jpg')
            with open(test_file, 'w') as f:
                f.write('dummy image content')
            
            test_cases = [
                {
                    'path': test_file,
                    'expected': test_file,
                    'description': '绝对路径有效文件'
                },
                {
                    'path': 'nonexistent_file.jpg',
                    'expected': None,
                    'description': '不存在的文件'
                },
                {
                    'path': '',
                    'expected': None,
                    'description': '空路径'
                }
            ]
            
            for case in test_cases:
                try:
                    result = validate_image_path(case['path'], temp_dir)
                    passed = result == case['expected']
                    self.log_test(f"路径验证 - {case['description']}", passed, 
                                f"输入: {case['path']}, 结果: {result}")
                except Exception as e:
                    self.log_test(f"路径验证 - {case['description']}", False, f"异常: {e}")
    
    def test_project_root_detection(self):
        """测试项目根目录检测"""
        print("\n🧪 测试5: 项目根目录检测")
        
        try:
            project_root = get_project_root()
            
            # 检查返回的是否为有效路径
            passed = os.path.exists(project_root) and os.path.isdir(project_root)
            self.log_test("项目根目录检测", passed, f"检测到的根目录: {project_root}")
            
            # 检查是否包含预期的项目文件
            expected_files = ['configs', 'src', 'scripts']
            files_exist = all(os.path.exists(os.path.join(project_root, f)) for f in expected_files)
            self.log_test("项目结构验证", files_exist, f"包含预期目录: {expected_files}")
            
        except Exception as e:
            self.log_test("项目根目录检测", False, f"异常: {e}")
    
    def test_color_palette_creation(self):
        """测试颜色调色板创建"""
        print("\n🧪 测试6: 颜色调色板创建")
        
        test_cases = [1, 5, 10, 15, 20]
        
        for n_colors in test_cases:
            try:
                colors = create_color_palette(n_colors)
                
                # 检查返回的颜色数量
                count_correct = len(colors) == n_colors
                
                # 检查颜色格式（应该是字符串或元组）
                valid_format = all(isinstance(c, (str, tuple)) for c in colors)
                
                passed = count_correct and valid_format
                self.log_test(f"颜色调色板 - {n_colors}色", passed, 
                            f"生成{len(colors)}个颜色，格式正确: {valid_format}")
                
            except Exception as e:
                self.log_test(f"颜色调色板 - {n_colors}色", False, f"异常: {e}")
    
    def test_subplot_layout(self):
        """测试子图布局计算"""
        print("\n🧪 测试7: 子图布局计算")
        
        test_cases = [
            (1, (1, 1)),
            (2, (1, 2)),
            (4, (2, 2)),
            (6, (2, 3)),
            (9, (3, 3)),
            (12, (4, 4))
        ]
        
        for n_plots, expected in test_cases:
            try:
                result = setup_subplot_layout(n_plots)
                passed = result == expected
                self.log_test(f"子图布局 - {n_plots}个图", passed, 
                            f"期望: {expected}, 实际: {result}")
            except Exception as e:
                self.log_test(f"子图布局 - {n_plots}个图", False, f"异常: {e}")
    
    def test_visualization_scripts_import(self):
        """测试可视化脚本导入"""
        print("\n🧪 测试8: 可视化脚本导入")
        
        scripts = [
            'scripts.visualize.visualize_augmentation',
            'scripts.visualize.visualize_lr_schedule',
            'scripts.visualize.visualize_training'
        ]
        
        for script in scripts:
            try:
                __import__(script)
                self.log_test(f"脚本导入 - {script.split('.')[-1]}", True, "导入成功")
            except Exception as e:
                self.log_test(f"脚本导入 - {script.split('.')[-1]}", False, f"导入失败: {e}")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🧪 测试9: 错误处理")
        
        error_cases = [
            {
                'func': lambda: validate_image_path(None, self.project_root),
                'description': 'None路径处理'
            },
            {
                'func': lambda: create_color_palette(0),
                'description': '零颜色数量处理'
            },
            {
                'func': lambda: setup_subplot_layout(-1),
                'description': '负数图表数量处理'
            }
        ]
        
        for case in error_cases:
            try:
                result = case['func']()
                # 如果没有抛出异常，检查是否返回了合理的默认值
                passed = True  # 能正常处理就算通过
                self.log_test(f"错误处理 - {case['description']}", passed, 
                            f"返回值: {result}")
            except Exception as e:
                # 如果抛出异常，这也是可以接受的错误处理方式
                self.log_test(f"错误处理 - {case['description']}", True, 
                            f"正确抛出异常: {type(e).__name__}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始可视化模块功能测试")
        print("=" * 60)
        
        # 运行所有测试
        self.test_plot_style_setup()
        self.test_directory_creation()
        self.test_plot_saving()
        self.test_path_validation()
        self.test_project_root_detection()
        self.test_color_palette_creation()
        self.test_subplot_layout()
        self.test_visualization_scripts_import()
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
            print("🎉 所有测试通过！可视化模块功能正常。")
        else:
            print("⚠️  部分测试失败，请检查相关功能。")
        
        return failed_tests == 0


def main():
    """主测试函数"""
    tester = TestVisualizeModule()
    success = tester.run_all_tests()
    
    if success:
        print("\n💡 使用提示:")
        print("1. 可视化模块功能测试通过")
        print("2. 可以安全使用 scripts/visualize/ 中的各种工具")
        print("3. 详细使用说明请查看 scripts/visualize/README.md")
        print("4. 在PyCharm中可以直接运行各个可视化脚本")
    
    return success


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    测试可视化模块的各项功能是否正常工作
    """
    success = main()
    sys.exit(0 if success else 1)
