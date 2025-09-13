"""
特征可视化功能测试
验证特征提取、Grad-CAM和可视化功能的正确性
"""

import os
import sys
import unittest
import torch
import numpy as np
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.evaluate.feature_visualization import FeatureExtractor, GradCAM, FeatureVisualizer
from src.models.vgg import vgg16


class TestFeatureVisualization(unittest.TestCase):
    """特征可视化功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试模型
        self.model = vgg16(pretrained=False, num_classes=4)
        self.model.eval()
        
        # 测试设备
        self.device = 'cpu'
        
        # 测试数据
        self.test_image = torch.randn(1, 3, 224, 224)
        self.class_names = ['class1_125-175', 'class2_180-230', 'class3_233-285', 'class4_250-285']
        
        # 创建测试输出目录
        self.test_output_dir = os.path.join(project_root, 'tests', 'test_outputs')
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def test_feature_extractor(self):
        """测试特征提取器"""
        print("🧪 测试特征提取器...")
        
        # 初始化特征提取器
        extractor = FeatureExtractor(self.model, self.device)
        
        # 测试特征提取
        layer_names = ['conv1_1', 'conv2_1', 'conv3_1']
        features = extractor.extract_features(self.test_image, layer_names)
        
        # 验证结果
        self.assertEqual(len(features), len(layer_names))
        
        for layer_name in layer_names:
            self.assertIn(layer_name, features)
            self.assertIsInstance(features[layer_name], torch.Tensor)
            self.assertEqual(len(features[layer_name].shape), 4)  # [B, C, H, W]
        
        # 清理钩子
        extractor.remove_hooks()
        
        print("✅ 特征提取器测试通过")
    
    def test_grad_cam(self):
        """测试Grad-CAM功能"""
        print("🧪 测试Grad-CAM...")
        
        # 初始化Grad-CAM
        grad_cam = GradCAM(self.model, 'conv5_3', self.device)
        
        # 生成CAM
        cam = grad_cam.generate_cam(self.test_image, class_idx=0)
        
        # 验证结果
        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(len(cam.shape), 2)  # 2D热力图
        self.assertTrue(np.all(cam >= 0))    # 非负值
        self.assertTrue(np.all(cam <= 1))    # 归一化到0-1
        
        print("✅ Grad-CAM测试通过")
    
    def test_feature_visualizer_init(self):
        """测试特征可视化器初始化"""
        print("🧪 测试特征可视化器初始化...")
        
        # 初始化可视化器
        visualizer = FeatureVisualizer(self.model, self.class_names, self.device)
        
        # 验证初始化
        self.assertEqual(visualizer.model, self.model)
        self.assertEqual(visualizer.class_names, self.class_names)
        self.assertEqual(visualizer.device, self.device)
        self.assertIsInstance(visualizer.feature_extractor, FeatureExtractor)
        self.assertIsInstance(visualizer.grad_cam, GradCAM)
        
        print("✅ 特征可视化器初始化测试通过")
    
    def test_activation_patterns_analysis(self):
        """测试激活模式分析"""
        print("🧪 测试激活模式分析...")
        
        # 初始化可视化器
        visualizer = FeatureVisualizer(self.model, self.class_names, self.device)
        
        # 准备测试数据
        image_tensors = [torch.randn(3, 224, 224) for _ in range(4)]
        labels = [0, 1, 2, 3]
        
        # 测试激活模式分析
        try:
            save_path = visualizer.analyze_activation_patterns(
                image_tensors, labels, self.test_output_dir
            )
            
            # 验证结果
            self.assertTrue(os.path.exists(save_path))
            self.assertTrue(save_path.endswith('.png'))
            
            print(f"   激活模式分析图保存在: {save_path}")
            
        except Exception as e:
            self.fail(f"激活模式分析失败: {e}")
        
        print("✅ 激活模式分析测试通过")
    
    def test_error_feature_comparison(self):
        """测试错误样本特征对比"""
        print("🧪 测试错误样本特征对比...")
        
        # 初始化可视化器
        visualizer = FeatureVisualizer(self.model, self.class_names, self.device)
        
        # 准备测试数据
        correct_samples = [
            (torch.randn(3, 224, 224), "test1.jpg", 0),
            (torch.randn(3, 224, 224), "test2.jpg", 1)
        ]
        
        error_samples = [
            (torch.randn(3, 224, 224), "test3.jpg", 0, 1),  # true=0, pred=1
            (torch.randn(3, 224, 224), "test4.jpg", 1, 2)   # true=1, pred=2
        ]
        
        # 测试特征对比
        try:
            save_path = visualizer.compare_error_features(
                correct_samples, error_samples, self.test_output_dir
            )
            
            # 验证结果
            self.assertTrue(os.path.exists(save_path))
            self.assertTrue(save_path.endswith('.png'))
            
            print(f"   错误特征对比图保存在: {save_path}")
            
        except Exception as e:
            self.fail(f"错误特征对比失败: {e}")
        
        print("✅ 错误样本特征对比测试通过")
    
    def tearDown(self):
        """测试后清理"""
        # 清理测试输出文件（可选）
        pass


def run_tests():
    """运行所有测试"""
    print("🚀 开始特征可视化功能测试")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureVisualization)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ 所有测试通过！特征可视化功能正常工作")
    else:
        print("❌ 部分测试失败，请检查代码")
        for failure in result.failures:
            print(f"失败: {failure[0]}")
            print(f"原因: {failure[1]}")
        for error in result.errors:
            print(f"错误: {error[0]}")
            print(f"原因: {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
