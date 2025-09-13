"""
图像噪声增强功能测试
验证各种噪声类型的正确性和参数有效性
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.dataset import DataGenerator


class TestNoiseAugmentation(unittest.TestCase):
    """噪声增强功能测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像数据
        self.test_image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.float32)
        
        # 创建测试数据生成器
        self.dummy_lines = ["0;test_image.jpg"]
        self.input_shape = [224, 224]
        
        # 基础配置
        self.base_config = {
            'noise': {
                'enable_noise': True,
                'noise_probability': 1.0,  # 确保应用噪声
                'gaussian_noise': {
                    'enable': False,
                    'mean': 0.0,
                    'std': 0.05,
                    'probability': 1.0
                },
                'salt_pepper_noise': {
                    'enable': False,
                    'salt_prob': 0.01,
                    'pepper_prob': 0.01,
                    'probability': 1.0
                },
                'uniform_noise': {
                    'enable': False,
                    'low': -0.05,
                    'high': 0.05,
                    'probability': 1.0
                },
                'poisson_noise': {
                    'enable': False,
                    'scale': 1.0,
                    'probability': 1.0
                },
                'blur_noise': {
                    'enable': False,
                    'kernel_size': 3,
                    'sigma': 0.5,
                    'probability': 1.0
                }
            }
        }
    
    def test_gaussian_noise(self):
        """测试高斯噪声"""
        print("🧪 测试高斯噪声...")
        
        # 启用高斯噪声
        config = self.base_config.copy()
        config['noise']['gaussian_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声添加
        noisy_image = data_generator.add_gaussian_noise(self.test_image_data, mean=0.0, std=0.05)
        
        # 验证结果
        self.assertEqual(noisy_image.shape, self.test_image_data.shape)
        self.assertTrue(np.all(noisy_image >= 0))
        self.assertTrue(np.all(noisy_image <= 255))
        self.assertFalse(np.array_equal(noisy_image, self.test_image_data))
        
        print("✅ 高斯噪声测试通过")
    
    def test_salt_pepper_noise(self):
        """测试椒盐噪声"""
        print("🧪 测试椒盐噪声...")
        
        # 启用椒盐噪声
        config = self.base_config.copy()
        config['noise']['salt_pepper_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声添加
        noisy_image = data_generator.add_salt_pepper_noise(self.test_image_data, 
                                                          salt_prob=0.02, pepper_prob=0.02)
        
        # 验证结果
        self.assertEqual(noisy_image.shape, self.test_image_data.shape)
        self.assertTrue(np.all(noisy_image >= 0))
        self.assertTrue(np.all(noisy_image <= 255))
        
        # 检查是否有盐噪声（白点）和椒噪声（黑点）
        has_salt = np.any(noisy_image == 255)
        has_pepper = np.any(noisy_image == 0)
        self.assertTrue(has_salt or has_pepper)
        
        print("✅ 椒盐噪声测试通过")
    
    def test_uniform_noise(self):
        """测试均匀噪声"""
        print("🧪 测试均匀噪声...")
        
        # 启用均匀噪声
        config = self.base_config.copy()
        config['noise']['uniform_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声添加
        noisy_image = data_generator.add_uniform_noise(self.test_image_data, low=-0.05, high=0.05)
        
        # 验证结果
        self.assertEqual(noisy_image.shape, self.test_image_data.shape)
        self.assertTrue(np.all(noisy_image >= 0))
        self.assertTrue(np.all(noisy_image <= 255))
        self.assertFalse(np.array_equal(noisy_image, self.test_image_data))
        
        print("✅ 均匀噪声测试通过")
    
    def test_blur_noise(self):
        """测试模糊噪声"""
        print("🧪 测试模糊噪声...")
        
        # 启用模糊噪声
        config = self.base_config.copy()
        config['noise']['blur_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声添加
        blurred_image = data_generator.add_blur_noise(self.test_image_data, kernel_size=3, sigma=0.5)
        
        # 验证结果
        self.assertEqual(blurred_image.shape, self.test_image_data.shape)
        self.assertTrue(np.all(blurred_image >= 0))
        self.assertTrue(np.all(blurred_image <= 255))
        self.assertFalse(np.array_equal(blurred_image, self.test_image_data))
        
        print("✅ 模糊噪声测试通过")
    
    def test_noise_probability(self):
        """测试噪声概率控制"""
        print("🧪 测试噪声概率控制...")
        
        # 设置低概率
        config = self.base_config.copy()
        config['noise']['noise_probability'] = 0.0  # 0%概率
        config['noise']['gaussian_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试多次，应该都不应用噪声
        for _ in range(10):
            result = data_generator.apply_noise_augmentation(self.test_image_data)
            self.assertTrue(np.array_equal(result, self.test_image_data))
        
        print("✅ 噪声概率控制测试通过")
    
    def test_noise_disabled(self):
        """测试噪声禁用"""
        print("🧪 测试噪声禁用...")
        
        # 禁用噪声
        config = self.base_config.copy()
        config['noise']['enable_noise'] = False
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声应用
        result = data_generator.apply_noise_augmentation(self.test_image_data)
        
        # 验证结果应该与原图相同
        self.assertTrue(np.array_equal(result, self.test_image_data))
        
        print("✅ 噪声禁用测试通过")
    
    def test_multiple_noise_types(self):
        """测试多种噪声类型组合"""
        print("🧪 测试多种噪声类型组合...")
        
        # 启用多种噪声
        config = self.base_config.copy()
        config['noise']['gaussian_noise']['enable'] = True
        config['noise']['salt_pepper_noise']['enable'] = True
        config['noise']['uniform_noise']['enable'] = True
        
        data_generator = DataGenerator(self.dummy_lines, self.input_shape, 
                                     random=True, augmentation_config=config)
        
        # 测试噪声应用
        result = data_generator.apply_noise_augmentation(self.test_image_data)
        
        # 验证结果
        self.assertEqual(result.shape, self.test_image_data.shape)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
        self.assertFalse(np.array_equal(result, self.test_image_data))
        
        print("✅ 多种噪声类型组合测试通过")


def run_tests():
    """运行所有测试"""
    print("🚀 开始图像噪声增强功能测试")
    print("=" * 50)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNoiseAugmentation)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ 所有测试通过！噪声增强功能正常工作")
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
