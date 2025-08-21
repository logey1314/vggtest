"""
模型工厂测试
"""

import unittest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import ModelFactory, create_model, get_available_models


class TestModelFactory(unittest.TestCase):
    """模型工厂测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_config = {
            'name': 'vgg16',
            'pretrained': False,  # 避免下载预训练权重
            'num_classes': 3,
            'dropout': 0.5
        }
    
    def test_get_available_models(self):
        """测试获取可用模型"""
        models = get_available_models()
        self.assertIsInstance(models, dict)
        self.assertIn('vgg16', models)
        self.assertIn('resnet18', models)
        self.assertIn('resnet50', models)
    
    def test_create_vgg16(self):
        """测试创建VGG16模型"""
        config = self.test_config.copy()
        config['name'] = 'vgg16'
        
        model = create_model(config)
        self.assertIsNotNone(model)
        
        # 测试模型输出形状
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 3))
    
    def test_create_resnet18(self):
        """测试创建ResNet18模型"""
        config = self.test_config.copy()
        config['name'] = 'resnet18'
        
        model = create_model(config)
        self.assertIsNotNone(model)
        
        # 测试模型输出形状
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 3))
    
    def test_create_resnet50(self):
        """测试创建ResNet50模型"""
        config = self.test_config.copy()
        config['name'] = 'resnet50'
        
        model = create_model(config)
        self.assertIsNotNone(model)
        
        # 测试模型输出形状
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        self.assertEqual(output.shape, (1, 3))
    
    def test_invalid_model(self):
        """测试无效模型名称"""
        config = self.test_config.copy()
        config['name'] = 'invalid_model'
        
        with self.assertRaises(ValueError):
            create_model(config)
    
    def test_model_factory_validate_config(self):
        """测试配置验证"""
        # 有效配置
        valid_config = self.test_config.copy()
        self.assertTrue(ModelFactory.validate_config(valid_config))
        
        # 无效配置 - 缺少name
        invalid_config = {'num_classes': 3}
        self.assertFalse(ModelFactory.validate_config(invalid_config))
        
        # 无效配置 - 无效的num_classes
        invalid_config = self.test_config.copy()
        invalid_config['num_classes'] = 0
        self.assertFalse(ModelFactory.validate_config(invalid_config))


if __name__ == '__main__':
    unittest.main()
