"""
图像噪声效果可视化工具
用于预览不同噪声类型对图像的影响效果
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.data.dataset import DataGenerator
from scripts.visualize.visualize_utils import setup_plot_style, save_plot_with_timestamp


def load_config():
    """加载配置文件"""
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_sample_image():
    """查找示例图像"""
    data_dir = os.path.join(project_root, 'data', 'raw')
    
    # 查找第一个可用的图像文件
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    return os.path.join(class_path, img_file)
    
    raise FileNotFoundError("未找到示例图像文件")


def create_noise_configs():
    """创建不同强度的噪声配置"""
    return {
        '原图': None,
        '轻度噪声': {
            'noise': {
                'enable_noise': True,
                'noise_probability': 1.0,  # 确保应用噪声
                'gaussian_noise': {
                    'enable': True,
                    'mean': 0.0,
                    'std': 0.02,
                    'probability': 1.0
                },
                'salt_pepper_noise': {
                    'enable': False
                },
                'uniform_noise': {
                    'enable': False
                },
                'blur_noise': {
                    'enable': False
                }
            }
        },
        '中度噪声': {
            'noise': {
                'enable_noise': True,
                'noise_probability': 1.0,
                'gaussian_noise': {
                    'enable': True,
                    'mean': 0.0,
                    'std': 0.05,
                    'probability': 1.0
                },
                'salt_pepper_noise': {
                    'enable': True,
                    'salt_prob': 0.01,
                    'pepper_prob': 0.01,
                    'probability': 1.0
                },
                'uniform_noise': {
                    'enable': False
                },
                'blur_noise': {
                    'enable': False
                }
            }
        },
        '重度噪声': {
            'noise': {
                'enable_noise': True,
                'noise_probability': 1.0,
                'gaussian_noise': {
                    'enable': True,
                    'mean': 0.0,
                    'std': 0.08,
                    'probability': 1.0
                },
                'salt_pepper_noise': {
                    'enable': True,
                    'salt_prob': 0.02,
                    'pepper_prob': 0.02,
                    'probability': 1.0
                },
                'uniform_noise': {
                    'enable': True,
                    'low': -0.08,
                    'high': 0.08,
                    'probability': 1.0
                },
                'blur_noise': {
                    'enable': True,
                    'kernel_size': 5,
                    'sigma': 1.0,
                    'probability': 1.0
                }
            }
        }
    }


def main():
    """主函数"""
    print("🎨 图像噪声效果可视化工具")
    print("=" * 50)
    
    # ==================== 加载配置和查找图像 ====================
    try:
        config = load_config()
        sample_image_path = find_sample_image()
        print(f"📸 使用示例图像: {sample_image_path}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # ==================== 设置绘图样式 ====================
    chinese_support = setup_plot_style()
    
    # ==================== 生成噪声效果 ====================
    input_shape = [224, 224]
    num_samples = 4  # 每种配置生成4个样本
    noise_configs = create_noise_configs()
    
    fig, axes = plt.subplots(len(noise_configs), num_samples, 
                            figsize=(16, 4 * len(noise_configs)))
    
    if len(noise_configs) == 1:
        axes = axes.reshape(1, -1)
    
    for config_idx, (config_name, noise_config) in enumerate(noise_configs.items()):
        print(f"📸 生成 {config_name} 效果...")
        
        # 创建数据生成器
        dummy_lines = [f"0;{sample_image_path}"]  # 虚拟标注行
        data_generator = DataGenerator(dummy_lines, input_shape, 
                                     random=(noise_config is not None), 
                                     augmentation_config=noise_config)
        
        for sample_idx in range(num_samples):
            if noise_config is None:
                # 原图，不使用增强
                image = Image.open(sample_image_path)
                image_data = data_generator.get_random_data(image, input_shape, random=False)
            else:
                # 使用噪声增强
                image = Image.open(sample_image_path)
                image_data = data_generator.get_random_data(image, input_shape, random=True)
            
            # 确保图像数据在正确范围内
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            
            # 显示图像
            axes[config_idx, sample_idx].imshow(image_data)
            axes[config_idx, sample_idx].set_title(f'{config_name}\n样本 {sample_idx + 1}', 
                                                  fontsize=10)
            axes[config_idx, sample_idx].axis('off')
    
    # ==================== 保存和显示结果 ====================
    plt.suptitle('图像噪声增强效果对比', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.join(project_root, 'outputs', 'plots')
    save_path = save_plot_with_timestamp(fig, output_dir, 'noise_effects', show_plot=True)
    
    print(f"\n✅ 噪声效果可视化完成!")
    print(f"📊 图表已保存: {save_path}")
    print(f"💡 提示: 可以修改配置文件中的噪声参数来调整效果强度")


if __name__ == "__main__":
    main()
