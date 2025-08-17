"""
数据增强可视化脚本
用于预览不同数据增强配置的效果
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.data.dataset import DataGenerator

def visualize_augmentation_effects():
    """可视化数据增强效果"""
    
    # ==================== 配置参数 ====================
    # 选择一张示例图片
    SAMPLE_IMAGE_PATH = 'data/raw/class1_125-175/7_24_19_08-170_001.jpg'
    
    # 不同的增强配置
    augmentation_configs = {
        '原图 (无增强)': None,
        
        '轻度增强': {
            'enable_flip': True,
            'enable_rotation': False,
            'enable_color_jitter': True,
            'enable_scale_jitter': False,
            'jitter': 0.1,
            'hue_range': 0.05,
            'saturation_range': 1.2,
            'value_range': 1.2,
            'flip_probability': 0.5,
        },
        
        '中度增强 (推荐)': {
            'enable_flip': True,
            'enable_rotation': True,
            'enable_color_jitter': True,
            'enable_scale_jitter': True,
            'jitter': 0.3,
            'rotation_range': 15,
            'hue_range': 0.1,
            'saturation_range': 1.5,
            'value_range': 1.5,
            'flip_probability': 0.5,
            'rotation_probability': 0.5,
        },
        
        '重度增强': {
            'enable_flip': True,
            'enable_rotation': True,
            'enable_color_jitter': True,
            'enable_scale_jitter': True,
            'jitter': 0.5,
            'rotation_range': 25,
            'hue_range': 0.2,
            'saturation_range': 2.0,
            'value_range': 2.0,
            'flip_probability': 0.5,
            'rotation_probability': 0.5,
        }
    }
    
    # ==================== 路径处理 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sample_image_path = os.path.join(project_root, SAMPLE_IMAGE_PATH)
    
    if not os.path.exists(sample_image_path):
        print(f"❌ 示例图片不存在: {sample_image_path}")
        print("请修改 SAMPLE_IMAGE_PATH 为有效的图片路径")
        return
    
    # ==================== 生成增强效果 ====================
    input_shape = [224, 224]
    num_samples = 4  # 每种配置生成4个样本
    
    fig, axes = plt.subplots(len(augmentation_configs), num_samples, 
                            figsize=(16, 4 * len(augmentation_configs)))
    
    if len(augmentation_configs) == 1:
        axes = axes.reshape(1, -1)
    
    for config_idx, (config_name, aug_config) in enumerate(augmentation_configs.items()):
        print(f"📸 生成 {config_name} 效果...")
        
        # 创建数据生成器
        dummy_lines = [f"0;{sample_image_path}"]  # 虚拟标注行
        data_generator = DataGenerator(dummy_lines, input_shape, 
                                     random=(aug_config is not None), 
                                     augmentation_config=aug_config)
        
        for sample_idx in range(num_samples):
            if aug_config is None:
                # 原图，不使用增强
                image = Image.open(sample_image_path)
                image_data = data_generator.get_random_data(image, input_shape, random=False)
            else:
                # 使用增强
                image_data, _ = data_generator[0]  # 获取增强后的图像
                # 转换回显示格式
                image_data = np.transpose(image_data, [1, 2, 0])
                image_data = (image_data + 1) * 127.5  # 反向预处理
                image_data = np.clip(image_data, 0, 255).astype(np.uint8)
            
            # 显示图像
            axes[config_idx, sample_idx].imshow(image_data)
            axes[config_idx, sample_idx].set_title(f'{config_name}\n样本 {sample_idx + 1}', 
                                                  fontsize=10)
            axes[config_idx, sample_idx].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'augmentation_effects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 数据增强效果图已保存: {output_path}")
    
    plt.show()


def print_augmentation_info():
    """打印数据增强详细信息"""
    print("🎨 数据增强功能详细说明")
    print("=" * 60)
    
    print("\n📋 可配置的增强类型:")
    print("1. 🔄 水平翻转 (enable_flip)")
    print("   • 随机水平翻转图像")
    print("   • 增加数据多样性，提高泛化能力")
    print("   • 概率控制: flip_probability")
    
    print("\n2. 🔄 随机旋转 (enable_rotation)")
    print("   • 在指定角度范围内随机旋转")
    print("   • 提高对图像方向的鲁棒性")
    print("   • 参数: rotation_range (角度), rotation_probability (概率)")
    
    print("\n3. 🎨 色彩抖动 (enable_color_jitter)")
    print("   • 随机调整色调、饱和度、明度")
    print("   • 提高对光照条件的适应性")
    print("   • 参数: hue_range, saturation_range, value_range")
    
    print("\n4. 📏 尺度抖动 (enable_scale_jitter)")
    print("   • 随机缩放和长宽比变化")
    print("   • 提高对目标大小和形状的适应性")
    print("   • 参数: scale_range, jitter (长宽比)")
    
    print("\n💡 推荐配置:")
    print("• 🥇 中度增强: 平衡效果与训练稳定性")
    print("• 🥈 轻度增强: 保守策略，适合小数据集")
    print("• 🥉 重度增强: 大数据集或需要强泛化能力时")
    
    print("\n⚠️  注意事项:")
    print("• 验证集不使用数据增强，确保评估准确性")
    print("• 过度增强可能导致训练不稳定")
    print("• 根据数据集特点调整增强强度")


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    print("🎨 数据增强可视化工具")
    print("=" * 50)
    
    # 打印详细信息
    print_augmentation_info()
    
    print("\n📸 生成数据增强效果对比图...")
    
    # 生成可视化
    try:
        visualize_augmentation_effects()
        print("\n✅ 可视化完成！")
        print("现在你可以根据效果图调整数据增强配置。")
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        print("请检查示例图片路径是否正确。")
