"""
PPT数据处理可视化生成脚本
生成用于PPT演示的数据处理流程图表
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import subprocess
import tempfile
import cv2
from collections import Counter

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.data.dataset import DataGenerator, preprocess_input, cvtColor
from src.utils.font_manager import setup_matplotlib_font, has_chinese_font, get_font_manager
from scripts.visualize.visualize_utils import create_color_palette


def load_config():
    """加载配置文件"""
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_sample_image():
    """查找示例图像"""
    data_dir = os.path.join(project_root, 'data', 'raw')
    
    # 遍历所有类别文件夹
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            # 查找第一张图片
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    return os.path.join(class_path, file)
    
    raise FileNotFoundError("未找到示例图像")


def create_output_directory():
    """创建输出目录"""
    output_dir = os.path.join(project_root, 'scripts', 'ppt', 'out')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_original_image(sample_image_path, output_dir):
    """生成原始图像展示"""
    print("📸 生成原始图像展示...")

    # 设置字体
    setup_matplotlib_font()
    
    # 加载原始图像
    original_image = Image.open(sample_image_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(original_image)
    ax.set_title('原始混凝土图像', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # 添加图像信息
    width, height = original_image.size
    info_text = f'尺寸: {width} × {height}\n格式: {original_image.mode}\n文件: {os.path.basename(sample_image_path)}'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_original_image.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 原始图像已保存: {output_path}")
    return output_path


def generate_rgb_conversion(sample_image_path, output_dir):
    """生成RGB格式转换对比"""
    print("🎨 生成RGB格式转换对比...")

    setup_matplotlib_font()
    
    # 加载图像
    original_image = Image.open(sample_image_path)
    rgb_image = cvtColor(original_image)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title(f'原始格式 ({original_image.mode})', fontsize=14)
    axes[0].axis('off')
    
    # RGB转换后
    axes[1].imshow(rgb_image)
    axes[1].set_title('RGB格式', fontsize=14)
    axes[1].axis('off')
    
    plt.suptitle('图像格式转换：确保RGB格式', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '02_rgb_conversion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ RGB转换对比已保存: {output_path}")
    return output_path


def generate_resize_effect(sample_image_path, output_dir):
    """生成尺寸调整效果"""
    print("📏 生成尺寸调整效果...")

    setup_matplotlib_font()
    
    # 加载和处理图像
    original_image = Image.open(sample_image_path)
    rgb_image = cvtColor(original_image)
    
    # 模拟DataGenerator的缩放逻辑
    iw, ih = rgb_image.size
    target_w, target_h = 224, 224
    
    # 计算缩放比例
    scale = min(target_w/iw, target_h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    
    # 缩放图像
    resized_image = rgb_image.resize((nw, nh), Image.BICUBIC)
    
    # 添加灰色填充
    final_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    dx = (target_w - nw) // 2
    dy = (target_h - nh) // 2
    final_image.paste(resized_image, (dx, dy))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(rgb_image)
    axes[0].set_title(f'原始尺寸\n{iw} × {ih}', fontsize=12)
    axes[0].axis('off')
    
    # 缩放后图像
    axes[1].imshow(resized_image)
    axes[1].set_title(f'等比缩放\n{nw} × {nh}', fontsize=12)
    axes[1].axis('off')
    
    # 最终图像
    axes[2].imshow(final_image)
    axes[2].set_title(f'填充到目标尺寸\n{target_w} × {target_h}', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle('图像尺寸调整：保持宽高比的智能缩放', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '03_resize_224.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 尺寸调整效果已保存: {output_path}")
    return output_path


def generate_channel_reorder(sample_image_path, output_dir):
    """生成通道重排可视化"""
    print("🔄 生成通道重排可视化...")

    setup_matplotlib_font()
    
    # 处理图像到最终格式
    original_image = Image.open(sample_image_path)
    rgb_image = cvtColor(original_image)
    
    # 调整尺寸
    final_image = Image.new('RGB', (224, 224), (128, 128, 128))
    iw, ih = rgb_image.size
    scale = min(224/iw, 224/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = rgb_image.resize((nw, nh), Image.BICUBIC)
    dx, dy = (224 - nw) // 2, (224 - nh) // 2
    final_image.paste(resized, (dx, dy))
    
    # 转换为numpy数组并应用预处理
    image_array = np.array(final_image, dtype=np.float32)
    normalized_array = preprocess_input(image_array)  # [-1, 1]
    
    # 通道重排：HWC -> CHW
    hwc_format = normalized_array  # (224, 224, 3)
    chw_format = np.transpose(normalized_array, [2, 0, 1])  # (3, 224, 224)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 显示HWC格式
    display_hwc = (hwc_format + 1) * 127.5  # 转回显示范围
    display_hwc = np.clip(display_hwc, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(display_hwc)
    axes[0, 0].set_title('HWC格式\n(Height, Width, Channel)', fontsize=12)
    axes[0, 0].axis('off')
    
    # 显示各个通道
    for i, (channel_name, color) in enumerate([('Red', 'Reds'), ('Green', 'Greens'), ('Blue', 'Blues')]):
        channel_data = hwc_format[:, :, i]
        axes[0, i].imshow(display_hwc)
        axes[0, i].set_title(f'原图', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(channel_data, cmap=color)
        axes[1, i].set_title(f'{channel_name}通道\n数值范围: [{channel_data.min():.2f}, {channel_data.max():.2f}]', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('通道重排与数值归一化：HWC → CHW，[0,255] → [-1,1]', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, '04_channel_reorder.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ 通道重排可视化已保存: {output_path}")
    return output_path


def generate_augmentation_effects(sample_image_path, output_dir):
    """生成数据增强效果对比"""
    print("🎭 生成数据增强效果对比...")

    setup_matplotlib_font()

    # 增强配置
    augmentation_configs = {
        '原图': None,
        '轻度增强': {
            'enable_flip': True,
            'enable_rotation': True,
            'enable_color_jitter': True,
            'flip_probability': 0.3,
            'rotation_range': 10,
            'hue_range': 0.05,
            'saturation_range': 1.2,
            'value_range': 1.2
        },
        '中度增强': {
            'enable_flip': True,
            'enable_rotation': True,
            'enable_color_jitter': True,
            'flip_probability': 0.6,
            'rotation_range': 15,
            'hue_range': 0.1,
            'saturation_range': 1.5,
            'value_range': 1.5
        },
        '重度增强': {
            'enable_flip': True,
            'enable_rotation': True,
            'enable_color_jitter': True,
            'flip_probability': 0.8,
            'rotation_range': 25,
            'hue_range': 0.15,
            'saturation_range': 2.0,
            'value_range': 2.0
        }
    }

    input_shape = [224, 224]
    num_samples = 3

    fig, axes = plt.subplots(len(augmentation_configs), num_samples,
                            figsize=(12, 4 * len(augmentation_configs)))

    for config_idx, (config_name, aug_config) in enumerate(augmentation_configs.items()):
        print(f"  📸 生成 {config_name} 效果...")

        # 创建数据生成器
        dummy_lines = [f"0;{sample_image_path}"]
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
                image = Image.open(sample_image_path)
                image_data = data_generator.get_random_data(image, input_shape, random=True)

            # 确保图像数据在正确范围内
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)

            # 显示图像
            axes[config_idx, sample_idx].imshow(image_data)
            axes[config_idx, sample_idx].set_title(f'{config_name}\n样本 {sample_idx + 1}', fontsize=10)
            axes[config_idx, sample_idx].axis('off')

    plt.suptitle('数据增强效果对比：提升模型泛化能力', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, '05_augmentation_effects.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 数据增强效果对比已保存: {output_path}")
    return output_path


def generate_dataset_distribution(output_dir):
    """生成数据集分布统计图"""
    print("📊 生成数据集分布统计图...")

    setup_matplotlib_font()
    config = load_config()

    # 读取标注文件
    annotation_path = os.path.join(project_root, config['data']['annotation_file'])
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    # 统计类别分布
    class_counts = Counter()
    for line in lines:
        class_id = int(line.split(';')[0])
        class_counts[class_id] += 1

    # 获取类别名称
    class_names = config['data']['class_names']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 柱状图
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    colors = create_color_palette(len(classes))

    bars = ax1.bar([class_names[cls] for cls in classes], counts, color=colors)
    ax1.set_title('各类别样本数量分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('坍落度范围 (mm)', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count:,}', ha='center', va='bottom', fontsize=10)

    # 饼图
    ax2.pie(counts, labels=[f'{class_names[cls]}\n({counts[cls]:,}样本)' for cls in classes],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('样本比例分布', fontsize=14, fontweight='bold')

    # 添加总体统计信息
    total_samples = sum(counts)
    info_text = f'总样本数: {total_samples:,}\n类别数: {len(classes)}\n平均每类: {total_samples//len(classes):,}'
    fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.suptitle('混凝土坍落度数据集分布统计', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, '06_dataset_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 数据集分布统计图已保存: {output_path}")
    return output_path


def generate_vgg_architecture(output_dir):
    """生成VGG架构3D立体图"""
    print("🏗️ 生成VGG架构3D立体图...")

    setup_matplotlib_font()

    # 导入必要的库
    import numpy as np

    # 创建3D图形
    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # VGG16精确架构配置
    vgg_layers = [
        # 输入层
        {'name': 'Input', 'type': 'input', 'size': (224, 224, 3), 'color': '#E8F4FD'},

        # Block 1
        {'name': 'Conv1_1', 'type': 'conv', 'size': (224, 224, 64), 'color': '#FFB6C1'},
        {'name': 'Conv1_2', 'type': 'conv', 'size': (224, 224, 64), 'color': '#FFB6C1'},
        {'name': 'Pool1', 'type': 'pool', 'size': (112, 112, 64), 'color': '#FF6B6B'},

        # Block 2
        {'name': 'Conv2_1', 'type': 'conv', 'size': (112, 112, 128), 'color': '#FFD93D'},
        {'name': 'Conv2_2', 'type': 'conv', 'size': (112, 112, 128), 'color': '#FFD93D'},
        {'name': 'Pool2', 'type': 'pool', 'size': (56, 56, 128), 'color': '#FF6B6B'},

        # Block 3
        {'name': 'Conv3_1', 'type': 'conv', 'size': (56, 56, 256), 'color': '#6BCF7F'},
        {'name': 'Conv3_2', 'type': 'conv', 'size': (56, 56, 256), 'color': '#6BCF7F'},
        {'name': 'Conv3_3', 'type': 'conv', 'size': (56, 56, 256), 'color': '#6BCF7F'},
        {'name': 'Pool3', 'type': 'pool', 'size': (28, 28, 256), 'color': '#FF6B6B'},

        # Block 4
        {'name': 'Conv4_1', 'type': 'conv', 'size': (28, 28, 512), 'color': '#4ECDC4'},
        {'name': 'Conv4_2', 'type': 'conv', 'size': (28, 28, 512), 'color': '#4ECDC4'},
        {'name': 'Conv4_3', 'type': 'conv', 'size': (28, 28, 512), 'color': '#4ECDC4'},
        {'name': 'Pool4', 'type': 'pool', 'size': (14, 14, 512), 'color': '#FF6B6B'},

        # Block 5
        {'name': 'Conv5_1', 'type': 'conv', 'size': (14, 14, 512), 'color': '#A8E6CF'},
        {'name': 'Conv5_2', 'type': 'conv', 'size': (14, 14, 512), 'color': '#A8E6CF'},
        {'name': 'Conv5_3', 'type': 'conv', 'size': (14, 14, 512), 'color': '#A8E6CF'},
        {'name': 'Pool5', 'type': 'pool', 'size': (7, 7, 512), 'color': '#FF6B6B'},

        # Classifier
        {'name': 'FC1', 'type': 'fc', 'size': (1, 1, 4096), 'color': '#DDA0DD'},
        {'name': 'FC2', 'type': 'fc', 'size': (1, 1, 4096), 'color': '#DDA0DD'},
        {'name': 'Output', 'type': 'output', 'size': (1, 1, 4), 'color': '#F0E68C'}
    ]

    def draw_3d_box(ax, x, y, z, width, height, depth, color, alpha=0.8):
        """绘制3D立体方块"""
        # 定义立体方块的8个顶点
        vertices = np.array([
            [x, y, z], [x+width, y, z], [x+width, y+height, z], [x, y+height, z],  # 底面
            [x, y, z+depth], [x+width, y, z+depth], [x+width, y+height, z+depth], [x, y+height, z+depth]  # 顶面
        ])

        # 定义6个面的顶点索引
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # 左面
        ]

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # 修复数据格式 - 直接传递面列表，不需要额外的嵌套
        try:
            poly3d = Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolors='black', linewidths=0.5)
            ax.add_collection3d(poly3d)
        except Exception as e:
            # 如果Poly3DCollection失败，使用简化的绘制方法
            print(f"警告: 3D绘制失败，使用简化方法: {e}")
            # 绘制线框
            for face in faces:
                face_array = np.array(face)
                # 闭合面
                face_closed = np.vstack([face_array, face_array[0]])
                ax.plot(face_closed[:, 0], face_closed[:, 1], face_closed[:, 2], 'k-', alpha=0.3)

            # 绘制填充的顶面（简化版）
            top_face = faces[1]  # 顶面
            top_array = np.array(top_face)
            ax.plot_trisurf(top_array[:, 0], top_array[:, 1], top_array[:, 2], color=color, alpha=alpha)

    # 绘制架构
    x_pos = 0
    max_height = 0

    for i, layer in enumerate(vgg_layers):
        h, w, c = layer['size']

        # 计算3D方块的尺寸（按比例缩放）
        if layer['type'] == 'input':
            box_width = 3
            box_height = 3
            box_depth = 0.3
        elif layer['type'] in ['conv', 'pool']:
            # 根据特征图尺寸调整方块大小
            scale_factor = min(h/224, 1.0)  # 相对于输入尺寸的缩放
            box_width = 2 + scale_factor * 2
            box_height = 2 + scale_factor * 2
            box_depth = min(c/64, 4)  # 根据通道数调整深度
        elif layer['type'] == 'fc':
            box_width = 1
            box_height = 1
            box_depth = 3
        else:  # output
            box_width = 1
            box_height = 1
            box_depth = 0.5

        # 绘制3D方块
        y_pos = 0
        z_pos = 0
        draw_3d_box(ax, x_pos, y_pos, z_pos, box_width, box_height, box_depth, layer['color'])

        # 添加尺寸标注
        size_text = f"{h} × {w} × {c}"
        ax.text(x_pos + box_width/2, y_pos + box_height + 0.5, z_pos + box_depth/2,
                size_text, fontsize=9, ha='center', va='bottom')

        # 添加层名称（在方块下方）
        if layer['type'] != 'input':
            ax.text(x_pos + box_width/2, y_pos - 0.5, z_pos + box_depth/2,
                    layer['name'], fontsize=8, ha='center', va='top')

        # 更新位置
        x_pos += box_width + 1
        max_height = max(max_height, box_height)

    # 设置坐标轴
    ax.set_xlim(0, x_pos)
    ax.set_ylim(-2, max_height + 2)
    ax.set_zlim(0, 5)

    # 设置视角
    ax.view_init(elev=20, azim=45)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # 添加标题
    ax.set_title('VGG16网络架构3D可视化', fontsize=18, fontweight='bold', pad=20)

    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', label='卷积层 (convolution + ReLU)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='最大池化层 (max pooling)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#DDA0DD', label='全连接层 (fully connected + ReLU)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#F0E68C', label='输出层 (softmax)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, '07_vgg_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ VGG架构3D立体图已保存: {output_path}")
    return output_path


def generate_vgg_plotneuralnet(output_dir):
    """使用PlotNeuralNet风格生成VGG16架构图"""
    print("🏗️ 生成PlotNeuralNet风格VGG16架构图...")

    # PlotNeuralNet风格的LaTeX代码模板
    latex_template = r"""
\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image

\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}

\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]

% Input Image
\node[canvas is zy plane at x=0] (temp) at (-3,0,0) {\includegraphics[width=8cm,height=8cm]{example-image}};

% VGG16 Architecture
% Block 1
\pic[shift={(0,0,0)}] at (0,0,0) {Box={name=conv1_1,caption=Conv1\_1,xlabel={{64, }},zlabel=224,fill=\ConvColor,height=40,width=2,depth=40}};
\pic[shift={(0,0,0)}] at (conv1_1-east) {Box={name=conv1_2,caption=Conv1\_2,xlabel={{64, }},zlabel=224,fill=\ConvColor,height=40,width=2,depth=40}};
\pic[shift={(1,0,0)}] at (conv1_2-east) {Box={name=pool1,caption=Pool1,xlabel={{64, }},zlabel=112,fill=\PoolColor,height=20,width=1,depth=20}};

% Block 2
\pic[shift={(1,0,0)}] at (pool1-east) {Box={name=conv2_1,caption=Conv2\_1,xlabel={{128, }},zlabel=112,fill=\ConvColor,height=20,width=2.5,depth=20}};
\pic[shift={(0,0,0)}] at (conv2_1-east) {Box={name=conv2_2,caption=Conv2\_2,xlabel={{128, }},zlabel=112,fill=\ConvColor,height=20,width=2.5,depth=20}};
\pic[shift={(1,0,0)}] at (conv2_2-east) {Box={name=pool2,caption=Pool2,xlabel={{128, }},zlabel=56,fill=\PoolColor,height=10,width=1,depth=10}};

% Block 3
\pic[shift={(1,0,0)}] at (pool2-east) {Box={name=conv3_1,caption=Conv3\_1,xlabel={{256, }},zlabel=56,fill=\ConvColor,height=10,width=3,depth=10}};
\pic[shift={(0,0,0)}] at (conv3_1-east) {Box={name=conv3_2,caption=Conv3\_2,xlabel={{256, }},zlabel=56,fill=\ConvColor,height=10,width=3,depth=10}};
\pic[shift={(0,0,0)}] at (conv3_2-east) {Box={name=conv3_3,caption=Conv3\_3,xlabel={{256, }},zlabel=56,fill=\ConvColor,height=10,width=3,depth=10}};
\pic[shift={(1,0,0)}] at (conv3_3-east) {Box={name=pool3,caption=Pool3,xlabel={{256, }},zlabel=28,fill=\PoolColor,height=5,width=1,depth=5}};

% Block 4
\pic[shift={(1,0,0)}] at (pool3-east) {Box={name=conv4_1,caption=Conv4\_1,xlabel={{512, }},zlabel=28,fill=\ConvColor,height=5,width=4,depth=5}};
\pic[shift={(0,0,0)}] at (conv4_1-east) {Box={name=conv4_2,caption=Conv4\_2,xlabel={{512, }},zlabel=28,fill=\ConvColor,height=5,width=4,depth=5}};
\pic[shift={(0,0,0)}] at (conv4_2-east) {Box={name=conv4_3,caption=Conv4\_3,xlabel={{512, }},zlabel=28,fill=\ConvColor,height=5,width=4,depth=5}};
\pic[shift={(1,0,0)}] at (conv4_3-east) {Box={name=pool4,caption=Pool4,xlabel={{512, }},zlabel=14,fill=\PoolColor,height=2.5,width=1,depth=2.5}};

% Block 5
\pic[shift={(1,0,0)}] at (pool4-east) {Box={name=conv5_1,caption=Conv5\_1,xlabel={{512, }},zlabel=14,fill=\ConvColor,height=2.5,width=4,depth=2.5}};
\pic[shift={(0,0,0)}] at (conv5_1-east) {Box={name=conv5_2,caption=Conv5\_2,xlabel={{512, }},zlabel=14,fill=\ConvColor,height=2.5,width=4,depth=2.5}};
\pic[shift={(0,0,0)}] at (conv5_2-east) {Box={name=conv5_3,caption=Conv5\_3,xlabel={{512, }},zlabel=14,fill=\ConvColor,height=2.5,width=4,depth=2.5}};
\pic[shift={(1,0,0)}] at (conv5_3-east) {Box={name=pool5,caption=Pool5,xlabel={{512, }},zlabel=7,fill=\PoolColor,height=1.5,width=1,depth=1.5}};

% Classifier
\pic[shift={(2,0,0)}] at (pool5-east) {Box={name=fc1,caption=FC1,xlabel={{4096, }},zlabel=1,fill=\FcColor,height=1,width=6,depth=1}};
\pic[shift={(0,0,0)}] at (fc1-east) {Box={name=fc2,caption=FC2,xlabel={{4096, }},zlabel=1,fill=\FcColor,height=1,width=6,depth=1}};
\pic[shift={(0,0,0)}] at (fc2-east) {Box={name=fc3,caption=Output,xlabel={{4, }},zlabel=1,fill=\SoftmaxColor,height=1,width=1,depth=1}};

% Connections
\draw [connection]  (conv1_1-east)    -- node {\midarrow} (conv1_2-west);
\draw [connection]  (conv1_2-east)    -- node {\midarrow} (pool1-west);
\draw [connection]  (pool1-east)      -- node {\midarrow} (conv2_1-west);
\draw [connection]  (conv2_1-east)    -- node {\midarrow} (conv2_2-west);
\draw [connection]  (conv2_2-east)    -- node {\midarrow} (pool2-west);
\draw [connection]  (pool2-east)      -- node {\midarrow} (conv3_1-west);
\draw [connection]  (conv3_1-east)    -- node {\midarrow} (conv3_2-west);
\draw [connection]  (conv3_2-east)    -- node {\midarrow} (conv3_3-west);
\draw [connection]  (conv3_3-east)    -- node {\midarrow} (pool3-west);
\draw [connection]  (pool3-east)      -- node {\midarrow} (conv4_1-west);
\draw [connection]  (conv4_1-east)    -- node {\midarrow} (conv4_2-west);
\draw [connection]  (conv4_2-east)    -- node {\midarrow} (conv4_3-west);
\draw [connection]  (conv4_3-east)    -- node {\midarrow} (pool4-west);
\draw [connection]  (pool4-east)      -- node {\midarrow} (conv5_1-west);
\draw [connection]  (conv5_1-east)    -- node {\midarrow} (conv5_2-west);
\draw [connection]  (conv5_2-east)    -- node {\midarrow} (conv5_3-west);
\draw [connection]  (conv5_3-east)    -- node {\midarrow} (pool5-west);
\draw [connection]  (pool5-east)      -- node {\midarrow} (fc1-west);
\draw [connection]  (fc1-east)        -- node {\midarrow} (fc2-west);
\draw [connection]  (fc2-east)        -- node {\midarrow} (fc3-west);

\end{tikzpicture}
\end{document}
"""

    try:
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_file = os.path.join(temp_dir, 'vgg16_architecture.tex')

            # 写入LaTeX文件
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_template)

            print(f"📄 LaTeX文件已生成: {tex_file}")

            # 尝试编译LaTeX（如果系统有LaTeX）
            try:
                # 编译为PDF
                result = subprocess.run(['pdflatex', '-output-directory', temp_dir, tex_file],
                                      capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    pdf_file = os.path.join(temp_dir, 'vgg16_architecture.pdf')

                    # 转换PDF为PNG（如果有ImageMagick或其他工具）
                    try:
                        png_output = os.path.join(output_dir, '07_vgg_plotneuralnet.png')
                        subprocess.run(['convert', '-density', '300', pdf_file, png_output],
                                     capture_output=True, timeout=30)
                        print(f"✅ PlotNeuralNet风格VGG架构图已保存: {png_output}")
                        return png_output
                    except:
                        # 如果转换失败，复制PDF文件
                        pdf_output = os.path.join(output_dir, '07_vgg_plotneuralnet.pdf')
                        import shutil
                        shutil.copy2(pdf_file, pdf_output)
                        print(f"✅ PlotNeuralNet风格VGG架构PDF已保存: {pdf_output}")
                        return pdf_output
                else:
                    print(f"⚠️ LaTeX编译失败: {result.stderr}")

            except subprocess.TimeoutExpired:
                print("⚠️ LaTeX编译超时")
            except FileNotFoundError:
                print("⚠️ 未找到pdflatex命令，请安装LaTeX")

            # 如果LaTeX编译失败，保存LaTeX源文件
            tex_output = os.path.join(output_dir, '07_vgg_plotneuralnet.tex')
            import shutil
            shutil.copy2(tex_file, tex_output)
            print(f"📄 LaTeX源文件已保存: {tex_output}")
            print("💡 提示: 请手动使用LaTeX编译器编译此文件")
            return tex_output

    except Exception as e:
        print(f"❌ 生成PlotNeuralNet架构图失败: {e}")

        # 降级方案：生成简化的matplotlib版本
        print("🔄 使用降级方案生成简化架构图...")
        return generate_simplified_vgg_architecture(output_dir)


def generate_simplified_vgg_architecture(output_dir):
    """生成简化的VGG架构图（降级方案）"""
    setup_matplotlib_font()

    fig, ax = plt.subplots(figsize=(18, 10))

    # VGG16层配置
    layers = [
        {'name': 'Input\n224×224×3', 'color': '#E8F4FD', 'width': 2, 'height': 3},
        {'name': 'Conv1_1\n224×224×64', 'color': '#FFB6C1', 'width': 1.8, 'height': 2.8},
        {'name': 'Conv1_2\n224×224×64', 'color': '#FFB6C1', 'width': 1.8, 'height': 2.8},
        {'name': 'Pool1\n112×112×64', 'color': '#FF6B6B', 'width': 1.5, 'height': 2.3},
        {'name': 'Conv2_1\n112×112×128', 'color': '#FFD93D', 'width': 1.5, 'height': 2.3},
        {'name': 'Conv2_2\n112×112×128', 'color': '#FFD93D', 'width': 1.5, 'height': 2.3},
        {'name': 'Pool2\n56×56×128', 'color': '#FF6B6B', 'width': 1.2, 'height': 1.8},
        {'name': 'Conv3_1\n56×56×256', 'color': '#6BCF7F', 'width': 1.2, 'height': 1.8},
        {'name': 'Conv3_2\n56×56×256', 'color': '#6BCF7F', 'width': 1.2, 'height': 1.8},
        {'name': 'Conv3_3\n56×56×256', 'color': '#6BCF7F', 'width': 1.2, 'height': 1.8},
        {'name': 'Pool3\n28×28×256', 'color': '#FF6B6B', 'width': 1, 'height': 1.5},
        {'name': 'Conv4_1\n28×28×512', 'color': '#4ECDC4', 'width': 1, 'height': 1.5},
        {'name': 'Conv4_2\n28×28×512', 'color': '#4ECDC4', 'width': 1, 'height': 1.5},
        {'name': 'Conv4_3\n28×28×512', 'color': '#4ECDC4', 'width': 1, 'height': 1.5},
        {'name': 'Pool4\n14×14×512', 'color': '#FF6B6B', 'width': 0.8, 'height': 1.2},
        {'name': 'Conv5_1\n14×14×512', 'color': '#A8E6CF', 'width': 0.8, 'height': 1.2},
        {'name': 'Conv5_2\n14×14×512', 'color': '#A8E6CF', 'width': 0.8, 'height': 1.2},
        {'name': 'Conv5_3\n14×14×512', 'color': '#A8E6CF', 'width': 0.8, 'height': 1.2},
        {'name': 'Pool5\n7×7×512', 'color': '#FF6B6B', 'width': 0.6, 'height': 1},
        {'name': 'FC1\n4096', 'color': '#DDA0DD', 'width': 0.5, 'height': 2},
        {'name': 'FC2\n4096', 'color': '#DDA0DD', 'width': 0.5, 'height': 2},
        {'name': 'Output\n4 classes', 'color': '#F0E68C', 'width': 0.5, 'height': 1}
    ]

    # 绘制层
    x_pos = 0
    for i, layer in enumerate(layers):
        # 绘制矩形
        rect = plt.Rectangle((x_pos, 0), layer['width'], layer['height'],
                           facecolor=layer['color'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # 添加文本
        ax.text(x_pos + layer['width']/2, layer['height']/2, layer['name'],
                ha='center', va='center', fontsize=8, fontweight='bold')

        # 添加箭头（除了最后一层）
        if i < len(layers) - 1:
            ax.arrow(x_pos + layer['width'], layer['height']/2, 0.1, 0,
                    head_width=0.1, head_length=0.05, fc='gray', ec='gray')

        x_pos += layer['width'] + 0.15

    # 设置坐标轴
    ax.set_xlim(-0.5, x_pos)
    ax.set_ylim(-0.5, 4)
    ax.set_title('VGG16网络架构图 (PlotNeuralNet风格)', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', label='卷积层'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FF6B6B', label='池化层'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#DDA0DD', label='全连接层'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#F0E68C', label='输出层')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, '07_vgg_plotneuralnet_simplified.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ 简化版VGG架构图已保存: {output_path}")
    return output_path


def generate_vgg_vs_human_comparison(output_dir):
    """生成VGG vs 人眼对比图"""
    print("👁️ 生成VGG vs 人眼对比图...")

    setup_matplotlib_font()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # 左侧：人眼识别过程
    ax1.set_title('人眼识别过程', fontsize=16, fontweight='bold', pad=20)

    # 人眼识别步骤
    human_steps = [
        {'step': '1. 观察图像', 'desc': '用眼睛看混凝土图片', 'time': '1-2秒'},
        {'step': '2. 经验判断', 'desc': '基于以往经验分析', 'time': '3-5秒'},
        {'step': '3. 主观评估', 'desc': '凭感觉估计坍落度', 'time': '2-3秒'},
        {'step': '4. 给出结果', 'desc': '"大概是中等坍落度"', 'time': '1秒'}
    ]

    for i, step in enumerate(human_steps):
        y_pos = 3.5 - i * 0.8

        # 绘制步骤框
        rect = plt.Rectangle((0.1, y_pos - 0.3), 3.8, 0.6,
                           facecolor='#FFE6E6', edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)

        # 添加步骤文本
        ax1.text(0.2, y_pos + 0.1, step['step'], fontsize=12, fontweight='bold')
        ax1.text(0.2, y_pos - 0.1, step['desc'], fontsize=10)
        ax1.text(3.5, y_pos, step['time'], fontsize=9, style='italic', ha='right')

        # 添加箭头（除了最后一步）
        if i < len(human_steps) - 1:
            ax1.arrow(2, y_pos - 0.35, 0, -0.1, head_width=0.1, head_length=0.05,
                     fc='gray', ec='gray')

    # 人眼识别特点
    human_features = [
        "✓ 快速直观",
        "✗ 主观性强",
        "✗ 容易疲劳",
        "✗ 结果不一致",
        "✗ 依赖经验"
    ]

    ax1.text(0.1, -0.5, "特点：", fontsize=12, fontweight='bold')
    for i, feature in enumerate(human_features):
        color = 'green' if feature.startswith('✓') else 'red'
        ax1.text(0.1, -0.8 - i * 0.2, feature, fontsize=10, color=color)

    # 右侧：VGG识别过程
    ax2.set_title('VGG模型识别过程', fontsize=16, fontweight='bold', pad=20)

    # VGG识别步骤
    vgg_steps = [
        {'step': '1. 图像预处理', 'desc': '标准化为224×224像素', 'time': '0.01秒'},
        {'step': '2. 特征提取', 'desc': '16层神经网络分析', 'time': '0.1秒'},
        {'step': '3. 特征整合', 'desc': '全连接层综合判断', 'time': '0.01秒'},
        {'step': '4. 输出结果', 'desc': '"160-200mm，置信度95%"', 'time': '0.001秒'}
    ]

    for i, step in enumerate(vgg_steps):
        y_pos = 3.5 - i * 0.8

        # 绘制步骤框
        rect = plt.Rectangle((0.1, y_pos - 0.3), 3.8, 0.6,
                           facecolor='#E6F3FF', edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)

        # 添加步骤文本
        ax2.text(0.2, y_pos + 0.1, step['step'], fontsize=12, fontweight='bold')
        ax2.text(0.2, y_pos - 0.1, step['desc'], fontsize=10)
        ax2.text(3.5, y_pos, step['time'], fontsize=9, style='italic', ha='right')

        # 添加箭头（除了最后一步）
        if i < len(vgg_steps) - 1:
            ax2.arrow(2, y_pos - 0.35, 0, -0.1, head_width=0.1, head_length=0.05,
                     fc='blue', ec='blue')

    # VGG识别特点
    vgg_features = [
        "✓ 极速处理",
        "✓ 客观准确",
        "✓ 永不疲劳",
        "✓ 结果一致",
        "✓ 持续学习"
    ]

    ax2.text(0.1, -0.5, "特点：", fontsize=12, fontweight='bold')
    for i, feature in enumerate(vgg_features):
        ax2.text(0.1, -0.8 - i * 0.2, feature, fontsize=10, color='green')

    # 设置坐标轴
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 4)
        ax.set_ylim(-2, 4)
        ax.axis('off')

    # 添加总标题
    fig.suptitle('人眼 vs VGG模型：混凝土坍落度识别对比', fontsize=18, fontweight='bold')

    # 添加总结对比
    comparison_text = (
        "总结对比：\n"
        "• 速度：VGG模型快100倍以上\n"
        "• 准确率：VGG模型达到90%+\n"
        "• 一致性：VGG模型结果完全一致\n"
        "• 成本：VGG模型一次训练，重复使用"
    )
    fig.text(0.5, 0.02, comparison_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(output_dir, '08_vgg_vs_human.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ VGG vs 人眼对比图已保存: {output_path}")
    return output_path


def main():
    """主函数"""
    print("🎨 PPT数据处理可视化生成工具")
    print("=" * 60)

    # 设置字体并显示状态
    print("🔤 设置字体...")
    setup_matplotlib_font()
    font_manager = get_font_manager()
    font_manager.print_font_status()

    try:
        # 创建输出目录
        output_dir = create_output_directory()
        print(f"📁 输出目录: {output_dir}")

        # 查找示例图像
        sample_image_path = find_sample_image()
        print(f"📸 示例图像: {sample_image_path}")

        # 生成各种可视化图表
        generated_files = []

        generated_files.append(generate_original_image(sample_image_path, output_dir))
        generated_files.append(generate_rgb_conversion(sample_image_path, output_dir))
        generated_files.append(generate_resize_effect(sample_image_path, output_dir))
        generated_files.append(generate_channel_reorder(sample_image_path, output_dir))
        generated_files.append(generate_augmentation_effects(sample_image_path, output_dir))
        generated_files.append(generate_dataset_distribution(output_dir))
        generated_files.append(generate_vgg_architecture(output_dir))
        generated_files.append(generate_vgg_plotneuralnet(output_dir))  # 新增PlotNeuralNet版本
        generated_files.append(generate_vgg_vs_human_comparison(output_dir))

        print("\n" + "=" * 60)
        print("✅ 所有PPT可视化图表生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print("\n📊 生成的图表文件:")
        for i, file_path in enumerate(generated_files, 1):
            filename = os.path.basename(file_path)
            print(f"   {i}. {filename}")

        print("\n💡 使用建议:")
        print("   • 这些图表可以直接用于PPT演示")
        print("   • 所有图表都是300 DPI高分辨率")
        print("   • 图表按照数据处理流程顺序命名")

    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
