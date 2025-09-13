"""
独立特征可视化分析工具
专门用于混凝土坍落度识别的深度特征可视化分析

功能包括:
- 完整的13层VGG16卷积特征图可视化
- 增强的类激活映射(CAM)分析
- t-SNE降维可视化
- 混凝土坍落度专用特征分析
- 统计分析和类别对比

使用方法:
1. 修改main函数中的train_path
2. 直接在PyCharm中运行此脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import yaml
import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from scripts.visualize.visualize_utils import setup_plot_style, save_plot_with_timestamp
from src.models.model_factory import create_model
from src.data.dataset import DataGenerator
from torch.utils.data import DataLoader
from src.utils.font_manager import setup_matplotlib_font, get_font_manager
from scripts.evaluate.evaluate_utils import (
    extract_timestamp_from_path,
    find_model_file,
    find_training_config,
    load_training_config,
    generate_eval_timestamp
)


class FeatureExtractor:
    """特征提取器 - 用于提取VGG16模型的中间层特征"""
    
    def __init__(self, model, device='cpu'):
        """
        初始化特征提取器
        
        Args:
            model: VGG16模型实例
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # 定义VGG16的完整13层卷积层位置
        self.layer_names = {
            # Block 1: 低级特征（边缘、纹理）- 224×224
            'conv1_1': 0,   # 64通道 - 基础边缘检测
            'conv1_2': 2,   # 64通道 - 增强边缘特征

            # Block 2: 中低级特征（形状、模式）- 112×112
            'conv2_1': 5,   # 128通道 - 简单形状
            'conv2_2': 7,   # 128通道 - 复杂形状

            # Block 3: 中级特征（局部模式）- 56×56
            'conv3_1': 10,  # 256通道 - 局部纹理模式
            'conv3_2': 12,  # 256通道 - 增强纹理特征
            'conv3_3': 14,  # 256通道 - 复杂纹理组合

            # Block 4: 中高级特征（对象部分）- 28×28
            'conv4_1': 17,  # 512通道 - 混凝土表面特征
            'conv4_2': 19,  # 512通道 - 坍落度相关特征
            'conv4_3': 21,  # 512通道 - 高级表面模式

            # Block 5: 高级特征（语义特征）- 14×14
            'conv5_1': 24,  # 512通道 - 坍落度语义特征
            'conv5_2': 26,  # 512通道 - 分类决策特征
            'conv5_3': 28,  # 512通道 - 最终语义表示
        }

        # 按Block组织的层名称
        self.layer_blocks = {
            'Block1_EdgeTexture': ['conv1_1', 'conv1_2'],
            'Block2_ShapePattern': ['conv2_1', 'conv2_2'],
            'Block3_LocalPattern': ['conv3_1', 'conv3_2', 'conv3_3'],
            'Block4_ObjectPart': ['conv4_1', 'conv4_2', 'conv4_3'],
            'Block5_Semantic': ['conv5_1', 'conv5_2', 'conv5_3']
        }

        # 完整的13层列表
        self.all_conv_layers = [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'
        ]
        
        # 存储特征图的钩子
        self.features = {}
        self.hooks = []
        
    def register_hooks(self, layer_names: List[str]):
        """
        注册前向钩子以提取指定层的特征
        
        Args:
            layer_names: 要提取特征的层名称列表
        """
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        # 清除之前的钩子
        self.remove_hooks()
        
        # 注册新的钩子
        for layer_name in layer_names:
            if layer_name in self.layer_names:
                layer_idx = self.layer_names[layer_name]
                hook = self.model.features[layer_idx].register_forward_hook(get_activation(layer_name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}
    
    def extract_features(self, image_tensor: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        提取指定层的特征图

        Args:
            image_tensor: 输入图像张量 [1, 3, H, W]
            layer_names: 要提取的层名称列表

        Returns:
            Dict[str, torch.Tensor]: 层名称到特征图的映射
        """
        # 验证并过滤层名称，只保留卷积层
        valid_layer_names = self._validate_layer_names(layer_names)
        if not valid_layer_names:
            print("⚠️  没有有效的卷积层可以提取特征")
            return {}

        self.register_hooks(valid_layer_names)

        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                _ = self.model(image_tensor)

            # 验证提取的特征维度
            validated_features = self._validate_feature_dimensions(self.features.copy())
            return validated_features

        except Exception as e:
            print(f"⚠️  特征提取失败: {e}")
            self.remove_hooks()
            return {}

    def _validate_layer_names(self, layer_names: List[str]) -> List[str]:
        """
        验证并过滤层名称，只保留有效的卷积层

        Args:
            layer_names: 要验证的层名称列表

        Returns:
            List[str]: 有效的卷积层名称列表
        """
        valid_layers = []
        for layer_name in layer_names:
            if layer_name in self.layer_names:
                # 确保是卷积层（conv开头）
                if layer_name.startswith('conv'):
                    valid_layers.append(layer_name)
                else:
                    print(f"⚠️  跳过非卷积层: {layer_name}")
            else:
                print(f"⚠️  未知层名称: {layer_name}")

        return valid_layers

    def _validate_feature_dimensions(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        验证特征图维度，过滤掉异常的特征

        Args:
            features: 原始特征字典

        Returns:
            Dict[str, torch.Tensor]: 验证后的特征字典
        """
        validated_features = {}

        for layer_name, feature_tensor in features.items():
            try:
                # 检查特征图维度
                if len(feature_tensor.shape) == 4:  # [B, C, H, W]
                    batch_size, channels, height, width = feature_tensor.shape

                    # 验证维度合理性 - 放宽批次大小限制
                    if batch_size > 0 and channels > 0 and height > 0 and width > 0:
                        # 检查特征图尺寸是否合理（不能太大或太小）
                        if 1 <= height <= 224 and 1 <= width <= 224 and 1 <= channels <= 512:
                            validated_features[layer_name] = feature_tensor
                            print(f"✅ {layer_name}: {batch_size}×{channels}×{height}×{width}")
                        else:
                            print(f"⚠️  {layer_name} 维度异常: {channels}×{height}×{width}")
                    else:
                        print(f"⚠️  {layer_name} 批次或维度异常: {feature_tensor.shape}")
                else:
                    print(f"⚠️  {layer_name} 不是4D特征图: {feature_tensor.shape}")

            except Exception as e:
                print(f"⚠️  验证 {layer_name} 特征时出错: {e}")

        return validated_features


class GradCAM:
    """Grad-CAM可视化类"""
    
    def __init__(self, model, target_layer_name: str, device='cpu'):
        """
        初始化Grad-CAM
        
        Args:
            model: VGG16模型实例
            target_layer_name: 目标层名称（通常是最后一个卷积层）
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(model, device)
        
        # 存储梯度和特征图
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        """保存梯度的钩子函数"""
        self.gradients = grad
    
    def generate_cam(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        生成类激活映射
        
        Args:
            image_tensor: 输入图像张量 [1, 3, H, W]
            class_idx: 目标类别索引
            
        Returns:
            np.ndarray: CAM热力图
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_()
        
        # 注册钩子提取特征和梯度
        self.feature_extractor.register_hooks([self.target_layer_name])
        
        # 前向传播
        output = self.model(image_tensor)
        
        # 获取特征图
        self.activations = self.feature_extractor.features[self.target_layer_name]
        
        # 注册梯度钩子
        self.activations.register_hook(self.save_gradient)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 计算权重（梯度的全局平均池化）
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化到0-1
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # 清理钩子
        self.feature_extractor.remove_hooks()
        
        return cam


class TSNEVisualizer:
    """t-SNE降维可视化类"""

    def __init__(self, class_names: List[str], use_english_labels: bool = False):
        """
        初始化t-SNE可视化器

        Args:
            class_names: 类别名称列表
            use_english_labels: 是否使用英文标签
        """
        self.class_names = class_names
        self.use_english_labels = use_english_labels

    def extract_features_for_tsne(self, model, dataloader, layer_name: str, device='cpu', max_samples=100):
        """
        提取指定层的特征用于t-SNE分析

        Args:
            model: 模型实例
            dataloader: 数据加载器
            layer_name: 要提取的层名称
            device: 计算设备
            max_samples: 最大样本数量

        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 标签数组)
        """
        model.eval()
        features_list = []
        labels_list = []

        # 创建特征提取器
        feature_extractor = FeatureExtractor(model, device)

        print(f"🔍 提取 {layer_name} 层特征用于t-SNE分析...")

        with torch.no_grad():
            sample_count = 0
            for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="提取特征")):
                # 处理不同的数据格式
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                elif len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    print(f"⚠️  未知的数据格式: {len(batch_data)} 个元素")
                    continue
                if sample_count >= max_samples:
                    break

                # 确保张量维度正确 [B, 3, H, W]
                if len(images.shape) == 5:  # [B, 1, 3, H, W] -> [B, 3, H, W]
                    images = images.squeeze(1)
                elif len(images.shape) == 3:  # [3, H, W] -> [1, 3, H, W]
                    images = images.unsqueeze(0)

                images = images.to(device)

                # 提取特征
                try:
                    layer_features = feature_extractor.extract_features(images, [layer_name])
                except Exception as e:
                    print(f"⚠️  特征提取失败: {e}")
                    continue

                if layer_name in layer_features:
                    # 获取特征并展平
                    feature_map = layer_features[layer_name]  # [B, C, H, W]
                    batch_size = feature_map.size(0)

                    # 验证特征图维度是否正确
                    if len(feature_map.shape) == 4 and batch_size > 0:
                        # 全局平均池化
                        pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1))  # [B, C, 1, 1]
                        pooled_features = pooled_features.view(batch_size, -1)  # [B, C]

                        features_list.append(pooled_features.cpu().numpy())
                        labels_list.extend(labels.cpu().numpy())

                        sample_count += batch_size
                        print(f"    ✅ {layer_name} 特征提取成功: {feature_map.shape} -> {pooled_features.shape}")
                    else:
                        print(f"    ⚠️  {layer_name} 特征图维度异常: {feature_map.shape}")
                else:
                    print(f"    ⚠️  {layer_name} 特征未找到")

        # 清理钩子
        feature_extractor.remove_hooks()

        if features_list:
            features_matrix = np.vstack(features_list)
            labels_array = np.array(labels_list)
            print(f"✅ 提取完成: {features_matrix.shape[0]} 个样本, {features_matrix.shape[1]} 维特征")
            return features_matrix, labels_array
        else:
            print("❌ 未能提取到任何特征")
            return None, None

    def visualize_tsne(self, features: np.ndarray, labels: np.ndarray,
                      layer_name: str, save_dir: str, perplexity=30) -> str:
        """
        生成t-SNE可视化图

        Args:
            features: 特征矩阵 [N, D]
            labels: 标签数组 [N]
            layer_name: 层名称
            save_dir: 保存目录
            perplexity: t-SNE困惑度参数

        Returns:
            str: 保存的文件路径
        """
        print(f"🎨 生成 {layer_name} 的t-SNE可视化...")

        # 如果特征维度太高，先用PCA降维
        if features.shape[1] > 50:
            print(f"   特征维度较高({features.shape[1]})，先用PCA降维到50维")
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)

        # t-SNE降维
        print(f"   执行t-SNE降维 (perplexity={perplexity})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                   init='pca', learning_rate='auto')
        features_2d = tsne.fit_transform(features)

        # 创建可视化
        fig, ax = plt.subplots(figsize=(12, 10))

        # 为每个类别分配颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))

        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if np.any(mask):
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                          c=[colors[i]], label=class_name, alpha=0.7, s=50)

        # 使用英文或中文标题
        if hasattr(self, 'use_english_labels') and self.use_english_labels:
            ax.set_title(f'{layer_name} Feature Space t-SNE Visualization', fontsize=16, pad=20)
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.legend(title='Slump Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.set_title(f'{layer_name} 特征空间 t-SNE 可视化', fontsize=16, pad=20)
            ax.set_xlabel('t-SNE 维度 1', fontsize=12)
            ax.set_ylabel('t-SNE 维度 2', fontsize=12)
            ax.legend(title='坍落度类别', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        info_text = f"样本数: {len(features_2d)}\n原始维度: {features.shape[1]}\n困惑度: {perplexity}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_plot_with_timestamp(fig, save_dir, f'tsne_{layer_name}', show_plot=False)

        return save_path


class FeatureVisualizer:
    """特征可视化主类"""
    
    def __init__(self, model, class_names: List[str], device='cpu'):
        """
        初始化特征可视化器
        
        Args:
            model: VGG16模型实例
            class_names: 类别名称列表
            device: 计算设备
        """
        self.model = model
        self.class_names = class_names
        self.device = device

        # 先设置绘图样式和字体
        setup_plot_style()
        has_chinese_font = setup_matplotlib_font()

        # 检查字体支持
        self.use_english_labels = not has_chinese_font
        if self.use_english_labels:
            print("⚠️  未检测到中文字体，将使用英文标题")

        # 初始化组件（在字体设置之后）
        self.feature_extractor = FeatureExtractor(model, device)
        self.grad_cam = GradCAM(model, 'conv5_3', device)  # 使用最后一个卷积层
        self.tsne_visualizer = TSNEVisualizer(class_names, self.use_english_labels)
    
    def visualize_feature_maps_by_blocks(self, image_tensor: torch.Tensor, image_path: str,
                                       save_dir: str) -> Dict[str, str]:
        """
        按Block结构可视化特征图

        Args:
            image_tensor: 输入图像张量
            image_path: 原始图像路径
            save_dir: 保存目录

        Returns:
            Dict[str, str]: Block名称到保存路径的映射
        """
        print(f"🎨 开始按Block结构生成特征图...")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        results = {}

        # 为每个Block生成特征图
        for block_name, layer_names in self.feature_extractor.layer_blocks.items():
            print(f"📊 生成 {block_name} 特征图...")

            try:
                save_path = self._visualize_single_block(
                    image_tensor, image_path, layer_names, block_name, save_dir
                )
                if save_path:
                    results[block_name] = save_path
                    print(f"  ✅ {block_name} 完成")
                else:
                    print(f"  ⚠️  {block_name} 生成失败")
            except Exception as e:
                print(f"  ❌ {block_name} 生成异常: {e}")

        # 生成概览图
        try:
            print("📊 生成所有Block概览图...")
            overview_path = self._visualize_blocks_overview(
                image_tensor, image_path, save_dir
            )
            if overview_path:
                results['overview'] = overview_path
                print("  ✅ 概览图完成")
        except Exception as e:
            print(f"  ❌ 概览图生成异常: {e}")

        return results

    def _visualize_single_block(self, image_tensor: torch.Tensor, image_path: str,
                               layer_names: List[str], block_name: str, save_dir: str) -> str:
        """
        可视化单个Block的特征图

        Args:
            image_tensor: 输入图像张量
            image_path: 原始图像路径
            layer_names: 该Block包含的层名称
            block_name: Block名称
            save_dir: 保存目录

        Returns:
            str: 保存的文件路径
        """
        # 提取该Block的特征
        features = self.feature_extractor.extract_features(image_tensor, layer_names)

        if not features:
            print(f"    ⚠️  {block_name} 未能提取到特征")
            return None

        valid_layers = list(features.keys())

        # 创建子图 - 每个Block的层数不多，使用合适的布局
        num_layers = len(valid_layers)
        num_channels = 8  # 每层显示8个通道

        fig, axes = plt.subplots(num_layers + 1, num_channels,
                                figsize=(20, 3 * (num_layers + 1)))
        
        # 显示原图
        try:
            original_image = Image.open(image_path).convert('RGB')
            axes[0, 0].imshow(original_image)
            if self.use_english_labels:
                axes[0, 0].set_title(f'Original Image\n{block_name}', fontsize=12, fontweight='bold')
            else:
                axes[0, 0].set_title(f'原始图像\n{block_name}', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
        except Exception as e:
            print(f"    ⚠️  加载原始图像失败: {e}")
            axes[0, 0].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            axes[0, 0].set_title(f'Original Image\n{block_name}', fontsize=12)
            axes[0, 0].axis('off')

        # 隐藏原图行的其他子图
        for i in range(1, num_channels):
            axes[0, i].axis('off')
        
        # 可视化每层的特征图
        for layer_idx, layer_name in enumerate(valid_layers):
            try:
                if layer_name not in features:
                    print(f"⚠️  {layer_name} 特征未找到，跳过")
                    continue

                feature_map = features[layer_name].squeeze().cpu().numpy()

                # 验证特征图维度
                if len(feature_map.shape) < 2:
                    print(f"⚠️  {layer_name} 特征图维度异常: {feature_map.shape}")
                    continue

                # 如果是3D特征图 [C, H, W]
                if len(feature_map.shape) == 3:
                    available_channels = min(num_channels, feature_map.shape[0])

                    for channel_idx in range(available_channels):
                        ax = axes[layer_idx + 1, channel_idx]

                        # 检查通道数据是否有效
                        channel_data = feature_map[channel_idx]
                        if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                            ax.text(0.5, 0.5, 'Invalid\nData', ha='center', va='center')
                            if self.use_english_labels:
                                ax.set_title(f'{layer_name}\nChannel {channel_idx}', fontsize=10)
                            else:
                                ax.set_title(f'{layer_name}\n通道 {channel_idx}', fontsize=10)
                            ax.axis('off')
                            continue

                        # 显示特征图
                        im = ax.imshow(channel_data, cmap='viridis')
                        if self.use_english_labels:
                            ax.set_title(f'{layer_name}\nChannel {channel_idx}', fontsize=10)
                        else:
                            ax.set_title(f'{layer_name}\n通道 {channel_idx}', fontsize=10)
                        ax.axis('off')

                        # 添加颜色条
                        try:
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        except:
                            pass  # 如果颜色条添加失败，继续执行

                    # 隐藏多余的子图
                    for channel_idx in range(available_channels, num_channels):
                        axes[layer_idx + 1, channel_idx].axis('off')

                # 如果是2D特征图 [H, W]
                elif len(feature_map.shape) == 2:
                    ax = axes[layer_idx + 1, 0]
                    im = ax.imshow(feature_map, cmap='viridis')
                    ax.set_title(f'{layer_name}', fontsize=10)
                    ax.axis('off')

                    try:
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    except:
                        pass

                    # 隐藏其他子图
                    for channel_idx in range(1, num_channels):
                        axes[layer_idx + 1, channel_idx].axis('off')

            except Exception as e:
                print(f"⚠️  可视化 {layer_name} 特征图时出错: {e}")
                # 填充错误信息
                for channel_idx in range(num_channels):
                    ax = axes[layer_idx + 1, channel_idx]
                    ax.text(0.5, 0.5, f'Error\n{layer_name}', ha='center', va='center')
                    ax.set_title(f'{layer_name}', fontsize=10)
                    ax.axis('off')
        
        plt.suptitle('VGG16 特征图可视化', fontsize=16, y=0.98)
        # 设置整体标题
        if self.use_english_labels:
            block_descriptions = {
                'Block1_EdgeTexture': 'Block 1: Edge & Texture Features',
                'Block2_ShapePattern': 'Block 2: Shape & Pattern Features',
                'Block3_LocalPattern': 'Block 3: Local Pattern Features',
                'Block4_ObjectPart': 'Block 4: Object Part Features',
                'Block5_Semantic': 'Block 5: Semantic Features'
            }
        else:
            block_descriptions = {
                'Block1_EdgeTexture': 'Block 1: 边缘纹理特征',
                'Block2_ShapePattern': 'Block 2: 形状模式特征',
                'Block3_LocalPattern': 'Block 3: 局部模式特征',
                'Block4_ObjectPart': 'Block 4: 对象部分特征',
                'Block5_Semantic': 'Block 5: 语义特征'
            }

        fig.suptitle(block_descriptions.get(block_name, block_name), fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图表
        filename = f"{block_name.lower()}_features"
        save_path = save_plot_with_timestamp(fig, save_dir, filename, show_plot=False)

        return save_path

    def _visualize_blocks_overview(self, image_tensor: torch.Tensor, image_path: str, save_dir: str) -> str:
        """
        生成所有Block的概览图

        Args:
            image_tensor: 输入图像张量
            image_path: 原始图像路径
            save_dir: 保存目录

        Returns:
            str: 保存的文件路径
        """
        # 选择每个Block的代表性层
        representative_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']

        # 提取代表性特征
        features = self.feature_extractor.extract_features(image_tensor, representative_layers)

        if not features:
            return None

        # 创建5x9的子图布局 (5个Block，每个Block显示8个通道+标题)
        fig, axes = plt.subplots(5, 9, figsize=(24, 15))

        # Block信息
        block_info = [
            ('conv1_2', 'Block1: Edge & Texture' if self.use_english_labels else 'Block1: 边缘纹理'),
            ('conv2_2', 'Block2: Shape & Pattern' if self.use_english_labels else 'Block2: 形状模式'),
            ('conv3_3', 'Block3: Local Pattern' if self.use_english_labels else 'Block3: 局部模式'),
            ('conv4_3', 'Block4: Object Part' if self.use_english_labels else 'Block4: 对象部分'),
            ('conv5_3', 'Block5: Semantic' if self.use_english_labels else 'Block5: 语义特征')
        ]

        # 显示原始图像（在第一行第一列）
        try:
            original_image = Image.open(image_path).convert('RGB')
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original' if self.use_english_labels else '原图', fontsize=10, fontweight='bold')
            axes[0, 0].axis('off')
        except:
            axes[0, 0].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
            axes[0, 0].axis('off')

        # 为每个Block生成概览
        for block_idx, (layer_name, block_title) in enumerate(block_info):
            if layer_name in features:
                feature_map = features[layer_name].squeeze().cpu().numpy()

                if len(feature_map.shape) == 3:  # [C, H, W]
                    # 显示前8个通道
                    num_show = min(8, feature_map.shape[0])

                    for ch_idx in range(num_show):
                        col_idx = ch_idx + 1 if block_idx == 0 else ch_idx
                        if col_idx < 9:
                            ax = axes[block_idx, col_idx]

                            channel_data = feature_map[ch_idx]
                            if not (np.isnan(channel_data).any() or np.isinf(channel_data).any()):
                                ax.imshow(channel_data, cmap='viridis')
                                ax.set_title(f'Ch{ch_idx}', fontsize=8)
                            else:
                                ax.text(0.5, 0.5, 'Invalid', ha='center', va='center')
                            ax.axis('off')

                    # 隐藏多余的子图
                    start_col = num_show + 1 if block_idx == 0 else num_show
                    for col_idx in range(start_col, 9):
                        axes[block_idx, col_idx].axis('off')
                else:
                    # 如果不是3D特征图，隐藏所有通道
                    start_col = 1 if block_idx == 0 else 0
                    for col_idx in range(start_col, 9):
                        axes[block_idx, col_idx].axis('off')
            else:
                # 如果没有该层特征，隐藏所有通道
                start_col = 1 if block_idx == 0 else 0
                for col_idx in range(start_col, 9):
                    axes[block_idx, col_idx].axis('off')

        # 设置整体标题
        if self.use_english_labels:
            fig.suptitle('VGG16 Feature Maps Overview - All Blocks', fontsize=18, fontweight='bold')
        else:
            fig.suptitle('VGG16 特征图概览 - 所有Block', fontsize=18, fontweight='bold')

        plt.tight_layout()

        # 保存概览图
        save_path = save_plot_with_timestamp(fig, save_dir, 'overview_all_blocks', show_plot=False)

        return save_path

    def visualize_grad_cam(self, image_tensor: torch.Tensor, image_path: str,
                          predicted_class: int, true_class: int, save_dir: str) -> str:
        """
        可视化Grad-CAM热力图

        Args:
            image_tensor: 输入图像张量
            image_path: 原始图像路径
            predicted_class: 预测类别索引
            true_class: 真实类别索引
            save_dir: 保存目录

        Returns:
            str: 保存的文件路径
        """
        # 生成CAM - 确保张量需要梯度
        try:
            # 设置张量需要梯度
            image_tensor = image_tensor.clone().detach().requires_grad_(True)

            cam_pred = self.grad_cam.generate_cam(image_tensor, predicted_class)
            cam_true = self.grad_cam.generate_cam(image_tensor, true_class)
        except Exception as e:
            print(f"⚠️  CAM生成失败: {e}")
            return None

        # 加载原图
        original_image = np.array(Image.open(image_path))

        # 创建热力图叠加
        def overlay_cam(image, cam, alpha=0.4):
            # 调整CAM尺寸到图像尺寸
            cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

            # 创建热力图
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 叠加到原图
            overlay = heatmap * alpha + image * (1 - alpha)
            return overlay.astype(np.uint8)

        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：预测类别的CAM
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('原始图像', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cam_pred, cmap='jet')
        axes[0, 1].set_title(f'预测类别 CAM\n{self.class_names[predicted_class]}', fontsize=12)
        axes[0, 1].axis('off')

        overlay_pred = overlay_cam(original_image, cam_pred)
        axes[0, 2].imshow(overlay_pred)
        axes[0, 2].set_title(f'预测类别叠加图\n{self.class_names[predicted_class]}', fontsize=12)
        axes[0, 2].axis('off')

        # 第二行：真实类别的CAM
        axes[1, 0].imshow(original_image)
        axes[1, 0].set_title('原始图像', fontsize=12)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cam_true, cmap='jet')
        axes[1, 1].set_title(f'真实类别 CAM\n{self.class_names[true_class]}', fontsize=12)
        axes[1, 1].axis('off')

        overlay_true = overlay_cam(original_image, cam_true)
        axes[1, 2].imshow(overlay_true)
        axes[1, 2].set_title(f'真实类别叠加图\n{self.class_names[true_class]}', fontsize=12)
        axes[1, 2].axis('off')

        plt.suptitle('Grad-CAM 类激活映射可视化', fontsize=16, y=0.95)
        plt.tight_layout()

        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_plot_with_timestamp(fig, save_dir, 'grad_cam', show_plot=False)

        return save_path

    def analyze_activation_patterns(self, image_tensors: List[torch.Tensor],
                                  labels: List[int], save_dir: str) -> str:
        """
        分析不同类别的激活模式

        Args:
            image_tensors: 图像张量列表
            labels: 对应的标签列表
            save_dir: 保存目录

        Returns:
            str: 保存的文件路径
        """
        layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # 存储每个类别每层的平均激活强度
        class_activations = {class_idx: {layer: [] for layer in layer_names}
                           for class_idx in range(len(self.class_names))}

        print("🔍 分析激活模式...")
        for i, (img_tensor, label) in enumerate(zip(image_tensors, labels)):
            try:
                # 确保张量维度正确
                if len(img_tensor.shape) == 3:  # [3, H, W]
                    input_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                elif len(img_tensor.shape) == 4:  # [1, 3, H, W]
                    input_tensor = img_tensor.squeeze(0).unsqueeze(0) if img_tensor.shape[0] == 1 else img_tensor
                else:
                    print(f"⚠️  样本 {i} 张量维度异常: {img_tensor.shape}")
                    continue

                features = self.feature_extractor.extract_features(input_tensor, layer_names)
            except Exception as e:
                print(f"⚠️  样本 {i} 特征提取失败: {e}")
                continue

            for layer_name in layer_names:
                if layer_name in features:
                    try:
                        # 计算该层的平均激活强度
                        activation = features[layer_name].mean().item()
                        class_activations[label][layer_name].append(activation)
                    except Exception as e:
                        print(f"⚠️  层 {layer_name} 激活计算失败: {e}")
                else:
                    print(f"⚠️  层 {layer_name} 特征未找到")

        # 计算每个类别每层的平均激活强度
        avg_activations = np.zeros((len(self.class_names), len(layer_names)))

        for class_idx in range(len(self.class_names)):
            for layer_idx, layer_name in enumerate(layer_names):
                if class_activations[class_idx][layer_name]:
                    avg_activations[class_idx, layer_idx] = np.mean(class_activations[class_idx][layer_name])

        # 创建热力图
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(avg_activations, cmap='viridis', aspect='auto')

        # 设置标签
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)

        # 添加数值标注
        for i in range(len(self.class_names)):
            for j in range(len(layer_names)):
                text = ax.text(j, i, f'{avg_activations[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=10)

        if self.use_english_labels:
            ax.set_title('Average Activation Intensity by Class and Layer', fontsize=14, pad=20)
            ax.set_xlabel('Convolutional Layers', fontsize=12)
            ax.set_ylabel('Classes', fontsize=12)
        else:
            ax.set_title('不同类别在各层的平均激活强度', fontsize=14, pad=20)
            ax.set_xlabel('卷积层', fontsize=12)
            ax.set_ylabel('类别', fontsize=12)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        if self.use_english_labels:
            cbar.set_label('Average Activation Intensity', fontsize=12)
        else:
            cbar.set_label('平均激活强度', fontsize=12)

        plt.tight_layout()

        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_plot_with_timestamp(fig, save_dir, 'activation_patterns', show_plot=False)

        return save_path

    def compare_error_features(self, correct_samples: List[Tuple[torch.Tensor, str, int]],
                             error_samples: List[Tuple[torch.Tensor, str, int, int]],
                             save_dir: str) -> str:
        """
        对比正确预测和错误预测样本的特征差异

        Args:
            correct_samples: 正确样本列表 [(tensor, path, true_label), ...]
            error_samples: 错误样本列表 [(tensor, path, true_label, pred_label), ...]
            save_dir: 保存目录

        Returns:
            str: 保存的文件路径
        """
        layer_names = ['conv3_1', 'conv4_1', 'conv5_1']  # 选择中高层特征

        # 分析正确样本的特征
        print("🔍 分析正确预测样本的特征...")
        correct_features = {layer: [] for layer in layer_names}

        for tensor, _, _ in correct_samples[:10]:  # 限制样本数量
            try:
                # 确保张量维度正确 [1, 3, H, W]
                if len(tensor.shape) == 3:  # [3, H, W]
                    input_tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
                elif len(tensor.shape) == 4:  # [1, 3, H, W]
                    input_tensor = tensor
                else:
                    print(f"⚠️  张量维度异常: {tensor.shape}")
                    continue

                features = self.feature_extractor.extract_features(input_tensor, layer_names)
                for layer_name in layer_names:
                    if layer_name in features:
                        # 计算特征图的统计信息
                        feature_map = features[layer_name].squeeze().cpu().numpy()

                        # 验证特征图数据有效性
                        if feature_map.size > 0 and not (np.isnan(feature_map).any() or np.isinf(feature_map).any()):
                            stats = {
                                'mean': np.mean(feature_map),
                                'std': np.std(feature_map),
                                'max': np.max(feature_map),
                                'sparsity': np.mean(feature_map == 0)  # 稀疏性
                            }
                            correct_features[layer_name].append(stats)
                        else:
                            print(f"⚠️  正确样本 {layer_name} 特征图数据无效")
                    else:
                        print(f"⚠️  正确样本未找到 {layer_name} 特征")
            except Exception as e:
                print(f"⚠️  处理正确样本特征时出错: {e}")

        # 分析错误样本的特征
        print("🔍 分析错误预测样本的特征...")
        error_features = {layer: [] for layer in layer_names}

        for tensor, _, _, _ in error_samples[:10]:  # 限制样本数量
            try:
                # 确保张量维度正确 [1, 3, H, W]
                if len(tensor.shape) == 3:  # [3, H, W]
                    input_tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
                elif len(tensor.shape) == 4:  # [1, 3, H, W]
                    input_tensor = tensor
                else:
                    print(f"⚠️  张量维度异常: {tensor.shape}")
                    continue

                features = self.feature_extractor.extract_features(input_tensor, layer_names)
                for layer_name in layer_names:
                    if layer_name in features:
                        feature_map = features[layer_name].squeeze().cpu().numpy()

                        # 验证特征图数据有效性
                        if feature_map.size > 0 and not (np.isnan(feature_map).any() or np.isinf(feature_map).any()):
                            stats = {
                                'mean': np.mean(feature_map),
                                'std': np.std(feature_map),
                                'max': np.max(feature_map),
                                'sparsity': np.mean(feature_map == 0)
                            }
                            error_features[layer_name].append(stats)
                        else:
                            print(f"⚠️  错误样本 {layer_name} 特征图数据无效")
                    else:
                        print(f"⚠️  错误样本未找到 {layer_name} 特征")
            except Exception as e:
                print(f"⚠️  处理错误样本特征时出错: {e}")

        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ['mean', 'std', 'max', 'sparsity']
        metric_names = ['平均值', '标准差', '最大值', '稀疏性']

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]

            # 计算每层的平均统计值
            correct_values = []
            error_values = []

            for layer_name in layer_names:
                if correct_features[layer_name]:
                    correct_avg = np.mean([stats[metric] for stats in correct_features[layer_name]])
                    correct_values.append(correct_avg)
                else:
                    correct_values.append(0)

                if error_features[layer_name]:
                    error_avg = np.mean([stats[metric] for stats in error_features[layer_name]])
                    error_values.append(error_avg)
                else:
                    error_values.append(0)

            # 绘制对比柱状图
            x = np.arange(len(layer_names))
            width = 0.35

            bars1 = ax.bar(x - width/2, correct_values, width, label='正确预测', alpha=0.8)
            bars2 = ax.bar(x + width/2, error_values, width, label='错误预测', alpha=0.8)

            ax.set_xlabel('卷积层')
            ax.set_ylabel(metric_name)
            ax.set_title(f'特征{metric_name}对比')
            ax.set_xticks(x)
            ax.set_xticklabels(layer_names)
            ax.legend()

            # 添加数值标注
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.suptitle('正确预测 vs 错误预测 特征对比分析', fontsize=16, y=0.95)
        plt.tight_layout()

        # 保存图表
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_plot_with_timestamp(fig, save_dir, 'error_feature_comparison', show_plot=False)

        return save_path

    def generate_comprehensive_feature_analysis(self, image_tensors: List[torch.Tensor],
                                              image_paths: List[str], true_labels: List[int],
                                              pred_labels: List[int], save_dir: str) -> Dict[str, str]:
        """
        生成综合特征分析报告

        Args:
            image_tensors: 图像张量列表
            image_paths: 图像路径列表
            true_labels: 真实标签列表
            pred_labels: 预测标签列表
            save_dir: 保存目录

        Returns:
            Dict[str, str]: 生成的文件路径字典
        """
        print("🎨 开始生成综合特征可视化分析...")
        print(f"📊 输入样本数量: {len(image_tensors)}")

        # 创建保存目录
        feature_dir = os.path.join(save_dir, 'feature_analysis')
        os.makedirs(feature_dir, exist_ok=True)

        results = {}

        # 检查输入数据
        if len(image_tensors) == 0:
            print("❌ 没有输入样本，跳过特征分析")
            return results

        # 1. 特征图可视化（选择几个代表性样本）
        print("📊 生成特征图可视化...")
        num_feature_samples = min(2, len(image_tensors))  # 减少样本数量
        sample_indices = np.random.choice(len(image_tensors), num_feature_samples, replace=False)
        print(f"🎯 选择 {num_feature_samples} 个样本进行特征图可视化: {sample_indices}")

        for i, idx in enumerate(sample_indices):
            try:
                print(f"🔍 处理特征图样本 {i+1}/{len(sample_indices)}: {os.path.basename(image_paths[idx])}")
                feature_map_dir = os.path.join(feature_dir, 'feature_maps')

                # 使用新的按Block结构的特征图可视化
                try:
                    block_results = self.visualize_feature_maps_by_blocks(
                        image_tensors[idx], image_paths[idx], feature_map_dir
                    )

                    if block_results:
                        # 将每个Block的结果添加到总结果中
                        for block_name, save_path in block_results.items():
                            results[f'sample_{i+1}_{block_name}'] = save_path
                        print(f"  ✅ 特征图生成成功: {len(block_results)} 个Block")
                    else:
                        print(f"  ⚠️  特征图生成失败")

                except Exception as e:
                    print(f"  ❌ 特征图生成异常: {e}")

                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️  样本 {i+1} 特征图生成异常: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 2. Grad-CAM可视化（选择正确和错误的样本）
        print("🔥 生成Grad-CAM可视化...")
        try:
            grad_cam_dir = os.path.join(feature_dir, 'grad_cam')

            # 找到正确和错误的样本
            correct_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == pred]
            error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true != pred]

            print(f"📊 正确样本: {len(correct_indices)} 个, 错误样本: {len(error_indices)} 个")

            # 可视化几个代表性样本
            if correct_indices:
                print("🎯 生成正确预测样本的Grad-CAM...")
                idx = correct_indices[0]
                try:
                    save_path = self.visualize_grad_cam(
                        image_tensors[idx], image_paths[idx], pred_labels[idx], true_labels[idx], grad_cam_dir
                    )
                    results['grad_cam_correct'] = save_path
                    print("  ✅ 正确样本Grad-CAM生成成功")
                except Exception as e:
                    print(f"  ⚠️  正确样本Grad-CAM生成失败: {e}")

            if error_indices:
                print("🎯 生成错误预测样本的Grad-CAM...")
                idx = error_indices[0]
                try:
                    save_path = self.visualize_grad_cam(
                        image_tensors[idx], image_paths[idx], pred_labels[idx], true_labels[idx], grad_cam_dir
                    )
                    results['grad_cam_error'] = save_path
                    print("  ✅ 错误样本Grad-CAM生成成功")
                except Exception as e:
                    print(f"  ⚠️  错误样本Grad-CAM生成失败: {e}")

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️  Grad-CAM可视化失败: {e}")

        # 3. 激活模式分析
        print("⚡ 分析激活模式...")
        try:
            activation_dir = os.path.join(feature_dir, 'activation_analysis')
            save_path = self.analyze_activation_patterns(image_tensors, true_labels, activation_dir)
            results['activation_patterns'] = save_path
            print("  ✅ 激活模式分析完成")
        except Exception as e:
            print(f"  ⚠️  激活模式分析失败: {e}")

        # 4. 错误样本特征对比
        if correct_indices and error_indices:
            print("🔍 对比错误样本特征...")
            try:
                comparison_dir = os.path.join(feature_dir, 'error_comparison')

                # 准备样本数据（减少样本数量）
                max_comparison_samples = min(3, len(correct_indices), len(error_indices))
                correct_samples = [(image_tensors[i], image_paths[i], true_labels[i])
                                 for i in correct_indices[:max_comparison_samples]]
                error_samples = [(image_tensors[i], image_paths[i], true_labels[i], pred_labels[i])
                               for i in error_indices[:max_comparison_samples]]

                print(f"📊 对比样本数量: 正确 {len(correct_samples)}, 错误 {len(error_samples)}")
                save_path = self.compare_error_features(correct_samples, error_samples, comparison_dir)
                results['error_feature_comparison'] = save_path
                print("  ✅ 错误样本特征对比完成")
            except Exception as e:
                print(f"  ⚠️  错误样本特征对比失败: {e}")
        else:
            print("⚠️  没有足够的正确或错误样本进行对比分析")

        # 5. t-SNE降维可视化
        print("🔬 生成t-SNE降维可视化...")
        try:
            tsne_dir = os.path.join(feature_dir, 'tsne_analysis')

            # 创建数据加载器用于t-SNE分析
            from torch.utils.data import TensorDataset, DataLoader

            # 准备数据
            tensor_list = []
            label_list = []
            for i, (tensor, label) in enumerate(zip(image_tensors, true_labels)):
                if i < 20:  # 限制样本数量以提高速度
                    tensor_list.append(tensor)
                    label_list.append(label)

            if tensor_list:
                batch_tensors = torch.stack(tensor_list)
                batch_labels = torch.tensor(label_list)

                temp_dataset = TensorDataset(batch_tensors, batch_labels)
                temp_dataloader = DataLoader(temp_dataset, batch_size=4, shuffle=False)

                # 对关键层进行t-SNE分析
                key_layers = ['conv1_2', 'conv3_2', 'conv5_3']  # 选择代表性层

                for layer_name in key_layers:
                    try:
                        print(f"  🔍 分析 {layer_name} 层...")
                        features, labels = self.tsne_visualizer.extract_features_for_tsne(
                            self.model, temp_dataloader, layer_name, self.device, max_samples=20
                        )

                        if features is not None and labels is not None:
                            save_path = self.tsne_visualizer.visualize_tsne(
                                features, labels, layer_name, tsne_dir
                            )
                            results[f'tsne_{layer_name}'] = save_path
                            print(f"    ✅ {layer_name} t-SNE完成")
                        else:
                            print(f"    ⚠️  {layer_name} 特征提取失败")

                    except Exception as e:
                        print(f"    ⚠️  {layer_name} t-SNE分析失败: {e}")

                print("  ✅ t-SNE降维可视化完成")
            else:
                print("  ⚠️  没有足够的样本进行t-SNE分析")

        except Exception as e:
            print(f"  ⚠️  t-SNE降维可视化失败: {e}")

        # 最终清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ 特征可视化分析完成！")
        print(f"📊 生成结果数量: {len(results)}")
        for key in results.keys():
            print(f"  • {key}")
        return results


# ==================== 独立运行支持函数 ====================

def load_model_for_analysis(model_path: str, config: dict, device='cpu'):
    """
    加载模型用于特征分析

    Args:
        model_path: 模型文件路径
        config: 配置字典
        device: 计算设备

    Returns:
        torch.nn.Module: 加载的模型
    """
    print(f"📥 加载模型: {model_path}")

    # 从配置文件获取模型信息
    model_config = config['model'].copy()
    model_config['num_classes'] = config['data']['num_classes']
    model_config['pretrained'] = False  # 分析时不需要预训练权重

    print(f"🤖 模型类型: {model_config['name']}")
    print(f"📊 分类数量: {model_config['num_classes']}")

    # 创建模型
    try:
        model = create_model(model_config)
        print(f"✅ 模型创建成功: {model.get_name() if hasattr(model, 'get_name') else model_config['name']}")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

    # 加载权重
    try:
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
        else:
            state_dict = torch.load(model_path)

        # 处理不同的保存格式
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"📋 加载检查点格式权重")
        else:
            model.load_state_dict(state_dict)
            print(f"📋 加载直接权重格式")

        model.to(device)
        model.eval()

        print(f"✅ 模型权重加载成功")
        return model

    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        return None


def create_analysis_dataloader(config: dict, project_root: str, max_samples=50):
    """
    创建用于特征分析的数据加载器

    Args:
        config: 配置字典
        project_root: 项目根目录
        max_samples: 最大样本数量

    Returns:
        DataLoader: 数据加载器
    """
    # 标注文件路径
    annotation_path = os.path.join(project_root, config['data']['annotation_file'])

    if not os.path.exists(annotation_path):
        print(f"❌ 标注文件不存在: {annotation_path}")
        return None

    # 读取标注文件
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) == 0:
        print(f"❌ 标注文件为空: {annotation_path}")
        return None

    # 限制样本数量以提高分析速度
    if len(lines) > max_samples:
        print(f"📊 限制样本数量: {len(lines)} -> {max_samples}")
        lines = lines[:max_samples]

    print(f"📊 加载标注文件: {annotation_path}")
    print(f"   分析样本数: {len(lines)}")

    # 创建数据集（不使用数据增强）
    input_shape = config['data']['input_shape']
    dataset = DataGenerator(lines, input_shape, random=False)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 小批次以节省内存
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return dataloader


def main(train_path: str):
    """
    主函数 - 独立特征可视化分析

    Args:
        train_path: 训练路径，例如 "/var/yjs/zes/vgg/outputs/logs/train_20250826_041814"
    """
    print("🎨" + "="*80)
    print("🎨 独立特征可视化分析工具")
    print("🎨 专门用于混凝土坍落度识别的深度特征分析")
    print("🎨" + "="*80)

    # ==================== 字体设置 ====================
    print("🔤 设置字体...")
    setup_matplotlib_font()
    font_manager = get_font_manager()
    font_manager.print_font_status()

    # 如果没有中文字体，显示安装指南
    if not font_manager.has_chinese_font:
        from src.utils.font_manager import print_font_installation_guide
        print_font_installation_guide()

    # ==================== 路径处理 ====================
    print(f"📁 输入训练路径: {train_path}")

    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # 提取时间戳
    timestamp = extract_timestamp_from_path(train_path)
    if not timestamp:
        print("❌ 无法从路径中提取时间戳")
        return

    print(f"⏰ 提取时间戳: {timestamp}")

    # ==================== 自动寻找文件 ====================
    # 查找模型文件
    model_path = find_model_file(project_root, timestamp)
    if not model_path:
        print("❌ 未找到对应的模型文件")
        return

    # 查找配置文件
    config, config_source = load_training_config(train_path, project_root)
    if config is None:
        print("❌ 无法加载配置文件")
        return

    # 生成输出目录
    eval_timestamp = generate_eval_timestamp(timestamp)
    output_dir = os.path.join(project_root, 'outputs', 'evaluation', eval_timestamp, 'feature_analysis')

    print(f"🤖 模型文件: {model_path}")
    print(f"📄 配置来源: {config_source}")
    print(f"💾 输出目录: {output_dir}")

    # ==================== 设备检查 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")

    # ==================== 加载模型 ====================
    model = load_model_for_analysis(model_path, config, device)
    if model is None:
        return

    # ==================== 创建数据加载器 ====================
    dataloader = create_analysis_dataloader(config, project_root, max_samples=30)
    if dataloader is None:
        return

    # ==================== 初始化可视化器 ====================
    class_names = config['data']['class_names']
    visualizer = FeatureVisualizer(model, class_names, device)

    print(f"🎯 类别名称: {class_names}")
    print(f"📊 开始特征可视化分析...")

    # ==================== 执行分析 ====================
    try:
        # 收集样本数据
        sample_tensors = []
        sample_paths = []
        sample_labels = []

        print("📊 收集样本数据...")
        with torch.no_grad():
            for batch_idx, (images, labels, paths) in enumerate(dataloader):
                sample_tensors.extend([img.unsqueeze(0) for img in images])
                sample_paths.extend(paths)
                sample_labels.extend(labels.cpu().numpy())

                if len(sample_tensors) >= 10:  # 限制样本数量
                    break

        print(f"✅ 收集完成: {len(sample_tensors)} 个样本")

        # 获取真实的预测标签
        print("🔮 执行模型推理获取预测标签...")
        pred_labels = []
        model.eval()
        with torch.no_grad():
            for i, tensor in enumerate(sample_tensors):
                try:
                    tensor = tensor.to(device)
                    outputs = model(tensor)
                    _, predicted = torch.max(outputs, 1)
                    pred_labels.append(predicted.cpu().item())
                    print(f"  样本 {i+1}: 真实={sample_labels[i]}, 预测={predicted.cpu().item()}")
                except Exception as e:
                    print(f"  ⚠️  样本 {i+1} 推理失败: {e}")
                    pred_labels.append(sample_labels[i])  # 使用真实标签作为备选

        print(f"✅ 推理完成: {len(pred_labels)} 个预测结果")

        # 执行综合特征分析
        results = visualizer.generate_comprehensive_feature_analysis(
            image_tensors=sample_tensors,
            image_paths=sample_paths,
            true_labels=sample_labels,
            pred_labels=pred_labels,  # 使用真实的预测标签
            save_dir=output_dir
        )

        print("\n🎉 特征可视化分析完成！")
        print(f"📁 结果保存在: {output_dir}")
        print("📊 生成的分析文件:")
        for key, path in results.items():
            if path:
                print(f"   • {key}: {os.path.basename(path)}")

    except Exception as e:
        print(f"❌ 特征分析失败: {e}")
        import traceback
        traceback.print_exc()


def test_font_support():
    """测试字体支持情况"""
    print("🔤 测试字体支持...")

    # 导入字体管理器
    from src.utils.font_manager import setup_matplotlib_font, get_font_manager, print_font_installation_guide
    import matplotlib.font_manager as fm

    # 设置字体
    setup_matplotlib_font()
    font_manager = get_font_manager()

    # 显示字体状态
    print(f"操作系统: {font_manager.system}")
    print(f"当前字体: {font_manager.current_font}")
    print(f"中文支持: {'是' if font_manager.has_chinese_font else '否'}")

    # 显示可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']

    print("\n🔍 检查中文字体:")
    for font in chinese_fonts:
        status = "✅" if font in available_fonts else "❌"
        print(f"  {status} {font}")

    if not font_manager.has_chinese_font:
        print("\n💡 字体安装建议:")
        print_font_installation_guide()

    return font_manager.has_chinese_font


if __name__ == "__main__":
    """
    独立运行配置

    使用方法:
    1. 修改下面的 train_path 为你的训练路径
    2. 在 PyCharm 中直接运行此脚本
    """

    # 首先测试字体支持
    print("="*60)
    has_chinese = test_font_support()
    print("="*60)

    if not has_chinese:
        print("⚠️  检测到中文字体问题，建议先安装中文字体")
        print("   继续运行将使用英文标题...")
        input("按回车键继续...")

    # 🎯 在这里修改你的训练路径
    train_path = "/var/yjs/zes/vgg/outputs/logs/train_20250826_041814"

    # 执行分析
    main(train_path)
