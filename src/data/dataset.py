"""
数据集处理模块
包含数据加载、预处理和数据增强功能
"""
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image


def preprocess_input(x):
    """
    输入数据预处理
    将像素值从 [0, 255] 范围转换到 [-1, 1] 范围
    """
    x /= 127.5
    x -= 1.
    return x


def cvtColor(image):
    """
    确保图像为RGB格式
    """
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


class DataGenerator(data.Dataset):
    """
    自定义数据集类
    支持数据增强和图像预处理
    """

    def __init__(self, annotation_lines, input_shape, random=True, augmentation_config=None):
        """
        初始化数据集

        Args:
            annotation_lines (list): 标注文件行列表
            input_shape (list): 输入图像尺寸 [height, width]
            random (bool): 是否启用随机数据增强
            augmentation_config (dict): 数据增强配置参数
        """
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.random = random

        # 默认数据增强配置
        self.default_aug_config = {
            'enable_flip': True,           # 是否启用水平翻转
            'enable_rotation': True,       # 是否启用旋转
            'enable_color_jitter': True,   # 是否启用色彩抖动
            'enable_scale_jitter': True,   # 是否启用尺度抖动

            'jitter': 0.3,                # 长宽比抖动范围
            'scale_range': (0.75, 1.25),  # 缩放范围
            'rotation_range': 15,          # 旋转角度范围 (-15, 15)
            'hue_range': 0.1,             # 色调变化范围
            'saturation_range': 1.5,       # 饱和度变化范围
            'value_range': 1.5,           # 明度变化范围
            'flip_probability': 0.5,       # 翻转概率
            'rotation_probability': 0.5,   # 旋转概率
        }

        # 合并用户配置
        if augmentation_config is not None:
            self.aug_config = {**self.default_aug_config, **augmentation_config}
        else:
            self.aug_config = self.default_aug_config

    def __len__(self):
        return len(self.annotation_lines)
    
    def __getitem__(self, index):
        """
        获取单个数据样本

        Args:
            index (int): 样本索引

        Returns:
            tuple: (image, label, image_path) 图像、标签和图像路径
        """
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)
        image = self.get_random_data(image, self.input_shape, random=self.random)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
        y = int(self.annotation_lines[index].split(';')[0])
        return image, y, annotation_path
    
    def rand(self, a=0, b=1):
        """生成随机数"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, random=True):
        """
        数据增强处理

        Args:
            image: PIL图像
            input_shape: 目标尺寸
            random: 是否启用随机增强

        Returns:
            numpy.ndarray: 处理后的图像数据
        """
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        
        if not random:
            # 验证模式：只进行缩放和填充
            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data
        
        # 训练模式：随机数据增强
        if self.aug_config['enable_scale_jitter']:
            # 尺度和长宽比抖动
            jitter = self.aug_config['jitter']
            new_ar = w/h * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
            scale_min, scale_max = self.aug_config['scale_range']
            scale = self.rand(scale_min, scale_max)
        else:
            # 不使用抖动，保持原始比例
            new_ar = w/h
            scale = min(w/iw, h/ih)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw)) if self.aug_config['enable_scale_jitter'] else (w-nw)//2
        dy = int(self.rand(0, h-nh)) if self.aug_config['enable_scale_jitter'] else (h-nh)//2
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 水平翻转
        if self.aug_config['enable_flip']:
            flip = self.rand() < self.aug_config['flip_probability']
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 旋转
        if self.aug_config['enable_rotation']:
            rotate = self.rand() < self.aug_config['rotation_probability']
            if rotate:
                rotation_range = self.aug_config['rotation_range']
                angle = np.random.randint(-rotation_range, rotation_range)
                a, b = w/2, h/2
                M = cv2.getRotationMatrix2D((a, b), angle, 1)
                image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        # 色彩抖动
        if self.aug_config['enable_color_jitter']:
            hue_range = self.aug_config['hue_range']
            sat_range = self.aug_config['saturation_range']
            val_range = self.aug_config['value_range']

            hue = self.rand(-hue_range, hue_range)
            sat = self.rand(1, sat_range) if self.rand() < .5 else 1/self.rand(1, sat_range)
            val = self.rand(1, val_range) if self.rand() < .5 else 1/self.rand(1, val_range)

            x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)  # 颜色空间转换
            x[..., 0] += hue * 360  # 色调调整
            x[..., 1] *= sat        # 饱和度调整
            x[..., 2] *= val        # 明度调整

            # 确保值在有效范围内
            x[x[:, :, 0] > 360, 0] = 360
            x[x[:, :, 0] < 0, 0] = 0
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0

            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        else:
            image_data = np.array(image, np.float32)

        return image_data
