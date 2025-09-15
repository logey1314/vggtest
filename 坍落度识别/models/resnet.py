"""
ResNet模型实现
基于torchvision的ResNet，针对混凝土坍落度分类任务优化
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from typing import Optional
import os
import urllib.request
from pathlib import Path


def auto_download_weights(model_name, target_path, project_root=None):
    """
    自动下载预训练权重到指定路径
    
    Args:
        model_name (str): 模型名称 (resnet18, resnet50)
        target_path (str): 目标保存路径（相对于项目根目录）
        project_root (str): 项目根目录路径
    
    Returns:
        bool: 下载是否成功
    """
    # 权重文件URL映射
    weight_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    }
    
    if model_name not in weight_urls:
        print(f"❌ 不支持的模型: {model_name}")
        return False
    
    url = weight_urls[model_name]
    
    # 处理路径：如果没有提供project_root，尝试自动推断
    if project_root is None:
        # 从当前文件路径推断项目根目录
        current_file = os.path.abspath(__file__)  # src/models/resnet.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级目录
    
    # 构建绝对路径
    if not os.path.isabs(target_path):
        absolute_target_path = os.path.join(project_root, target_path)
    else:
        absolute_target_path = target_path
    
    try:
        # 创建目标目录
        os.makedirs(os.path.dirname(absolute_target_path), exist_ok=True)
        
        print(f"🌐 正在下载 {model_name} 权重到项目目录...")
        print(f"   源地址: {url}")
        print(f"   目标路径: {absolute_target_path}")
        
        # 下载权重文件
        urllib.request.urlretrieve(url, absolute_target_path)
        
        # 验证文件下载成功
        if os.path.exists(absolute_target_path) and os.path.getsize(absolute_target_path) > 0:
            file_size = os.path.getsize(absolute_target_path) / 1024 / 1024  # MB
            print(f"✅ 下载成功! 文件大小: {file_size:.1f}MB")
            return True
        else:
            print(f"❌ 下载失败: 文件大小为0")
            return False
            
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


class ResNet(nn.Module):
    """
    ResNet模型包装器
    支持ResNet18和ResNet50架构
    """
    
    def __init__(self,
                 variant: str = "resnet50",
                 pretrained: bool = True,
                 num_classes: int = 3,
                 dropout: float = 0.5,
                 replace_stride_with_dilation: Optional[list] = None,
                 zero_init_residual: bool = False,
                 pretrained_weights: Optional[str] = None):
        """
        初始化ResNet模型

        Args:
            variant (str): ResNet变体，支持 "resnet18", "resnet50"
            pretrained (bool): 是否使用预训练权重
            num_classes (int): 分类数量
            dropout (float): Dropout概率
            replace_stride_with_dilation (list): 是否用膨胀卷积替换步长
            zero_init_residual (bool): 是否对残差块的最后一个BN层进行零初始化
            pretrained_weights (str): 自定义预训练权重文件路径
        """
        super(ResNet, self).__init__()
        
        self.variant = variant.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # 创建基础ResNet模型
        if self.variant == "resnet18":
            self.feature_dim = 512
            if pretrained and pretrained_weights:
                # 权重文件路径已指定，处理相对路径
                current_file = os.path.abspath(__file__)  # src/models/resnet.py
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                if not os.path.isabs(pretrained_weights):
                    absolute_weights_path = os.path.join(project_root, pretrained_weights)
                else:
                    absolute_weights_path = pretrained_weights
                
                if os.path.exists(absolute_weights_path):
                    # 使用现有权重文件
                    print(f"   📁 加载自定义ResNet18权重: {absolute_weights_path}")
                    from torchvision.models import resnet18 as torch_resnet18
                    self.backbone = torch_resnet18(weights=None)
                    self.backbone.load_state_dict(torch.load(absolute_weights_path, map_location='cpu'))
                else:
                    # 权重文件不存在，自动下载到指定路径
                    print(f"   🔍 权重文件不存在，准备自动下载...")
                    if auto_download_weights('resnet18', pretrained_weights, project_root):
                        print(f"   📁 加载下载的ResNet18权重: {absolute_weights_path}")
                        from torchvision.models import resnet18 as torch_resnet18
                        self.backbone = torch_resnet18(weights=None)
                        self.backbone.load_state_dict(torch.load(absolute_weights_path, map_location='cpu'))
                    else:
                        print(f"   ⚠️  自动下载失败，使用PyTorch默认权重")
                        weights = ResNet18_Weights.IMAGENET1K_V1
                        from torchvision.models import resnet18 as torch_resnet18
                        self.backbone = torch_resnet18(weights=weights)
            elif pretrained:
                # 使用PyTorch默认预训练权重（缓存方式）
                print(f"   🌐 使用PyTorch默认预训练权重（缓存到系统目录）")
                weights = ResNet18_Weights.IMAGENET1K_V1
                from torchvision.models import resnet18 as torch_resnet18
                self.backbone = torch_resnet18(weights=weights)
            else:
                # 不使用预训练权重
                from torchvision.models import resnet18 as torch_resnet18
                self.backbone = torch_resnet18(weights=None)

        elif self.variant == "resnet50":
            self.feature_dim = 2048
            if pretrained and pretrained_weights:
                # 权重文件路径已指定，处理相对路径
                current_file = os.path.abspath(__file__)  # src/models/resnet.py
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                if not os.path.isabs(pretrained_weights):
                    absolute_weights_path = os.path.join(project_root, pretrained_weights)
                else:
                    absolute_weights_path = pretrained_weights
                
                if os.path.exists(absolute_weights_path):
                    # 使用现有权重文件
                    print(f"   📁 加载自定义ResNet50权重: {absolute_weights_path}")
                    from torchvision.models import resnet50 as torch_resnet50
                    self.backbone = torch_resnet50(weights=None)
                    self.backbone.load_state_dict(torch.load(absolute_weights_path, map_location='cpu'))
                else:
                    # 权重文件不存在，自动下载到指定路径
                    print(f"   🔍 权重文件不存在，准备自动下载...")
                    if auto_download_weights('resnet50', pretrained_weights, project_root):
                        print(f"   📁 加载下载的ResNet50权重: {absolute_weights_path}")
                        from torchvision.models import resnet50 as torch_resnet50
                        self.backbone = torch_resnet50(weights=None)
                        self.backbone.load_state_dict(torch.load(absolute_weights_path, map_location='cpu'))
                    else:
                        print(f"   ⚠️  自动下载失败，使用PyTorch默认权重")
                        weights = ResNet50_Weights.IMAGENET1K_V2
                        from torchvision.models import resnet50 as torch_resnet50
                        self.backbone = torch_resnet50(weights=weights)
            elif pretrained:
                # 使用PyTorch默认预训练权重（缓存方式）
                print(f"   🌐 使用PyTorch默认预训练权重（缓存到系统目录）")
                weights = ResNet50_Weights.IMAGENET1K_V2
                from torchvision.models import resnet50 as torch_resnet50
                self.backbone = torch_resnet50(weights=weights)
            else:
                # 不使用预训练权重
                from torchvision.models import resnet50 as torch_resnet50
                self.backbone = torch_resnet50(weights=None)

        else:
            raise ValueError(f"不支持的ResNet变体: {variant}. 支持的变体: resnet18, resnet50")
        
        # 应用高级配置
        if replace_stride_with_dilation:
            self._apply_dilation_config(replace_stride_with_dilation)
        
        if zero_init_residual:
            self._apply_zero_init_residual()
        
        # 替换分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # 记录模型信息
        self.model_info = {
            'name': f'ResNet-{variant.upper()}',
            'variant': variant,
            'pretrained': pretrained,
            'num_classes': num_classes,
            'dropout': dropout,
            'feature_dim': self.feature_dim,
            'replace_stride_with_dilation': replace_stride_with_dilation,
            'zero_init_residual': zero_init_residual
        }
    
    def _apply_dilation_config(self, replace_stride_with_dilation):
        """应用膨胀卷积配置"""
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation必须是长度为3的列表")
        
        # 这里需要重新创建模型以应用膨胀配置
        # 由于torchvision的限制，我们需要在创建时就指定这个参数
        print(f"⚠️  膨胀卷积配置将在下个版本中支持: {replace_stride_with_dilation}")
    
    def _apply_zero_init_residual(self):
        """对残差块的最后一个BN层进行零初始化"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                # 找到残差块中的最后一个BN层并进行零初始化
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)
    
    def get_name(self):
        """获取模型名称"""
        return self.model_info['name']
    
    def get_info(self):
        """获取模型详细信息"""
        return self.model_info.copy()
    
    def get_feature_dim(self):
        """获取特征维度"""
        return self.feature_dim
    
    def get_parameter_count(self):
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }


def resnet18_concrete(pretrained=True, num_classes=3, dropout=0.5, **kwargs):
    """
    创建用于混凝土坍落度分类的ResNet18模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        num_classes (int): 分类数量
        dropout (float): Dropout概率
        **kwargs: 其他参数
        
    Returns:
        ResNet: ResNet18模型实例
    """
    return ResNet(
        variant="resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    )


def resnet50_concrete(pretrained=True, num_classes=3, dropout=0.5, **kwargs):
    """
    创建用于混凝土坍落度分类的ResNet50模型
    
    Args:
        pretrained (bool): 是否使用预训练权重
        num_classes (int): 分类数量
        dropout (float): Dropout概率
        **kwargs: 其他参数
        
    Returns:
        ResNet: ResNet50模型实例
    """
    return ResNet(
        variant="resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs
    )



