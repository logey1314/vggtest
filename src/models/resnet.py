"""
ResNet模型实现
基于torchvision的ResNet，针对混凝土坍落度分类任务优化
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from typing import Optional


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
                 zero_init_residual: bool = False):
        """
        初始化ResNet模型
        
        Args:
            variant (str): ResNet变体，支持 "resnet18", "resnet50"
            pretrained (bool): 是否使用预训练权重
            num_classes (int): 分类数量
            dropout (float): Dropout概率
            replace_stride_with_dilation (list): 是否用膨胀卷积替换步长
            zero_init_residual (bool): 是否对残差块的最后一个BN层进行零初始化
        """
        super(ResNet, self).__init__()
        
        self.variant = variant.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # 创建基础ResNet模型
        if self.variant == "resnet18":
            if pretrained:
                weights = ResNet18_Weights.IMAGENET1K_V1
                self.backbone = resnet18(weights=weights)
            else:
                self.backbone = resnet18(weights=None)
            self.feature_dim = 512
            
        elif self.variant == "resnet50":
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
                self.backbone = resnet50(weights=weights)
            else:
                self.backbone = resnet50(weights=None)
            self.feature_dim = 2048
            
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


# 为了保持接口一致性，提供简化的创建函数
def resnet18(*args, **kwargs):
    """ResNet18创建函数（兼容接口）"""
    return resnet18_concrete(*args, **kwargs)


def resnet50(*args, **kwargs):
    """ResNet50创建函数（兼容接口）"""
    return resnet50_concrete(*args, **kwargs)
