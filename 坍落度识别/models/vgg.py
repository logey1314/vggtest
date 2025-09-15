import torch
import torch.nn as nn
import os
import urllib.request
from torch.hub import load_state_dict_from_url


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

}#权重下载网址


def auto_download_vgg_weights(target_path, project_root=None):
    """
    自动下载VGG16预训练权重到指定路径
    
    Args:
        target_path (str): 目标保存路径（相对于项目根目录）
        project_root (str): 项目根目录路径
    
    Returns:
        bool: 下载是否成功
    """
    url = model_urls['vgg16']
    
    # 处理路径：如果没有提供project_root，尝试自动推断
    if project_root is None:
        # 从当前文件路径推断项目根目录
        current_file = os.path.abspath(__file__)  # src/models/vgg.py
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # 上三级目录
    
    # 构建绝对路径
    if not os.path.isabs(target_path):
        absolute_target_path = os.path.join(project_root, target_path)
    else:
        absolute_target_path = target_path
    
    try:
        # 创建目标目录
        os.makedirs(os.path.dirname(absolute_target_path), exist_ok=True)
        
        print(f"🌐 正在下载 VGG16 权重到项目目录...")
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


class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weights= True, dropout = 0.5):
        super(VGG,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))#AdaptiveAvgPool2d使处于不同大小的图片也能进行分类
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),#完成4096的全连接
            nn.Linear(4096, num_classes),#对num_classes的分类
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)#对输入层进行平铺，转化为一维数据
        x = self.classifier(x)
        return x

    def get_name(self):
        """获取模型名称"""
        return "VGG16"

    def get_info(self):
        """获取模型详细信息"""
        return {
            'name': 'VGG16',
            'num_classes': self.classifier[-1].out_features
        }

    def get_parameter_count(self):
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }


def make_layers(cfg, batch_norm = False):#make_layers对输入的cfg进行循环
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

}

def vgg16(pretrained=False, progress=True, num_classes=1000, model_dir='./models/pretrained', dropout=0.5, pretrained_weights=None):
    """
    VGG16 模型

    Args:
        pretrained (bool): 是否使用预训练权重
        progress (bool): 是否显示下载进度
        num_classes (int): 分类数量
        model_dir (str): 预训练模型保存目录
        dropout (float): Dropout比例
        pretrained_weights (str): 自定义预训练权重文件路径

    Returns:
        VGG: VGG16 模型实例
    """
    model = VGG(make_layers(cfgs['D']), dropout=dropout)
    if pretrained:
        if pretrained_weights:
            # 权重文件路径已指定，处理相对路径
            current_file = os.path.abspath(__file__)  # src/models/vgg.py
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            if not os.path.isabs(pretrained_weights):
                absolute_weights_path = os.path.join(project_root, pretrained_weights)
            else:
                absolute_weights_path = pretrained_weights
            
            if os.path.exists(absolute_weights_path):
                # 使用现有权重文件
                print(f"   📁 加载自定义VGG16权重: {absolute_weights_path}")
                state_dict = torch.load(absolute_weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
            else:
                # 权重文件不存在，自动下载到指定路径
                print(f"   🔍 权重文件不存在，准备自动下载...")
                if auto_download_vgg_weights(pretrained_weights, project_root):
                    print(f"   📁 加载下载的VGG16权重: {absolute_weights_path}")
                    state_dict = torch.load(absolute_weights_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                else:
                    print(f"   ⚠️  自动下载失败，使用默认下载方式")
                    state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir=model_dir, progress=progress)
                    model.load_state_dict(state_dict)
        else:
            # 使用默认预训练权重（缓存到指定目录）
            print(f"   🌐 使用默认预训练权重（缓存到指定目录）")
            state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir=model_dir, progress=progress)
            model.load_state_dict(state_dict)
    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),  # 使用传入的dropout参数
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),  # 使用传入的dropout参数
            nn.Linear(4096, num_classes),
        )
    return model

if __name__=='__main__':
    in_data=torch.ones(1,3,224,224)
    net=vgg16(pretrained=False, progress=True, num_classes=3)  # 示例：3分类
    out=net(in_data)
    print(out)
