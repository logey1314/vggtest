import torch
import torch.nn as nn
import os
from torch.hub import load_state_dict_from_url


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

}#权重下载网址


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
        if pretrained_weights and os.path.exists(pretrained_weights):
            # 使用自定义权重文件
            print(f"   📁 加载自定义VGG16权重: {pretrained_weights}")
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            # 使用默认预训练权重
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
