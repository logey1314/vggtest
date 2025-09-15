#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坍落度识别系统安装脚本
自动安装所需依赖包
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ 错误: 需要Python 3.7或更高版本")
        print(f"   当前版本: Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python版本检查通过: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """安装依赖包"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ 错误: 找不到requirements.txt文件")
        return False
    
    print("📦 正在安装依赖包...")
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # 安装依赖
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def check_torch_installation():
    """检查PyTorch安装和GPU支持"""
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU支持: 检测到 {gpu_count} 个GPU")
            print(f"   主GPU: {gpu_name}")
        else:
            print("⚠️  GPU支持: 未检测到CUDA GPU，将使用CPU")
        
        return True
        
    except ImportError:
        print("❌ PyTorch未正确安装")
        return False

def create_directories():
    """创建必要的目录"""
    dirs = [
        "logs",
        "config",
        "output",
        "temp"
    ]
    
    for dir_name in dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

def copy_default_config():
    """复制默认配置文件"""
    config_dir = Path(__file__).parent / "config"
    default_config = config_dir / "default_config.json"
    user_config = config_dir / "recognition_config.json"
    
    if default_config.exists() and not user_config.exists():
        import shutil
        shutil.copy(default_config, user_config)
        print("📄 创建用户配置文件")

def test_installation():
    """测试安装"""
    print("\n🧪 测试安装...")
    
    try:
        # 测试主要模块导入
        import torch
        import torchvision
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import PIL
        
        print("✅ 所有核心模块导入成功")
        
        # 测试模型创建
        sys.path.append(str(Path(__file__).parent))
        from models import get_available_models
        
        models = get_available_models()
        print(f"✅ 模型工厂测试成功，可用模型: {list(models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 安装测试失败: {e}")
        return False

def main():
    """主安装流程"""
    print("🚀 坍落度识别系统安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 创建目录
    create_directories()
    
    # 安装依赖
    if not install_requirements():
        return False
    
    # 检查PyTorch
    if not check_torch_installation():
        return False
    
    # 复制配置文件
    copy_default_config()
    
    # 测试安装
    if not test_installation():
        return False
    
    print("\n✅ 安装完成！")
    print("\n🎯 使用方法:")
    print("   1. 启动GUI: python slump_recognition_gui.py")
    print("   2. 或使用启动脚本: 双击 '启动程序.bat'")
    print("\n📚 说明文档:")
    print("   - README.md: 详细使用说明")
    print("   - config/: 配置文件目录")
    print("   - logs/: 日志文件目录")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\n按回车键退出...")
        sys.exit(1)
    else:
        input("\n按回车键退出...")
