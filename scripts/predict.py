"""
VGG16 图像分类预测脚本
支持单张图像和批量预测
"""
import os
import sys
import argparse
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vgg import vgg16

def predict_image(image_path, model_path="models/checkpoints/best_model.pth", show_plot=True):
    """
    对单张图像进行预测
    
    Args:
        image_path (str): 图像文件路径
        model_path (str): 模型权重文件路径
        show_plot (bool): 是否显示预测结果图
        
    Returns:
        dict: 预测结果字典
    """
    # 类别名称 (根据您的数据集调整)
    class_names = ['Class1 (125-175)', 'Class2 (180-230)', 'Class3 (233-285)']

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
        return None

    try:
        # 加载图像
        print(f"📷 加载图像: {image_path}")
        image = Image.open(image_path)
        original_image = image.copy()

        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        input_tensor = transform(image).unsqueeze(0)  # 添加batch维度

        # 加载模型
        print(f"🤖 加载模型: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = vgg16(pretrained=False, num_classes=3)  # 预测时dropout不影响结果

        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            net.load_state_dict(checkpoint)
        
        net.to(device)
        net.eval()

        # 预测
        print("🔍 进行预测...")
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = net(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 获取所有类别的概率
        probs = probabilities.cpu().numpy()[0]
        predicted_class = predicted.item()
        confidence_score = confidence.item()

        # 打印结果
        print(f"\n🎯 预测结果:")
        print(f"   预测类别: {class_names[predicted_class]}")
        print(f"   置信度: {confidence_score:.4f}")
        print(f"\n📊 各类别概率:")
        for i, (class_name, prob) in enumerate(zip(class_names, probs)):
            print(f"   {class_name}: {prob:.4f}")

        # 可视化结果
        if show_plot:
            plt.figure(figsize=(12, 5))
            
            # 显示原图
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(f'原始图像\n{os.path.basename(image_path)}', fontsize=12)
            plt.axis('off')
            
            # 显示预测概率
            plt.subplot(1, 2, 2)
            colors = ['red' if i == predicted_class else 'skyblue' for i in range(len(class_names))]
            bars = plt.bar(range(len(class_names)), probs, color=colors)
            plt.title(f'预测结果\n{class_names[predicted_class]} (置信度: {confidence_score:.3f})', fontsize=12)
            plt.xlabel('类别')
            plt.ylabel('概率')
            plt.xticks(range(len(class_names)), [name.split()[0] for name in class_names], rotation=45)
            
            # 添加数值标签
            for bar, prob in zip(bars, probs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{prob:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存结果图
            output_dir = "outputs/predictions"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            output_path = os.path.join(output_dir, f"prediction_{os.path.splitext(os.path.basename(image_path))[0]}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📁 预测结果已保存: {output_path}")
            
            plt.show()

        return {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence_score,
            'probabilities': probs.tolist(),
            'class_names': class_names
        }

    except Exception as e:
        print(f"❌ 预测过程中出现错误: {str(e)}")
        return None


def predict_batch(image_dir, model_path="models/checkpoints/best_model.pth"):
    """
    批量预测文件夹中的所有图像
    
    Args:
        image_dir (str): 图像文件夹路径
        model_path (str): 模型权重文件路径
    """
    if not os.path.exists(image_dir):
        print(f"❌ 图像文件夹不存在: {image_dir}")
        return

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"❌ 在文件夹 {image_dir} 中未找到图像文件")
        return

    print(f"📁 找到 {len(image_files)} 张图像，开始批量预测...")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
        result = predict_image(image_path, model_path, show_plot=False)
        if result:
            results.append({
                'file': os.path.basename(image_path),
                'result': result
            })
    
    # 保存批量预测结果
    output_dir = "outputs/predictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "batch_prediction_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("批量预测结果\n")
        f.write("="*50 + "\n\n")
        
        for item in results:
            f.write(f"文件: {item['file']}\n")
            f.write(f"预测类别: {item['result']['class_name']}\n")
            f.write(f"置信度: {item['result']['confidence']:.4f}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n📁 批量预测结果已保存: {output_file}")


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    # ==================== 配置参数 ====================
    # 预测模式选择: 'single' 或 'batch'
    PREDICT_MODE = 'single'  # 'single': 单张图像预测, 'batch': 批量预测

    # 输入路径（相对于项目根目录）
    INPUT_PATH = 'data/raw/class1_125-175/7_24_19_08-170_001.jpg'  # 单张图像路径
    # INPUT_PATH = 'data/raw/class1_125-175'  # 批量预测时使用文件夹路径

    # 模型路径（相对于项目根目录）
    MODEL_PATH = 'models/checkpoints/best_model.pth'

    # 是否显示预测结果图
    SHOW_PLOT = True

    # ==================== 路径处理 ====================
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 转换为绝对路径
    input_path_abs = os.path.join(project_root, INPUT_PATH)
    model_path_abs = os.path.join(project_root, MODEL_PATH)

    print(f"📁 项目根目录: {project_root}")
    print(f"📄 输入路径: {input_path_abs}")
    print(f"🤖 模型路径: {model_path_abs}")

    # ==================== 执行预测 ====================
    if PREDICT_MODE == 'single':
        if os.path.isfile(input_path_abs):
            predict_image(input_path_abs, model_path_abs, show_plot=SHOW_PLOT)
        else:
            print(f"❌ 图像文件不存在: {input_path_abs}")
    elif PREDICT_MODE == 'batch':
        if os.path.isdir(input_path_abs):
            predict_batch(input_path_abs, model_path_abs)
        else:
            print(f"❌ 文件夹不存在: {input_path_abs}")
    else:
        print(f"❌ 无效的预测模式: {PREDICT_MODE}")
        print("请设置 PREDICT_MODE 为 'single' 或 'batch'")
