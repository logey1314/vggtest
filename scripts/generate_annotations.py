"""
数据集标注文件生成脚本
"""
import os
import yaml
import numpy as np
from os import getcwd

def load_config(config_path):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 配置文件格式错误: {e}")
        return None


def generate_annotations(data_dir='data/raw', classes=None, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=10101):
    """
    生成数据集标注文件 - 三分法数据划分

    Args:
        data_dir (str): 数据目录路径
        classes (list): 类别名称列表
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        random_seed (int): 随机种子
    """
    if classes is None:
        # 尝试从配置文件读取类别信息
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'configs', 'training_config.yaml')

        config = load_config(config_path)
        if config and 'data' in config and 'class_names' in config['data']:
            classes = config['data']['class_names']
            print(f"📄 从配置文件读取类别信息: {classes}")
        else:
            # 如果无法从配置文件读取，使用默认值
            classes = ['class1_125-175', 'class2_180-230', 'class3_233-285']
            print(f"⚠️  使用默认类别信息: {classes}")
    
    # 确保输出目录存在
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'annotations')
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保数据已放置在正确的目录中")
        return False
    
    print(f"📁 扫描数据目录: {data_dir}")
    print(f"🏷️  预期类别: {classes}")
    
    total_samples = 0
    class_counts = {}
    all_lines = []  # 存储所有数据行
    
    print(f"🔍 扫描数据目录: {data_dir}")
    types_name = os.listdir(data_dir)
    
    for type_name in types_name:
        if type_name not in classes:
            print(f"⚠️  跳过未知类别: {type_name}")
            continue
            
        cls_id = classes.index(type_name)
        photos_path = os.path.join(data_dir, type_name)
        
        if not os.path.isdir(photos_path):
            print(f"⚠️  跳过非目录: {photos_path}")
            continue
        
        photos_name = os.listdir(photos_path)
        class_count = 0
        
        print(f"📂 处理类别 {cls_id}: {type_name}")
        
        for photo_name in photos_name:
            _, postfix = os.path.splitext(photo_name)
            if postfix.lower() not in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']:
                continue
            
            # 生成相对路径，不使用绝对路径
            relative_path = os.path.join(photos_path, photo_name).replace('\\', '/')
            all_lines.append(f"{cls_id};{relative_path}")
            class_count += 1
            total_samples += 1
        
        class_counts[type_name] = class_count
        print(f"   ✅ 找到 {class_count} 张图像")

    # ==================== 三分法数据划分 ====================
    print(f"\n🎯 启用三分法模式，正在进行分层采样划分...")
    print(f"   比例配置: 训练集{train_ratio*100:.0f}% / 验证集{val_ratio*100:.0f}% / 测试集{test_ratio*100:.0f}%")
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 验证比例和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️  数据集比例和不为1: {total_ratio:.3f}, 将自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # 按类别分组
    class_lines = {}
    for line in all_lines:
        class_id = int(line.split(';')[0])
        if class_id not in class_lines:
            class_lines[class_id] = []
        class_lines[class_id].append(line)
    
    # 分层划分
    train_lines = []
    val_lines = []
    test_lines = []
    
    print(f"\n📊 各类别三分法分层采样:")
    for class_id, lines in class_lines.items():
        # 打乱当前类别的数据
        np.random.shuffle(lines)
        
        # 计算划分点
        total_class_samples = len(lines)
        train_split = int(total_class_samples * train_ratio)
        val_split = int(total_class_samples * (train_ratio + val_ratio))
        
        # 确保每个数据集至少有一个样本（如果该类别有足够样本）
        if total_class_samples >= 3:
            train_split = max(1, train_split)
            val_split = max(train_split + 1, val_split)
            val_split = min(total_class_samples - 1, val_split)
        
        # 划分
        class_train = lines[:train_split]
        class_val = lines[train_split:val_split]
        class_test = lines[val_split:]
        
        train_lines.extend(class_train)
        val_lines.extend(class_val)
        test_lines.extend(class_test)
        
        print(f"   类别{class_id}: {len(class_train)}训练 + {len(class_val)}验证 + {len(class_test)}测试")
    
    # 打乱最终的数据集
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    np.random.shuffle(test_lines)
    
    # 生成三个标注文件
    train_file = os.path.join(output_dir, 'cls_train.txt')
    val_file = os.path.join(output_dir, 'cls_val.txt')
    test_file = os.path.join(output_dir, 'cls_test.txt')
    
    # 保存训练集
    with open(train_file, 'w') as f:
        for line in train_lines:
            f.write(line + '\n')
    
    # 保存验证集
    with open(val_file, 'w') as f:
        for line in val_lines:
            f.write(line + '\n')
    
    # 保存测试集
    with open(test_file, 'w') as f:
        for line in test_lines:
            f.write(line + '\n')
    
    print(f"\n📊 三分法标注文件生成完成!")
    print(f"   训练集文件: {train_file} ({len(train_lines)} 样本)")
    print(f"   验证集文件: {val_file} ({len(val_lines)} 样本)")
    print(f"   测试集文件: {test_file} ({len(test_lines)} 样本)")
    print(f"   总样本数: {len(train_lines) + len(val_lines) + len(test_lines)}")
    
    # 重置随机种子
    np.random.seed(None)
    
    return True


if __name__ == '__main__':
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    # ==================== 路径处理 ====================
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # ==================== 从配置文件读取参数 ====================
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    print(f"📄 加载配置文件: {config_path}")

    config = load_config(config_path)
    if config is None:
        print("❌ 无法加载配置文件，使用默认配置")
        # 使用默认配置
        data_dir = 'data/raw'
        classes = ['class1_125-175', 'class2_180-230', 'class3_233-285']
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        random_seed = 10101
    else:
        # 从配置文件读取参数
        data_dir = config['data']['data_dir']
        classes = config['data']['class_names']
        train_ratio = config['data'].get('train_ratio', 0.6)
        val_ratio = config['data'].get('val_ratio', 0.2)
        test_ratio = config['data'].get('test_ratio', 0.2)
        random_seed = config['data'].get('random_seed', 10101)

        print(f"✅ 配置加载成功:")
        print(f"   类别数量: {config['data']['num_classes']}")
        print(f"   类别名称: {classes}")
        print(f"   数据划分模式: 三分法 ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%)")

    # 转换为绝对路径
    data_dir_abs = os.path.join(project_root, data_dir)

    print(f"\n📁 项目根目录: {project_root}")
    print(f"📁 数据目录: {data_dir_abs}")
    print(f"📁 输出目录: data/annotations/")

    # ==================== 数据目录检查 ====================
    if not os.path.exists(data_dir_abs):
        print(f"❌ 数据目录不存在: {data_dir_abs}")
        print("请确保数据目录存在")
        exit(1)

    # ==================== 生成标注文件 ====================
    success = generate_annotations(
        data_dir_abs, 
        classes, 
        train_ratio, 
        val_ratio, 
        test_ratio, 
        random_seed
    )

    if success:
        print(f"\n🎉 标注文件生成成功!")
        print(f"现在可以运行训练脚本: python scripts/train.py")
    else:
        print(f"\n❌ 标注文件生成失败!")
