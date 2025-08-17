"""
数据集标注文件生成脚本
"""
import os
from os import getcwd

def generate_annotations(data_dir='data/raw', output_file='data/annotations/cls_train.txt', classes=None):
    """
    生成数据集标注文件
    
    Args:
        data_dir (str): 数据目录路径
        output_file (str): 输出标注文件路径
        classes (list): 类别名称列表
    """
    if classes is None:
        classes = ['class1_125-175', 'class2_180-230', 'class3_233-285']
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保数据已放置在正确的目录中")
        return False
    
    print(f"📁 扫描数据目录: {data_dir}")
    print(f"🏷️  预期类别: {classes}")
    
    wd = getcwd()
    total_samples = 0
    class_counts = {}
    
    with open(output_file, 'w') as list_file:
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
                
                full_path = os.path.join(wd, photos_path, photo_name)
                list_file.write(f"{cls_id};{full_path}\n")
                class_count += 1
                total_samples += 1
            
            class_counts[type_name] = class_count
            print(f"   ✅ 找到 {class_count} 张图像")
    
    print(f"\n📊 标注文件生成完成!")
    print(f"   输出文件: {output_file}")
    print(f"   总样本数: {total_samples}")
    print(f"   类别分布:")
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"     {class_name}: {count} ({percentage:.1f}%)")
    
    return True


if __name__ == '__main__':
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    # ==================== 配置参数 ====================
    # 数据目录路径（相对于项目根目录）
    data_dir = 'data/raw'

    # 输出标注文件路径
    output_file = 'data/annotations/cls_train.txt'

    # 类别名称列表（必须与文件夹名称一致）
    classes = ['class1_125-175', 'class2_180-230', 'class3_233-285']

    # ==================== 路径处理 ====================
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 转换为绝对路径
    data_dir_abs = os.path.join(project_root, data_dir)
    output_file_abs = os.path.join(project_root, output_file)

    print(f"📁 项目根目录: {project_root}")
    print(f"📁 数据目录: {data_dir_abs}")
    print(f"📄 输出文件: {output_file_abs}")

    # ==================== 兼容性检查 ====================
    # 如果新目录不存在，尝试使用旧的目录结构
    if not os.path.exists(data_dir_abs):
        old_data_dir = os.path.join(project_root, 'train')
        if os.path.exists(old_data_dir):
            print(f"⚠️  新数据目录不存在，使用旧目录结构")
            data_dir_abs = old_data_dir
            output_file_abs = os.path.join(project_root, 'cls_train.txt')
        else:
            print(f"❌ 数据目录不存在: {data_dir_abs}")
            print("请确保数据目录存在")
            exit(1)

    # ==================== 生成标注文件 ====================
    success = generate_annotations(data_dir_abs, output_file_abs, classes)

    if success:
        print(f"\n🎉 标注文件生成成功!")
        print(f"现在可以运行训练脚本: python scripts/train.py")
    else:
        print(f"\n❌ 标注文件生成失败!")
