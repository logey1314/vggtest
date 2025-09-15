#!/usr/bin/env python3
"""
复制测试集中所有类别9的图片到指定目录
"""

import os
import shutil
import sys
from pathlib import Path

def copy_class9_images():
    """复制测试集中所有类别9的图片"""
    print("🚀 开始复制类别9图片...")
    
    # 目标目录
    target_dir = "/var/yjs/zes/vgg/data/class9"
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    print(f"✅ 创建目标目录: {target_dir}")
    
    # 查找测试集文件
    possible_paths = [
        "data/annotations/cls_test.txt",
        "/var/yjs/zes/vgg/data/annotations/cls_test.txt",
        "../data/annotations/cls_test.txt",
        "../../data/annotations/cls_test.txt"
    ]
    
    test_file = None
    for path in possible_paths:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        print("❌ 测试集文件不存在")
        return
    
    print(f"✅ 找到测试集文件: {test_file}")
    
    # 读取测试集数据
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    # 筛选类别9的图片
    class9_paths = []
    for line in lines:
        parts = line.split(';')
        if len(parts) >= 2:
            class_id = int(parts[0])
            image_path = parts[1]
            
            if class_id == 8:  # 类别9对应索引8
                class9_paths.append(image_path)
    
    print(f"📊 找到 {len(class9_paths)} 个类别9图片")
    
    # 复制图片
    copied_count = 0
    failed_count = 0
    
    for i, source_path in enumerate(class9_paths):
        try:
            # 检查源文件是否存在
            if not os.path.exists(source_path):
                print(f"⚠️  源文件不存在: {source_path}")
                failed_count += 1
                continue
            
            # 获取文件名
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_dir, filename)
            
            # 如果目标文件已存在，添加序号
            if os.path.exists(target_path):
                name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    new_filename = f"{name}_{counter}{ext}"
                    target_path = os.path.join(target_dir, new_filename)
                    counter += 1
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
            # 显示进度
            if (i + 1) % 20 == 0:
                print(f"   已处理: {i + 1}/{len(class9_paths)}")
                
        except Exception as e:
            print(f"❌ 复制失败 {source_path}: {e}")
            failed_count += 1
            continue
    
    print(f"\n✅ 复制完成!")
    print(f"   成功复制: {copied_count} 个文件")
    print(f"   失败: {failed_count} 个文件")
    print(f"   目标目录: {target_dir}")
    
    # 验证复制结果
    if os.path.exists(target_dir):
        copied_files = os.listdir(target_dir)
        print(f"   目标目录中的文件数: {len(copied_files)}")
        
        # 显示前几个文件名
        print("   复制的文件示例:")
        for i, filename in enumerate(copied_files[:5]):
            print(f"     {i+1}: {filename}")
        if len(copied_files) > 5:
            print(f"     ... 还有 {len(copied_files) - 5} 个文件")

def main():
    """主函数"""
    print("📁 类别9图片复制脚本")
    print("="*50)
    
    try:
        copy_class9_images()
        print("="*50)
        print("🎉 任务完成!")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
