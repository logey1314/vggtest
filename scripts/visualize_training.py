"""
训练历史可视化脚本
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.visualization import plot_training_history, load_training_history_from_checkpoints
import argparse

if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    # ==================== 配置参数 ====================
    # 检查点目录路径（使用latest目录）
    CHECKPOINT_DIR = 'models/checkpoints/latest'

    # 输出图片路径（使用latest目录）
    OUTPUT_PATH = 'outputs/plots/latest/training_history.png'

    # ==================== 路径处理 ====================
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 转换为绝对路径
    checkpoint_dir_abs = os.path.join(project_root, CHECKPOINT_DIR)
    output_path_abs = os.path.join(project_root, OUTPUT_PATH)

    print(f"📁 项目根目录: {project_root}")
    print(f"📁 检查点目录: {checkpoint_dir_abs}")
    print(f"📄 输出路径: {output_path_abs}")

    # ==================== 加载和可视化 ====================
    print("📊 加载训练历史数据...")
    train_losses, val_losses, val_accuracies = load_training_history_from_checkpoints(checkpoint_dir_abs)

    if train_losses is None:
        print("❌ 无法加载训练历史数据")
        print("请确保已经完成训练并且检查点文件存在")
        exit(1)

    print(f"✅ 成功加载 {len(train_losses)} 个epoch的训练数据")
    print("📈 生成训练历史图表...")

    plot_training_history(train_losses, val_losses, val_accuracies, output_path_abs)

    # 打印训练总结
    print(f"\n📋 训练总结:")
    print(f"   总训练轮数: {len(train_losses)}")
    print(f"   最终训练损失: {train_losses[-1]:.4f}")
    print(f"   最终验证损失: {val_losses[-1]:.4f}")
    print(f"   最终验证精度: {val_accuracies[-1]:.2f}%")
    print(f"   最佳验证精度: {max(val_accuracies):.2f}%")
