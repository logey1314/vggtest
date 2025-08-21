"""
训练历史可视化脚本 - 更新版本
现在训练脚本会自动生成图表，此脚本主要用于重新生成或查看历史训练的图表
"""
import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .visualize_utils import (
    get_project_root,
    print_tool_header
)


def show_training_visualization_info():
    """显示训练可视化信息"""
    project_root = get_project_root()
    
    print("ℹ️  注意：从新版本开始，训练脚本会自动生成所有可视化图表")
    print()
    print("📁 图表位置：")
    print(f"   - {os.path.join(project_root, 'outputs/plots/latest/')}          (最新训练结果)")
    print(f"   - {os.path.join(project_root, 'outputs/plots/train_YYYYMMDD_HHMMSS/')}  (历史训练结果)")
    print()
    print("📊 自动生成的图表包括：")
    print("   1. training_history.png          - 训练历史曲线")
    print("   2. confusion_matrix.png          - 标准混淆矩阵")
    print("   3. confusion_matrix_normalized.png - 归一化混淆矩阵")
    print("   4. roc_curves.png               - ROC曲线")
    print("   5. pr_curves.png                - 精确率-召回率曲线")
    print("   6. class_performance_radar.png   - 类别性能雷达图")
    print()
    print("🚀 如需测试可视化功能，请运行：")
    print(f"   python {os.path.join(project_root, 'tests/test_visualization.py')}")
    print()
    print("✅ 如果您刚完成训练，请直接查看 outputs/plots/latest/ 目录中的图表！")
    print()
    print("🔧 其他可视化工具：")
    print("   • 数据增强效果预览: python scripts/visualize/visualize_augmentation.py")
    print("   • 学习率调度器对比: python scripts/visualize/visualize_lr_schedule.py")


def list_available_plots():
    """列出可用的训练图表"""
    project_root = get_project_root()
    plots_dir = os.path.join(project_root, 'outputs', 'plots')
    
    if not os.path.exists(plots_dir):
        print("❌ 图表目录不存在，请先进行训练")
        return
    
    print("\n📊 可用的训练图表：")
    print("=" * 50)
    
    # 检查latest目录
    latest_dir = os.path.join(plots_dir, 'latest')
    if os.path.exists(latest_dir):
        files = [f for f in os.listdir(latest_dir) if f.endswith('.png')]
        if files:
            print(f"\n📁 最新训练结果 ({latest_dir}):")
            for file in sorted(files):
                print(f"   • {file}")
    
    # 检查历史训练目录
    train_dirs = [d for d in os.listdir(plots_dir) 
                  if d.startswith('train_') and os.path.isdir(os.path.join(plots_dir, d))]
    
    if train_dirs:
        print(f"\n📁 历史训练结果:")
        for train_dir in sorted(train_dirs, reverse=True)[:5]:  # 只显示最近5个
            train_path = os.path.join(plots_dir, train_dir)
            files = [f for f in os.listdir(train_path) if f.endswith('.png')]
            print(f"   📂 {train_dir}: {len(files)} 个图表")
        
        if len(train_dirs) > 5:
            print(f"   ... 还有 {len(train_dirs) - 5} 个历史训练目录")
    
    print("\n💡 提示：")
    print("   • 使用图片查看器打开 .png 文件查看图表")
    print("   • 训练完成后图表会自动保存到对应目录")
    print("   • latest 目录始终包含最新的训练结果")


def main():
    """主函数"""
    print_tool_header("训练历史可视化信息", "查看训练图表位置和相关信息")
    
    # 显示基本信息
    show_training_visualization_info()
    
    # 列出可用的图表
    list_available_plots()


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    main()
