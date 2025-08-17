"""
训练可视化工具模块
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_training_history(train_losses, val_losses, val_accuracies, save_path="outputs/plots/training_plots.png"):
    """
    绘制训练历史曲线
    
    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        val_accuracies (list): 验证精度列表
        save_path (str): 保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_title('训练和验证损失', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制精度曲线
    ax2.plot(epochs, val_accuracies, 'g-', label='验证精度', linewidth=2)
    ax2.set_title('验证精度', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加最佳精度标记
    best_acc_idx = np.argmax(val_accuracies)
    best_acc = val_accuracies[best_acc_idx]
    ax2.plot(best_acc_idx + 1, best_acc, 'ro', markersize=8)
    ax2.annotate(f'最佳: {best_acc:.2f}%', 
                xy=(best_acc_idx + 1, best_acc), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 训练历史图表已保存: {save_path}")
    plt.show()


def load_training_history_from_checkpoints(checkpoint_dir="models/checkpoints"):
    """
    从检查点文件中加载训练历史
    
    Args:
        checkpoint_dir (str): 检查点目录
        
    Returns:
        tuple: (train_losses, val_losses, val_accuracies)
    """
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 检查点目录不存在: {checkpoint_dir}")
        return None, None, None
    
    # 获取所有检查点文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f"❌ 在 {checkpoint_dir} 中未找到检查点文件")
        return None, None, None
    
    # 按epoch排序
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"📁 找到 {len(checkpoint_files)} 个检查点文件")
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            train_losses.append(checkpoint['train_loss'])
            val_losses.append(checkpoint['val_loss'])
            val_accuracies.append(checkpoint['val_acc'])
        except Exception as e:
            print(f"⚠️  加载检查点 {checkpoint_file} 时出错: {e}")
            continue
    
    return train_losses, val_losses, val_accuracies


def plot_model_comparison(results_dict, save_path="outputs/plots/model_comparison.png"):
    """
    绘制多个模型的性能比较图
    
    Args:
        results_dict (dict): 模型结果字典，格式: {'model_name': {'train_losses': [], 'val_losses': [], 'val_accuracies': []}}
        save_path (str): 保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(results['val_losses']) + 1)
        
        # 绘制验证损失
        ax1.plot(epochs, results['val_losses'], color=color, label=model_name, linewidth=2)
        
        # 绘制验证精度
        ax2.plot(epochs, results['val_accuracies'], color=color, label=model_name, linewidth=2)
    
    ax1.set_title('验证损失比较', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('验证精度比较', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 模型比较图表已保存: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="outputs/plots/confusion_matrix.png"):
    """
    绘制混淆矩阵
    
    Args:
        y_true (array): 真实标签
        y_pred (array): 预测标签
        class_names (list): 类别名称列表
        save_path (str): 保存路径
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 混淆矩阵已保存: {save_path}")
    plt.show()
