"""
训练可视化工具模块 - 重新设计版本
支持自动生成完整的训练分析图表
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.font_manager as fm
from .font_manager import setup_matplotlib_font, has_chinese_font, get_text_labels

# 初始化字体设置
setup_matplotlib_font()

def plot_enhanced_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path, use_english=False):
    """
    绘制增强版训练历史图表 - 2x2子图布局

    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        train_accuracies (list): 训练准确率列表
        val_accuracies (list): 验证准确率列表
        save_path (str): 保存路径
        use_english (bool): 是否使用英文标题（解决中文字体问题）
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(train_losses) + 1)

    # 设置标题文本
    if use_english:
        titles = {
            'loss': 'Training Loss vs Validation Loss',
            'acc': 'Training Accuracy vs Validation Accuracy',
            'val_acc': 'Validation Accuracy Progress',
            'gap': 'Overfitting Analysis (Train Acc - Val Acc)',
            'train_loss': 'Training Loss',
            'val_loss': 'Validation Loss',
            'train_acc': 'Training Accuracy',
            'val_acc': 'Validation Accuracy',
            'best': 'Best',
            'gap_label': 'Accuracy Gap (%)'
        }
    else:
        titles = {
            'loss': '训练损失 vs 验证损失',
            'acc': '训练准确率 vs 验证准确率',
            'val_acc': '验证准确率变化',
            'gap': '过拟合分析 (训练准确率 - 验证准确率)',
            'train_loss': '训练损失',
            'val_loss': '验证损失',
            'train_acc': '训练准确率',
            'val_acc': '验证准确率',
            'best': '最佳',
            'gap_label': '准确率差距 (%)'
        }

    # 1. 损失曲线对比
    ax1.plot(epochs, train_losses, 'b-', label=titles['train_loss'], linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label=titles['val_loss'], linewidth=2)
    ax1.set_title(titles['loss'], fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 准确率曲线对比
    ax2.plot(epochs, train_accuracies, 'b-', label=titles['train_acc'], linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label=titles['val_acc'], linewidth=2)
    ax2.set_title(titles['acc'], fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 验证准确率详细视图（带最佳点标记）
    ax3.plot(epochs, val_accuracies, 'g-', label=titles['val_acc'], linewidth=2)
    best_acc_idx = np.argmax(val_accuracies)
    best_acc = val_accuracies[best_acc_idx]
    ax3.plot(best_acc_idx + 1, best_acc, 'ro', markersize=8)
    ax3.annotate(f'{titles["best"]}: {best_acc:.2f}%',
                xy=(best_acc_idx + 1, best_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax3.set_title(titles['val_acc'], fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 过拟合分析（训练与验证的差距）
    acc_gap = np.array(train_accuracies) - np.array(val_accuracies)
    ax4.plot(epochs, acc_gap, 'purple', label=titles['gap_label'], linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(epochs, acc_gap, 0, alpha=0.3, color='purple')
    ax4.set_title(titles['gap'], fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel(titles['gap_label'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 增强版训练历史图表已保存: {save_path}")
    plt.close()


def evaluate_model_comprehensive(model, dataloader, device, class_names):
    """
    对模型进行综合评估，获取所有需要的指标数据

    Args:
        model: 训练好的模型
        dataloader: 验证数据加载器
        device: 设备
        class_names: 类别名称列表

    Returns:
        dict: 包含所有评估结果的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("🔍 正在进行模型评估...")

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算分类报告
    report = classification_report(all_labels, all_preds,
                                 target_names=class_names,
                                 output_dict=True)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = confusion_matrix(all_labels, all_preds, normalize='true')

    print("✅ 模型评估完成")

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'class_names': class_names
    }


def plot_confusion_matrices(evaluation_results, save_dir):
    """
    绘制标准和归一化混淆矩阵

    Args:
        evaluation_results (dict): 模型评估结果
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = evaluation_results['confusion_matrix']
    cm_normalized = evaluation_results['confusion_matrix_normalized']
    class_names = evaluation_results['class_names']

    # 1. 标准混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    plt.title('混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()

    save_path_standard = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path_standard, dpi=300, bbox_inches='tight')
    print(f"📊 标准混淆矩阵已保存: {save_path_standard}")
    plt.close()

    # 2. 归一化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '百分比'})
    plt.title('归一化混淆矩阵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.tight_layout()

    save_path_normalized = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plt.savefig(save_path_normalized, dpi=300, bbox_inches='tight')
    print(f"📊 归一化混淆矩阵已保存: {save_path_normalized}")
    plt.close()


def plot_roc_curves(evaluation_results, save_path):
    """
    绘制多类别ROC曲线

    Args:
        evaluation_results (dict): 模型评估结果
        save_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = evaluation_results['labels']
    probabilities = evaluation_results['probabilities']
    class_names = evaluation_results['class_names']
    n_classes = len(class_names)

    # 将标签转换为二进制格式
    labels_binarized = label_binarize(labels, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 为每个类别绘制ROC曲线
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正率 (True Positive Rate)', fontsize=12)
    plt.title('ROC曲线 - 多类别分类', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 ROC曲线已保存: {save_path}")
    plt.close()


def plot_pr_curves(evaluation_results, save_path):
    """
    绘制多类别精确率-召回率曲线

    Args:
        evaluation_results (dict): 模型评估结果
        save_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = evaluation_results['labels']
    probabilities = evaluation_results['probabilities']
    class_names = evaluation_results['class_names']
    n_classes = len(class_names)

    # 将标签转换为二进制格式
    labels_binarized = label_binarize(labels, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 为每个类别绘制PR曲线
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(labels_binarized[:, i], probabilities[:, i])
        avg_precision = average_precision_score(labels_binarized[:, i], probabilities[:, i])

        plt.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                label=f'{class_names[i]} (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线 - 多类别分类', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 PR曲线已保存: {save_path}")
    plt.close()


def plot_class_performance_radar(evaluation_results, save_path):
    """
    绘制类别性能雷达图

    Args:
        evaluation_results (dict): 模型评估结果
        save_path (str): 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    report = evaluation_results['classification_report']
    class_names = evaluation_results['class_names']

    # 提取每个类别的指标
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = []

    for class_name in class_names:
        class_data = report[class_name]
        class_metrics.append([
            class_data['precision'],
            class_data['recall'],
            class_data['f1-score']
        ])

    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 为每个类别绘制雷达图
    for i, (class_name, values) in enumerate(zip(class_names, class_metrics)):
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2,
                label=class_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    # 设置标签和格式
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['精确率', '召回率', 'F1分数'], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)

    plt.title('类别性能雷达图', fontsize=16, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 类别性能雷达图已保存: {save_path}")
    plt.close()


def generate_all_plots(train_losses, val_losses, train_accuracies, val_accuracies,
                      model, dataloader, device, class_names, plot_dir, latest_plot_dir):
    """
    生成所有可视化图表的主函数

    Args:
        train_losses, val_losses, train_accuracies, val_accuracies: 训练历史数据
        model: 训练好的模型
        dataloader: 验证数据加载器
        device: 设备
        class_names: 类别名称
        plot_dir: 时间戳图表目录
        latest_plot_dir: 最新图表目录
    """
    print("📊 开始生成所有可视化图表...")

    # 检测中文字体支持
    use_english = not has_chinese_font()
    if use_english:
        print("⚠️  未检测到中文字体，将使用英文标题")

    # 1. 生成增强版训练历史图表
    print("📈 生成训练历史图表...")
    plot_enhanced_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies,
        os.path.join(plot_dir, "training_history.png"), use_english=use_english
    )
    plot_enhanced_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies,
        os.path.join(latest_plot_dir, "training_history.png"), use_english=use_english
    )

    # 2. 进行模型评估
    print("🔍 进行综合模型评估...")
    evaluation_results = evaluate_model_comprehensive(model, dataloader, device, class_names)

    # 3. 生成混淆矩阵
    print("📊 生成混淆矩阵...")
    plot_confusion_matrices(evaluation_results, plot_dir)
    plot_confusion_matrices(evaluation_results, latest_plot_dir)

    # 4. 生成ROC曲线
    print("📈 生成ROC曲线...")
    plot_roc_curves(evaluation_results, os.path.join(plot_dir, "roc_curves.png"))
    plot_roc_curves(evaluation_results, os.path.join(latest_plot_dir, "roc_curves.png"))

    # 5. 生成PR曲线
    print("📈 生成PR曲线...")
    plot_pr_curves(evaluation_results, os.path.join(plot_dir, "pr_curves.png"))
    plot_pr_curves(evaluation_results, os.path.join(latest_plot_dir, "pr_curves.png"))

    # 6. 生成类别性能雷达图
    print("🎯 生成类别性能雷达图...")
    plot_class_performance_radar(evaluation_results, os.path.join(plot_dir, "class_performance_radar.png"))
    plot_class_performance_radar(evaluation_results, os.path.join(latest_plot_dir, "class_performance_radar.png"))

    print("✅ 所有可视化图表生成完成!")
    print(f"📁 时间戳目录: {plot_dir}")
    print(f"📁 最新目录: {latest_plot_dir}")

    return evaluation_results
