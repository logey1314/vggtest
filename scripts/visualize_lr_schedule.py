"""
学习率调度器可视化脚本
用于预览不同学习率调度器的变化曲线
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def visualize_lr_schedules():
    """可视化不同学习率调度器的变化曲线"""
    
    # 配置参数
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    
    # 创建一个简单的模型用于测试
    model = torch.nn.Linear(10, 1)
    
    # 不同调度器配置
    schedulers_config = {
        'StepLR': {
            'scheduler': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.5),
            'color': 'blue',
            'linestyle': '-'
        },
        'MultiStepLR': {
            'scheduler': lambda opt: optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 15], gamma=0.1),
            'color': 'red',
            'linestyle': '--'
        },
        'CosineAnnealingLR': {
            'scheduler': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-6),
            'color': 'green',
            'linestyle': '-.'
        },
        'ReduceLROnPlateau': {
            'scheduler': lambda opt: optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3),
            'color': 'orange',
            'linestyle': ':'
        }
    }
    
    plt.figure(figsize=(12, 8))
    
    for name, config in schedulers_config.items():
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = config['scheduler'](optimizer)
        
        learning_rates = []
        epochs = list(range(EPOCHS))
        
        for epoch in range(EPOCHS):
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 模拟训练过程
            if name == 'ReduceLROnPlateau':
                # 模拟验证损失变化（前期下降，后期平稳）
                if epoch < 5:
                    fake_val_loss = 1.0 - epoch * 0.1
                elif epoch < 10:
                    fake_val_loss = 0.5 - (epoch - 5) * 0.05
                else:
                    fake_val_loss = 0.25 + np.random.normal(0, 0.01)  # 添加噪声模拟平稳期
                scheduler.step(fake_val_loss)
            else:
                scheduler.step()
        
        # 绘制学习率曲线
        plt.plot(epochs, learning_rates, 
                label=name, 
                color=config['color'], 
                linestyle=config['linestyle'],
                linewidth=2,
                marker='o',
                markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('不同学习率调度器的变化曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标更好地显示变化
    
    # 添加说明文本
    plt.text(0.02, 0.98, 
             '说明：\n'
             '• StepLR: 每7个epoch降低50%\n'
             '• MultiStepLR: 第10,15个epoch降低90%\n'
             '• CosineAnnealingLR: 余弦退火到1e-6\n'
             '• ReduceLROnPlateau: 验证损失不降时减半',
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, 'outputs', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'lr_schedules_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 学习率调度器对比图已保存: {output_path}")
    
    plt.show()


def print_lr_schedule_info():
    """打印各种学习率调度器的详细信息"""
    print("📚 学习率调度器详细说明")
    print("=" * 60)
    
    print("\n1. 📉 StepLR (阶梯式衰减)")
    print("   • 特点: 每隔固定epoch数降低学习率")
    print("   • 适用: 训练稳定，需要定期降低学习率的场景")
    print("   • 参数: step_size=7, gamma=0.5 (每7个epoch降低50%)")
    
    print("\n2. 📊 MultiStepLR (多阶梯衰减)")
    print("   • 特点: 在指定的epoch降低学习率")
    print("   • 适用: 需要在特定训练阶段调整学习率")
    print("   • 参数: milestones=[10,15], gamma=0.1 (第10,15个epoch降低90%)")
    
    print("\n3. 🌊 CosineAnnealingLR (余弦退火)")
    print("   • 特点: 学习率按余弦函数平滑下降")
    print("   • 适用: 需要平滑学习率变化，避免突然跳跃")
    print("   • 参数: T_max=20, eta_min=1e-6 (20个epoch内降到1e-6)")
    print("   • 优势: 平滑变化，有助于模型收敛到更好的局部最优")
    
    print("\n4. 🎯 ReduceLROnPlateau (自适应衰减)")
    print("   • 特点: 当监控指标停止改善时自动降低学习率")
    print("   • 适用: 不确定何时降低学习率的场景")
    print("   • 参数: patience=3, factor=0.5 (连续3个epoch无改善时降低50%)")
    print("   • 优势: 自适应调整，无需手动设定时机")
    
    print("\n💡 推荐使用场景:")
    print("   • 🥇 CosineAnnealingLR: 大多数情况下的首选")
    print("   • 🥈 ReduceLROnPlateau: 不确定训练特性时")
    print("   • 🥉 StepLR: 训练过程可预测时")


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    print("🔍 学习率调度器可视化工具")
    print("=" * 50)
    
    # 打印详细信息
    print_lr_schedule_info()
    
    print("\n📈 生成学习率变化曲线...")
    
    # 生成可视化图表
    visualize_lr_schedules()
    
    print("\n✅ 可视化完成！")
    print("现在你可以根据图表选择合适的学习率调度器。")
