"""
恢复训练脚本
从检查点恢复训练，避免重新开始训练
支持通过配置文件控制，可在PyCharm中直接运行
"""

import os
import sys
import yaml
import torch
import datetime
import logging
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入项目模块
from src.models.vgg import vgg16
from src.data.dataset import get_data_loaders
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.visualization import generate_all_plots


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


def setup_logging(log_dir, timestamp):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'resume_training.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("🔄 恢复训练开始")
    logger.info("="*80)
    logger.info(f"训练时间戳: {timestamp}")
    logger.info(f"日志文件: {log_file}")
    
    return logger


def create_optimizer(model, optimizer_name, learning_rate):
    """创建优化器"""
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        print(f"⚠️  未知的优化器类型: {optimizer_name}，使用默认Adam")
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


def create_scheduler(optimizer, scheduler_type, scheduler_configs, epochs):
    """创建学习率调度器"""
    if scheduler_type == 'None' or scheduler_type is None:
        return None
    
    # 处理CosineAnnealingLR的T_max参数
    cosine_config = scheduler_configs.get('CosineAnnealingLR', {}).copy()
    if cosine_config.get('T_max') == 'auto':
        cosine_config['T_max'] = epochs
    
    if 'eta_min' in cosine_config and isinstance(cosine_config['eta_min'], str):
        cosine_config['eta_min'] = float(cosine_config['eta_min'])
    
    scheduler_map = {
        'StepLR': lambda: torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_configs['StepLR']),
        'MultiStepLR': lambda: torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_configs['MultiStepLR']),
        'CosineAnnealingLR': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cosine_config),
        'ReduceLROnPlateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_configs['ReduceLROnPlateau'])
    }
    
    if scheduler_type in scheduler_map:
        return scheduler_map[scheduler_type]()
    else:
        print(f"⚠️  未知的调度器类型: {scheduler_type}，不使用调度器")
        return None


def create_loss_function(loss_function_config, train_lines, num_classes):
    """创建损失函数"""
    loss_name = loss_function_config['name']
    
    if loss_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    
    elif loss_name == "WeightedCrossEntropyLoss":
        # 统计训练集中各类别样本数量
        class_counts = [0] * num_classes
        for line in train_lines:
            class_id = int(line.split(';')[0])
            if 0 <= class_id < num_classes:
                class_counts[class_id] += 1
        
        # 计算权重
        total_samples = sum(class_counts)
        if loss_function_config.get('auto_weight', True):
            # 自动计算权重：样本少的类别权重大
            weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts]
        else:
            # 使用手动指定的权重
            weights = loss_function_config.get('manual_weights', [1.0] * num_classes)
        
        weights_tensor = torch.FloatTensor(weights)
        print(f"📊 类别权重: {[f'{w:.3f}' for w in weights]}")
        
        return torch.nn.CrossEntropyLoss(weight=weights_tensor)
    
    else:
        print(f"⚠️  未知的损失函数: {loss_name}，使用默认CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()


def main():
    """主函数 - 恢复训练"""
    print("🔄 VGG16 恢复训练脚本")
    print("="*80)
    
    # ==================== 路径和配置加载 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    
    print(f"📁 项目根目录: {project_root}")
    print(f"📄 配置文件: {config_path}")
    
    # 加载配置
    config = load_config(config_path)
    if config is None:
        print("❌ 无法加载配置文件，退出程序")
        return
    
    # 检查是否启用恢复训练
    resume_config = config.get('resume', {})
    if not resume_config.get('enabled', False):
        print("❌ 恢复训练未启用，请在配置文件中设置 resume.enabled: true")
        print("💡 提示: 修改 configs/training_config.yaml 中的 resume.enabled 为 true")
        return
    
    print("✅ 恢复训练模式已启用")
    
    # ==================== 检查点管理 ====================
    checkpoint_manager = CheckpointManager(project_root)
    
    # 打印检查点摘要
    checkpoint_manager.print_checkpoint_summary()
    
    # 查找检查点文件
    checkpoint_path = None
    
    # 1. 优先使用指定的检查点路径
    if resume_config.get('checkpoint_path'):
        checkpoint_path = os.path.join(project_root, resume_config['checkpoint_path'])
        if not os.path.exists(checkpoint_path):
            print(f"❌ 指定的检查点文件不存在: {checkpoint_path}")
            checkpoint_path = None
    
    # 2. 使用指定的训练目录
    if not checkpoint_path and resume_config.get('train_dir'):
        checkpoint_path = checkpoint_manager.find_latest_checkpoint(resume_config['train_dir'])
    
    # 3. 自动查找最新检查点
    if not checkpoint_path and resume_config.get('auto_find_latest', True):
        checkpoint_path = checkpoint_manager.find_latest_checkpoint()
    
    if not checkpoint_path:
        print("❌ 没有找到可用的检查点文件")
        print("💡 请检查:")
        print("   1. 是否已经进行过训练并保存了检查点")
        print("   2. 配置文件中的检查点路径是否正确")
        print("   3. 检查点文件是否存在")
        return
    
    print(f"🎯 将使用检查点: {checkpoint_path}")
    
    # ==================== 提取配置参数 ====================
    # 从配置文件中提取所有必要的参数
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']
    BATCH_SIZE = config['dataloader']['batch_size']
    NUM_WORKERS = config['dataloader']['num_workers']
    NUM_CLASSES = config['data']['num_classes']
    CLASS_NAMES = config['data']['class_names']
    
    print(f"📊 训练配置:")
    print(f"   目标轮数: {EPOCHS}")
    print(f"   学习率: {LEARNING_RATE}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   类别数量: {NUM_CLASSES}")
    
    # ==================== 设备和模型初始化 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 创建模型
    model = vgg16(
        pretrained=config['model']['pretrained'],
        num_classes=NUM_CLASSES,
        dropout=config['model']['dropout']
    )
    model.to(device)

    # 创建优化器
    optimizer = create_optimizer(model, config['training']['optimizer'], LEARNING_RATE)

    # 创建学习率调度器
    scheduler_config = config['training']['scheduler']
    scheduler_type = scheduler_config['type']
    scheduler_configs = {
        'StepLR': scheduler_config.get('step_lr', {'step_size': 7, 'gamma': 0.5}),
        'MultiStepLR': scheduler_config.get('multi_step_lr', {'milestones': [10, 15], 'gamma': 0.1}),
        'CosineAnnealingLR': scheduler_config.get('cosine_annealing_lr', {'T_max': 'auto', 'eta_min': 0}),
        'ReduceLROnPlateau': scheduler_config.get('reduce_lr_on_plateau', {'mode': 'min', 'factor': 0.5, 'patience': 3})
    }
    scheduler = create_scheduler(optimizer, scheduler_type, scheduler_configs, EPOCHS)

    # ==================== 加载检查点 ====================
    try:
        start_epoch, best_acc = checkpoint_manager.load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        print(f"✅ 检查点加载成功")
    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        return

    # 检查是否需要继续训练
    if start_epoch >= EPOCHS:
        print(f"⚠️  检查点的epoch ({start_epoch}) 已达到或超过目标轮数 ({EPOCHS})")
        print("💡 如需继续训练，请增加配置文件中的 training.epochs 值")
        return

    # ==================== 数据加载 ====================
    print(f"\n📊 准备数据加载器...")

    annotation_path = os.path.join(project_root, config['data']['annotation_file'])

    try:
        train_loader, val_loader, train_lines, val_lines = get_data_loaders(
            annotation_path=annotation_path,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_val_split=config['data']['train_val_split'],
            random_seed=config['data']['random_seed'],
            stratified_split=config['data']['stratified_split'],
            num_classes=NUM_CLASSES
        )
        print(f"✅ 数据加载器创建成功")
        print(f"   训练样本: {len(train_lines)}")
        print(f"   验证样本: {len(val_lines)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # ==================== 损失函数 ====================
    criterion = create_loss_function(
        config['training']['loss_function'],
        train_lines,
        NUM_CLASSES
    )
    criterion.to(device)

    # ==================== 目录设置 ====================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建新的时间戳目录（恢复训练使用新目录）
    base_checkpoint_dir = os.path.join(project_root, config['save']['checkpoint_dir'])
    base_log_dir = os.path.join(project_root, config['logging']['log_dir'])
    base_plot_dir = os.path.join(project_root, config['logging']['plot_dir'])

    checkpoint_dir = os.path.join(base_checkpoint_dir, f"resume_{timestamp}")
    log_dir = os.path.join(base_log_dir, f"resume_{timestamp}")
    plot_dir = os.path.join(base_plot_dir, f"resume_{timestamp}")

    # latest目录
    latest_checkpoint_dir = os.path.join(base_checkpoint_dir, "latest")
    latest_log_dir = os.path.join(base_log_dir, "latest")
    latest_plot_dir = os.path.join(base_plot_dir, "latest")

    # 创建目录
    for dir_path in [checkpoint_dir, log_dir, plot_dir, latest_checkpoint_dir, latest_log_dir, latest_plot_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 设置日志
    logger = setup_logging(log_dir, timestamp)
    logger.info(f"从检查点恢复训练: {checkpoint_path}")
    logger.info(f"起始epoch: {start_epoch + 1}")
    logger.info(f"目标epoch: {EPOCHS}")
    logger.info(f"当前最佳精度: {best_acc:.2f}%")

    # 备份配置文件
    config_backup_path = os.path.join(log_dir, 'config_backup.yaml')
    with open(config_backup_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n💾 输出目录:")
    print(f"   检查点: {checkpoint_dir}")
    print(f"   日志: {log_dir}")
    print(f"   图表: {plot_dir}")

    # ==================== 开始恢复训练 ====================
    print(f"\n🚀 开始恢复训练 (从第{start_epoch + 1}轮到第{EPOCHS}轮)")
    print("="*80)

    # 训练历史记录（从恢复点开始重新记录）
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        # ==================== 训练阶段 ====================
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]',
                         ncols=100, leave=False)

        for batch_idx, (img, label, _) in enumerate(train_pbar):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(img)
            train_loss = criterion(output, label)
            train_loss.backward()
            optimizer.step()

            # 统计
            total_train_loss += train_loss.item()
            _, predicted = output.max(1)
            train_total += label.size(0)
            train_correct += predicted.eq(label).sum().item()

            # 更新进度条
            current_loss = total_train_loss / (batch_idx + 1)
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        # 学习率调度
        if scheduler is not None and scheduler_type != 'ReduceLROnPlateau':
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ==================== 验证阶段 ====================
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]  ',
                       ncols=100, leave=False)

        with torch.no_grad():
            for batch_idx, (img, label, _) in enumerate(val_pbar):
                img = img.to(device)
                label = label.to(device)

                output = model(img)
                val_loss = criterion(output, label)

                # 统计
                total_val_loss += val_loss.item()
                _, predicted = output.max(1)
                val_total += label.size(0)
                val_correct += predicted.eq(label).sum().item()

                # 更新进度条
                current_loss = total_val_loss / (batch_idx + 1)
                current_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        # 计算平均值
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # ReduceLROnPlateau 需要在验证后调用
        if scheduler is not None and scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

        # 打印epoch结果
        print(f"Epoch {epoch+1}/{EPOCHS} 完成:")
        print(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f}")

        # 记录到日志
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if config['save']['save_best_only']:
                # 保存到时间戳目录
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                # 同时保存到latest目录
                torch.save(model.state_dict(), os.path.join(latest_checkpoint_dir, "best_model.pth"))
                print(f"  🎉 新的最佳验证精度: {best_acc:.2f}% (模型已保存)")
                logger.info(f"  🎉 新的最佳验证精度: {best_acc:.2f}% (模型已保存)")
            else:
                print(f"  🎉 新的最佳验证精度: {best_acc:.2f}%")
                logger.info(f"  🎉 新的最佳验证精度: {best_acc:.2f}%")

        # 根据配置保存检查点
        should_save_checkpoint = False
        if config['save']['save_checkpoint_every_epoch']:
            should_save_checkpoint = True
        elif (epoch + 1) % config['save']['save_frequency'] == 0:
            should_save_checkpoint = True

        if should_save_checkpoint:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'scheduler_type': scheduler_type
            }

            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            # 保存检查点
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            torch.save(checkpoint_data, os.path.join(latest_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        print("-" * 80)

    # ==================== 训练完成 ====================
    print(f"\n🎯 恢复训练完成!")
    print(f"最佳验证精度: {best_acc:.2f}%")
    print(f"模型保存在: {checkpoint_dir}/")
    print("="*80)

    # ==================== 自动生成可视化图表 ====================
    print(f"\n📊 开始生成训练分析图表...")
    try:
        # 生成所有可视化图表
        evaluation_results = generate_all_plots(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            model=model,
            dataloader=val_loader,
            device=device,
            class_names=CLASS_NAMES,
            plot_dir=plot_dir,
            latest_plot_dir=latest_plot_dir
        )

        print(f"✅ 所有图表生成完成!")
        print(f"📁 图表保存位置:")
        print(f"   📊 时间戳目录: {plot_dir}")
        print(f"   📊 最新目录: {latest_plot_dir}")

        logger.info("📊 可视化图表生成完成")
        logger.info(f"图表保存位置: {plot_dir}")

    except Exception as e:
        print(f"⚠️  图表生成过程中出现错误: {e}")
        logger.error(f"图表生成错误: {e}")
        print("训练已完成，但图表生成失败。可以稍后手动运行可视化脚本。")

    # 记录训练完成信息
    training_end_time = datetime.datetime.now()

    logger.info("="*80)
    logger.info("🎯 恢复训练完成!")
    logger.info("="*80)
    logger.info(f"最佳验证精度: {best_acc:.2f}%")
    logger.info(f"训练开始时间: {timestamp}")
    logger.info(f"训练结束时间: {training_end_time.strftime('%Y%m%d_%H%M%S')}")
    logger.info(f"模型保存位置: {checkpoint_dir}/")
    logger.info(f"日志保存位置: {log_dir}/")
    logger.info(f"配置备份位置: {config_backup_path}")
    logger.info("="*80)


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    恢复训练功能通过配置文件控制

    使用方法:
    1. 修改 configs/training_config.yaml 中的配置:
       - training.epochs: 设置目标轮数 (如100)
       - resume.enabled: 设置为 true
       - resume.auto_find_latest: 设置为 true (自动查找最新检查点)

    2. 在 PyCharm 中直接运行此脚本

    3. 脚本会自动:
       - 查找最新的检查点文件
       - 从检查点恢复训练状态
       - 继续训练到目标轮数
       - 生成完整的分析图表
    """
    main()
