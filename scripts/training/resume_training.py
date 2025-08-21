"""
恢复训练脚本
从检查点恢复训练，避免重新开始训练
支持通过配置文件控制，可在PyCharm中直接运行
"""

import os
import sys
import yaml
import torch
import time
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# 导入项目模块
from src.models.vgg import vgg16
from src.data.dataset import get_data_loaders
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.visualization import generate_all_plots
from .training_utils import (
    validate_resume_config,
    setup_training_directories,
    backup_config_file,
    setup_training_logger,
    print_training_summary,
    check_training_prerequisites,
    get_training_status_summary
)


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


def create_optimizer(model, optimizer_name, learning_rate):
    """创建优化器"""
    optimizer_map = {
        'adam': torch.optim.Adam,
        'sgd': lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop
    }
    
    optimizer_class = optimizer_map.get(optimizer_name.lower())
    if optimizer_class:
        return optimizer_class(model.parameters(), lr=learning_rate)
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
        try:
            return scheduler_map[scheduler_type]()
        except Exception as e:
            print(f"⚠️  创建调度器失败: {e}，不使用调度器")
            return None
    else:
        print(f"⚠️  未知的调度器类型: {scheduler_type}，不使用调度器")
        return None


def create_loss_function(loss_config, train_lines, num_classes):
    """创建损失函数"""
    loss_name = loss_config['name']
    
    if loss_name == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    elif loss_name == 'WeightedCrossEntropyLoss':
        # 计算类别权重
        class_counts = [0] * num_classes
        for line in train_lines:
            class_id = int(line.split(';')[0])
            class_counts[class_id] += 1
        
        total_samples = sum(class_counts)
        class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 
                        for count in class_counts]
        weights = torch.FloatTensor(class_weights)
        return torch.nn.CrossEntropyLoss(weight=weights)
    else:
        print(f"⚠️  未知的损失函数: {loss_name}，使用默认CrossEntropyLoss")
        return torch.nn.CrossEntropyLoss()


def find_checkpoint_file(checkpoint_manager, resume_config, project_root):
    """查找检查点文件"""
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
    
    return checkpoint_path


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]',
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
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    return avg_train_loss, train_acc


def validate_one_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    """验证一个epoch"""
    model.eval()
    total_val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]',
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
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    return avg_val_loss, val_acc


def main():
    """主函数 - 恢复训练"""
    print("🔄 VGG16 恢复训练脚本")
    print("="*80)
    
    # ==================== 路径和配置加载 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # 上两级目录
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')
    
    print(f"📁 项目根目录: {project_root}")
    print(f"📄 配置文件: {config_path}")
    
    # 加载配置
    config = load_config(config_path)
    if config is None:
        print("❌ 无法加载配置文件，退出程序")
        return
    
    # ==================== 配置验证 ====================
    valid, error_msg = validate_resume_config(config)
    if not valid:
        print(f"❌ {error_msg}")
        print("💡 提示: 修改 configs/training_config.yaml 中的相关配置")
        return
    
    print("✅ 恢复训练模式已启用")
    
    # ==================== 前置条件检查 ====================
    prerequisites_ok, errors = check_training_prerequisites(project_root, config)
    if not prerequisites_ok:
        print("❌ 训练前置条件检查失败:")
        for error in errors:
            print(f"   • {error}")
        return
    
    print("✅ 训练前置条件检查通过")
    
    # ==================== 检查点管理 ====================
    checkpoint_manager = CheckpointManager(project_root)
    checkpoint_manager.print_checkpoint_summary()
    
    resume_config = config.get('resume', {})
    checkpoint_path = find_checkpoint_file(checkpoint_manager, resume_config, project_root)
    
    if not checkpoint_path:
        print("❌ 没有找到可用的检查点文件")
        print("💡 请检查:")
        print("   1. 是否已经进行过训练并保存了检查点")
        print("   2. 配置文件中的检查点路径是否正确")
        print("   3. 检查点文件是否存在")
        return
    
    print(f"🎯 将使用检查点: {checkpoint_path}")
    
    # ==================== 设置训练目录 ====================
    directories = setup_training_directories(project_root, config, "resume")
    
    # ==================== 打印训练摘要 ====================
    print_training_summary(config, directories, checkpoint_path)
    
    # ==================== 提取配置参数 ====================
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']
    BATCH_SIZE = config['dataloader']['batch_size']
    NUM_WORKERS = config['dataloader']['num_workers']
    NUM_CLASSES = config['data']['num_classes']
    CLASS_NAMES = config['data']['class_names']

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
        print(f"   起始epoch: {start_epoch + 1}")
        print(f"   当前最佳精度: {best_acc:.2f}%")
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

    # ==================== 设置日志和备份配置 ====================
    logger = setup_training_logger(directories['log_dir'], directories['timestamp'], "resume_training")
    logger.info(f"从检查点恢复训练: {checkpoint_path}")
    logger.info(f"起始epoch: {start_epoch + 1}")
    logger.info(f"目标epoch: {EPOCHS}")
    logger.info(f"当前最佳精度: {best_acc:.2f}%")

    # 备份配置文件
    backup_config_file(config, directories['log_dir'])

    # ==================== 提取配置参数 ====================
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']
    BATCH_SIZE = config['dataloader']['batch_size']
    NUM_WORKERS = config['dataloader']['num_workers']
    NUM_CLASSES = config['data']['num_classes']
    CLASS_NAMES = config['data']['class_names']

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
        print(f"   起始epoch: {start_epoch + 1}")
        print(f"   当前最佳精度: {best_acc:.2f}%")
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

    # ==================== 设置日志和备份配置 ====================
    logger = setup_training_logger(directories['log_dir'], directories['timestamp'], "resume_training")
    logger.info(f"从检查点恢复训练: {checkpoint_path}")
    logger.info(f"起始epoch: {start_epoch + 1}")
    logger.info(f"目标epoch: {EPOCHS}")
    logger.info(f"当前最佳精度: {best_acc:.2f}%")

    # 备份配置文件
    backup_config_file(config, directories['log_dir'])

    print(f"\n🚀 开始恢复训练 (从第{start_epoch + 1}轮到第{EPOCHS}轮)")
    print("="*80)

    # ==================== 训练循环 ====================
    # 训练历史记录（从恢复点开始重新记录）
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        # ==================== 训练阶段 ====================
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, EPOCHS
        )

        # ==================== 验证阶段 ====================
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch, EPOCHS
        )

        # 学习率调度
        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # 检查是否为最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        # 保存检查点
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            save_dir=directories['checkpoint_dir'],
            latest_dir=directories['latest_checkpoint_dir'],
            is_best=is_best,
            save_frequency=config['save']['save_frequency']
        )

        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time

        # 打印epoch结果
        print(f"\nEpoch {epoch + 1}/{EPOCHS} 完成:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  最佳验证精度: {best_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  本轮用时: {epoch_time:.1f}秒")

        # 记录到日志
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                   f"Best Acc: {best_acc:.2f}%, LR: {current_lr:.6f}")

        # 每5个epoch打印状态摘要
        if (epoch + 1) % 5 == 0:
            status_summary = get_training_status_summary(
                epoch + 1 - start_epoch, EPOCHS - start_epoch,
                train_acc, val_acc, best_acc, elapsed_time
            )
            print(status_summary)

    # ==================== 训练完成 ====================
    total_time = time.time() - start_time
    print(f"\n🎯 恢复训练完成!")
    print(f"最佳验证精度: {best_acc:.2f}%")
    print(f"总训练时间: {total_time/3600:.2f}小时")
    print(f"模型保存在: {directories['checkpoint_dir']}/")
    print("="*80)

    logger.info("🎯 恢复训练完成")
    logger.info(f"最佳验证精度: {best_acc:.2f}%")
    logger.info(f"总训练时间: {total_time/3600:.2f}小时")

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
            plot_dir=directories['plot_dir'],
            latest_plot_dir=directories['latest_plot_dir']
        )

        print(f"✅ 所有图表生成完成!")
        print(f"📁 图表保存位置:")
        print(f"   📊 时间戳目录: {directories['plot_dir']}")
        print(f"   📊 最新目录: {directories['latest_plot_dir']}")

        logger.info("📊 可视化图表生成完成")
        logger.info(f"图表保存位置: {directories['plot_dir']}")
        logger.info(f"最新图表位置: {directories['latest_plot_dir']}")

    except Exception as e:
        print(f"⚠️  图表生成失败: {e}")
        logger.error(f"图表生成失败: {e}")

    print(f"\n🎉 恢复训练流程全部完成!")
    print(f"💡 提示:")
    print(f"   • 查看训练图表: {directories['latest_plot_dir']}")
    print(f"   • 查看训练日志: {directories['log_dir']}")
    print(f"   • 模型文件位置: {directories['latest_checkpoint_dir']}")
    print("="*80)


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
