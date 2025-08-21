"""
多模型图像分类训练脚本
支持VGG16、ResNet等多种模型架构
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import yaml
import datetime
import logging
import shutil

# 导入工厂系统
from src.models import create_model
from src.training import create_optimizer, create_scheduler, create_loss_function
from src.data.dataset import DataGenerator
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


def stratified_train_val_split(lines, train_ratio=0.8, random_seed=None, num_classes=None):
    """
    分层采样划分训练集和验证集
    确保训练集和验证集中各类别的比例保持一致

    Args:
        lines: 标注数据行列表
        train_ratio: 训练集比例
        random_seed: 随机种子
        num_classes: 类别数量（动态适应）

    Returns:
        train_lines, val_lines: 训练集和验证集数据
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 动态确定类别数量
    if num_classes is None:
        # 从数据中自动推断类别数量
        all_class_ids = set()
        for line in lines:
            class_id = int(line.split(';')[0])
            all_class_ids.add(class_id)
        num_classes = len(all_class_ids)
        print(f"🔍 自动检测到 {num_classes} 个类别: {sorted(all_class_ids)}")

    # 动态创建类别分组字典
    class_lines = {i: [] for i in range(num_classes)}
    for line in lines:
        class_id = int(line.split(';')[0])
        if class_id < num_classes:  # 确保类别ID在有效范围内
            class_lines[class_id].append(line)
        else:
            print(f"⚠️  跳过无效类别ID: {class_id} (超出范围 0-{num_classes-1})")

    # 统计原始分布
    total_samples = len(lines)
    print(f"\n📊 原始数据分布:")
    for class_id, class_data in class_lines.items():
        count = len(class_data)
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"   类别{class_id}: {count:,}样本 ({percentage:.1f}%)")

    # 对每个类别进行分层划分
    train_lines = []
    val_lines = []

    print(f"\n🎯 分层采样划分 (训练集{train_ratio*100:.0f}% / 验证集{(1-train_ratio)*100:.0f}%):")

    for class_id, class_data in class_lines.items():
        if len(class_data) == 0:
            print(f"   类别{class_id}: 0训练 + 0验证 (无数据)")
            continue

        # 打乱当前类别的数据
        np.random.shuffle(class_data)

        # 计算划分点
        split_point = int(len(class_data) * train_ratio)

        # 划分训练集和验证集
        class_train = class_data[:split_point]
        class_val = class_data[split_point:]

        train_lines.extend(class_train)
        val_lines.extend(class_val)

        print(f"   类别{class_id}: {len(class_train)}训练 + {len(class_val)}验证")

    # 打乱最终的训练集和验证集
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)

    # 验证分层效果
    print(f"\n✅ 分层采样结果验证:")

    # 动态创建类别计数字典
    train_class_counts = {i: 0 for i in range(num_classes)}
    for line in train_lines:
        class_id = int(line.split(';')[0])
        if class_id in train_class_counts:
            train_class_counts[class_id] += 1

    # 统计验证集分布
    val_class_counts = {i: 0 for i in range(num_classes)}
    for line in val_lines:
        class_id = int(line.split(';')[0])
        if class_id in val_class_counts:
            val_class_counts[class_id] += 1

    print(f"   训练集分布:")
    for class_id, count in train_class_counts.items():
        percentage = count / len(train_lines) * 100 if len(train_lines) > 0 else 0
        print(f"     类别{class_id}: {count:,}样本 ({percentage:.1f}%)")

    print(f"   验证集分布:")
    for class_id, count in val_class_counts.items():
        percentage = count / len(val_lines) * 100 if len(val_lines) > 0 else 0
        print(f"     类别{class_id}: {count:,}样本 ({percentage:.1f}%)")

    if random_seed is not None:
        np.random.seed(None)  # 重置随机种子

    return train_lines, val_lines


def main():
    """主训练函数"""

    # ==================== 加载配置文件 ====================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'configs', 'training_config.yaml')

    print(f"📄 加载配置文件: {config_path}")
    config = load_config(config_path)

    if config is None:
        print("❌ 无法加载配置文件，程序退出")
        return

    # ==================== 解析配置参数 ====================
    # 训练参数
    EPOCHS = config['training']['epochs']
    BATCH_SIZE = config['dataloader']['batch_size']
    LEARNING_RATE = config['training']['learning_rate']
    NUM_CLASSES = config['data']['num_classes']
    INPUT_SHAPE = config['data']['input_shape']

    # 数据集配置
    TRAIN_VAL_SPLIT = config['data']['train_val_split']
    RANDOM_SEED = config['data']['random_seed']
    STRATIFIED_SPLIT = config['data']['stratified_split']

    # 数据加载器配置
    NUM_WORKERS = config['dataloader']['num_workers']
    SHUFFLE_TRAIN = config['dataloader']['shuffle']



    # 数据增强配置
    aug_config = config['augmentation']

    # 检查是否使用预设配置
    preset_name = aug_config.get('use_preset', None)
    if preset_name and preset_name in aug_config.get('presets', {}):
        print(f"🎨 使用数据增强预设: {preset_name}")
        preset_config = aug_config['presets'][preset_name]
        # 合并预设配置和当前配置，预设优先
        merged_config = {**aug_config, **preset_config}
        aug_config = merged_config

    AUGMENTATION_CONFIG = {
        'enable_flip': aug_config.get('enable_flip', True),
        'enable_rotation': aug_config.get('enable_rotation', True),
        'enable_color_jitter': aug_config.get('enable_color_jitter', True),
        'enable_scale_jitter': aug_config.get('enable_scale_jitter', True),
        'jitter': aug_config.get('jitter', 0.3),
        'scale_range': tuple(aug_config.get('scale_range', [0.75, 1.25])),
        'rotation_range': aug_config.get('rotation_range', 15),
        'hue_range': aug_config.get('hue_range', 0.1),
        'saturation_range': aug_config.get('saturation_range', 1.5),
        'value_range': aug_config.get('value_range', 1.5),
        'flip_probability': aug_config.get('flip_probability', 0.5),
        'rotation_probability': aug_config.get('rotation_probability', 0.5),
    }

    # 路径配置
    ANNOTATION_PATH = config['data']['annotation_file']
    CHECKPOINT_DIR = config['save']['checkpoint_dir']

    # 保存配置
    SAVE_BEST_ONLY = config['save']['save_best_only']
    SAVE_FREQUENCY = config['save']['save_frequency']
    SAVE_CHECKPOINT_EVERY_EPOCH = config['save']['save_checkpoint_every_epoch']

    # 日志配置
    VERBOSE = config['logging']['verbose']

    # 数据配置
    DATA_DIR = config['data']['data_dir']
    CLASS_NAMES = config['data']['class_names']

    # ==================== 路径处理 ====================
    # 创建时间戳用于区分不同训练
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 转换为绝对路径
    annotation_path = os.path.join(project_root, ANNOTATION_PATH)

    # 创建时间戳目录结构
    base_checkpoint_dir = os.path.join(project_root, CHECKPOINT_DIR)
    base_log_dir = os.path.join(project_root, config['logging']['log_dir'])
    base_plot_dir = os.path.join(project_root, config['logging']['plot_dir'])

    # 时间戳目录
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"train_{timestamp}")
    log_dir = os.path.join(base_log_dir, f"train_{timestamp}")
    plot_dir = os.path.join(base_plot_dir, f"train_{timestamp}")

    # latest目录（用于脚本自动调用）
    latest_checkpoint_dir = os.path.join(base_checkpoint_dir, "latest")
    latest_log_dir = os.path.join(base_log_dir, "latest")
    latest_plot_dir = os.path.join(base_plot_dir, "latest")

    if VERBOSE:
        print(f"📁 项目根目录: {project_root}")
        print(f"📄 标注文件: {annotation_path}")
        print(f"🤖 使用预训练模型: {'是' if config['model']['pretrained'] else '否'}")
        print(f"⏰ 训练时间戳: {timestamp}")
        print(f"💾 检查点目录: {checkpoint_dir}")
        print(f"📝 日志目录: {log_dir}")
        print(f"📊 图表目录: {plot_dir}")

    # ==================== 创建目录和设置日志 ====================
    # 创建所有必要的目录
    for directory in [checkpoint_dir, log_dir, plot_dir, latest_checkpoint_dir, latest_log_dir, latest_plot_dir]:
        os.makedirs(directory, exist_ok=True)

    # 备份配置文件到日志目录
    config_backup_path = os.path.join(log_dir, "config_backup.yaml")
    config_backup_latest_path = os.path.join(latest_log_dir, "config_backup.yaml")
    shutil.copy2(config_path, config_backup_path)
    shutil.copy2(config_path, config_backup_latest_path)

    # 设置日志记录
    log_file_path = os.path.join(log_dir, "training.log")
    log_file_latest_path = os.path.join(latest_log_dir, "training.log")

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.FileHandler(log_file_latest_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    logger = logging.getLogger(__name__)

    # 记录训练开始信息
    logger.info("="*80)
    logger.info("🚀 开始新的训练任务")
    logger.info("="*80)
    logger.info(f"训练时间戳: {timestamp}")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"配置备份: {config_backup_path}")
    logger.info(f"检查点目录: {checkpoint_dir}")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"图表目录: {plot_dir}")

    print(f"📝 日志文件: {log_file_path}")
    print(f"📋 配置备份: {config_backup_path}")

    # ==================== 数据集准备 ====================
    # 如果标注文件不存在，尝试从旧位置读取
    if not os.path.exists(annotation_path):
        old_annotation_path = os.path.join(project_root, 'cls_train.txt')
        if os.path.exists(old_annotation_path):
            annotation_path = old_annotation_path
            print(f"⚠️  使用旧标注文件: {annotation_path}")
        else:
            print(f"❌ 标注文件不存在: {annotation_path}")
            print("请先运行 scripts/generate_annotations.py 生成标注文件")
            return
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # 数据集划分
    if STRATIFIED_SPLIT:
        print(f"\n🎯 使用分层采样划分数据集...")
        train_lines, val_lines = stratified_train_val_split(
            lines,
            train_ratio=TRAIN_VAL_SPLIT,
            random_seed=RANDOM_SEED,
            num_classes=NUM_CLASSES
        )
    else:
        print(f"\n📊 使用随机划分数据集...")
        np.random.seed(RANDOM_SEED)  # 使用配置文件中的随机种子
        np.random.shuffle(lines)  # 数据打乱
        np.random.seed(None)

        # 随机划分训练集和验证集
        train_lines = lines[:int(len(lines) * TRAIN_VAL_SPLIT)]
        val_lines = lines[int(len(lines) * TRAIN_VAL_SPLIT):]

        print(f"   训练样本: {len(train_lines):,}")
        print(f"   验证样本: {len(val_lines):,}")

    # 更新样本数量
    num_train = len(train_lines)
    num_val = len(val_lines)

    print(f"📊 数据集信息:")
    print(f"   总样本数: {len(lines)}")
    print(f"   训练样本: {num_train}")
    print(f"   验证样本: {num_val}")
    print(f"   训练/验证比例: {num_train/len(lines):.1%}/{num_val/len(lines):.1%}")

    # 记录数据集信息到日志
    logger.info("="*50)
    logger.info("📊 数据集信息")
    logger.info("="*50)
    logger.info(f"标注文件: {annotation_path}")
    logger.info(f"总样本数: {len(lines)}")
    logger.info(f"训练样本: {num_train}")
    logger.info(f"验证样本: {num_val}")
    logger.info(f"训练/验证比例: {num_train/len(lines):.1%}/{num_val/len(lines):.1%}")

    # 创建数据集
    train_data = DataGenerator(train_lines, INPUT_SHAPE, True, AUGMENTATION_CONFIG)
    val_data = DataGenerator(val_lines, INPUT_SHAPE, False, None)  # 验证集不使用数据增强
    val_len = len(val_data)

    # ==================== 数据加载器 ====================
    gen_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TRAIN, num_workers=NUM_WORKERS)
    gen_test = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   训练批次数: {len(gen_train)}")
    print(f"   验证批次数: {len(gen_test)}")

    # ==================== 构建网络 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  设备信息:")
    print(f"   使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 记录设备信息到日志
    logger.info("="*50)
    logger.info("🖥️ 设备信息")
    logger.info("="*50)
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\n🏗️  构建网络:")

    # 使用模型工厂创建模型
    model_config = config['model'].copy()
    model_config['num_classes'] = NUM_CLASSES
    net = create_model(model_config)
    net.to(device)

    # 获取模型信息
    if hasattr(net, 'get_parameter_count'):
        param_info = net.get_parameter_count()
        total_params = param_info['total']
        trainable_params = param_info['trainable']
    else:
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # 获取模型名称
    model_name = net.get_name() if hasattr(net, 'get_name') else config['model']['name'].upper()

    # 记录模型信息到日志
    logger.info("="*50)
    logger.info("🏗️ 模型信息")
    logger.info("="*50)
    logger.info(f"模型名称: {model_name}")
    logger.info(f"类别数量: {NUM_CLASSES}")
    logger.info(f"使用预训练: {config['model']['pretrained']}")
    logger.info(f"模型参数总量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    # ==================== 优化器和学习率调度 ====================
    # 使用优化器工厂创建优化器
    optimizer_config = config['training']['optimizer'].copy()
    optimizer_config['params']['lr'] = LEARNING_RATE  # 使用配置文件中的学习率
    optim = create_optimizer(net, optimizer_config)

    # 使用调度器工厂创建学习率调度器
    scheduler_config = config['training']['scheduler']
    scheduler = create_scheduler(optim, scheduler_config, EPOCHS)

    print(f"\n⚙️  训练配置:")
    print(f"   优化器: {config['training']['optimizer']['name']}")
    print(f"   初始学习率: {LEARNING_RATE}")
    print(f"   训练轮数: {EPOCHS}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   损失函数: {config['training']['loss_function']['name']}")
    print(f"   保存最佳模型: {'是' if SAVE_BEST_ONLY else '否'}")
    print(f"   保存频率: 每{SAVE_FREQUENCY}个epoch" if not SAVE_CHECKPOINT_EVERY_EPOCH else "每个epoch")
    if scheduler is not None:
        print(f"   学习率调度器: {config['training']['scheduler']['name']}")
    else:
        print(f"   学习率调度器: 无")

    print(f"\n🎨 数据增强配置:")
    print(f"   水平翻转: {'✅' if AUGMENTATION_CONFIG['enable_flip'] else '❌'}")
    print(f"   随机旋转: {'✅' if AUGMENTATION_CONFIG['enable_rotation'] else '❌'} (±{AUGMENTATION_CONFIG['rotation_range']}°)")
    print(f"   色彩抖动: {'✅' if AUGMENTATION_CONFIG['enable_color_jitter'] else '❌'}")
    print(f"   尺度抖动: {'✅' if AUGMENTATION_CONFIG['enable_scale_jitter'] else '❌'} ({AUGMENTATION_CONFIG['scale_range']})")
    print(f"   长宽比抖动: {AUGMENTATION_CONFIG['jitter']}")
    print(f"   翻转概率: {AUGMENTATION_CONFIG['flip_probability']}")
    print(f"   旋转概率: {AUGMENTATION_CONFIG['rotation_probability']}")

    print(f"\n🚀 开始训练 (共 {EPOCHS} 个 epoch)")
    print("="*80)

    # 记录训练开始
    logger.info("="*80)
    logger.info(f"🚀 开始训练 (共 {EPOCHS} 个 epoch)")
    logger.info("="*80)

    # 目录已在前面创建，记录训练配置到日志
    logger.info("="*50)
    logger.info("📊 训练配置参数")
    logger.info("="*50)
    logger.info(f"训练轮数: {EPOCHS}")
    logger.info(f"批次大小: {BATCH_SIZE}")
    logger.info(f"学习率: {LEARNING_RATE}")
    logger.info(f"类别数量: {NUM_CLASSES}")
    logger.info(f"输入尺寸: {INPUT_SHAPE}")
    logger.info(f"模型类型: {config['model']['name']}")
    logger.info(f"使用预训练: {config['model']['pretrained']}")
    logger.info(f"优化器: {config['training']['optimizer']['name']}")
    logger.info(f"学习率调度器: {config['training']['scheduler']['name']}")
    logger.info(f"数据增强配置: {AUGMENTATION_CONFIG is not None}")

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accuracies = []  # 新增：记录训练准确率
    val_accuracies = []
    best_acc = 0.0

    # ==================== 创建损失函数 ====================
    # 使用损失函数工厂创建损失函数
    loss_config = config['training']['loss_function']
    criterion = create_loss_function(loss_config, train_lines, NUM_CLASSES)
    criterion.to(device)

    # 记录损失函数信息到日志
    logger.info("="*50)
    logger.info("🎯 损失函数配置")
    logger.info("="*50)
    logger.info(f"损失函数类型: {config['training']['loss_function']['name']}")
    logger.info(f"分层采样: {'启用' if STRATIFIED_SPLIT else '未启用'}")

    # ==================== 训练循环 ====================
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # ==================== 训练阶段 ====================
        net.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练进度条
        train_pbar = tqdm(gen_train, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]',
                         ncols=100, leave=False)

        for batch_idx, (img, label, _) in enumerate(train_pbar):
            img = img.to(device)
            label = label.to(device)

            optim.zero_grad()
            output = net(img)
            train_loss = criterion(output, label)  # 使用加权损失函数
            train_loss.backward()
            optim.step()

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
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        current_lr = optim.param_groups[0]['lr']

        # ==================== 验证阶段 ====================
        net.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 验证进度条
        val_pbar = tqdm(gen_test, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]  ',
                       ncols=100, leave=False)

        with torch.no_grad():
            for batch_idx, (img, label, _) in enumerate(val_pbar):
                img = img.to(device)
                label = label.to(device)

                output = net(img)
                val_loss = criterion(output, label)  # 使用加权损失函数

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
        avg_train_loss = total_train_loss / len(gen_train)
        avg_val_loss = total_val_loss / len(gen_test)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)  # 新增：记录训练准确率
        val_accuracies.append(val_acc)

        # ReduceLROnPlateau 需要在验证后调用
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
            # 更新学习率（可能在step后发生变化）
            current_lr = optim.param_groups[0]['lr']

        # 计算训练时间
        epoch_time = time.time() - epoch_start_time

        # 打印epoch总结
        print(f"\nEpoch {epoch+1}/{EPOCHS} 完成:")
        print(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  学习率: {current_lr:.6f}")
        print(f"  用时: {epoch_time:.1f}s")

        # 记录epoch结果到日志
        logger.info(f"Epoch {epoch+1}/{EPOCHS} 完成:")
        logger.info(f"  训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  验证 - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        logger.info(f"  学习率: {current_lr:.6f}")
        logger.info(f"  用时: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if SAVE_BEST_ONLY:  # 根据配置决定是否保存最佳模型
                # 保存到时间戳目录
                torch.save(net.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
                # 同时保存到latest目录（用于评估脚本）
                torch.save(net.state_dict(), os.path.join(latest_checkpoint_dir, "best_model.pth"))
                print(f"  🎉 新的最佳验证精度: {best_acc:.2f}% (模型已保存)")
                logger.info(f"  🎉 新的最佳验证精度: {best_acc:.2f}% (模型已保存)")
            else:
                print(f"  🎉 新的最佳验证精度: {best_acc:.2f}%")
                logger.info(f"  🎉 新的最佳验证精度: {best_acc:.2f}%")

        # 根据配置保存检查点
        should_save_checkpoint = False
        if SAVE_CHECKPOINT_EVERY_EPOCH:
            should_save_checkpoint = True
        elif (epoch + 1) % SAVE_FREQUENCY == 0:
            should_save_checkpoint = True

        if should_save_checkpoint:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'scheduler_type': config['training']['scheduler']['name']
            }

            # 只有当调度器存在时才保存其状态
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            # 保存检查点到时间戳目录
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            # 同时保存到latest目录
            torch.save(checkpoint_data, os.path.join(latest_checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

        print("-" * 80)

    print(f"\n🎯 训练完成!")
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
            model=net,
            dataloader=gen_test,
            device=device,
            class_names=CLASS_NAMES,
            plot_dir=plot_dir,
            latest_plot_dir=latest_plot_dir
        )

        print(f"✅ 所有图表生成完成!")
        print(f"📁 图表保存位置:")
        print(f"   📊 时间戳目录: {plot_dir}")
        print(f"   📊 最新目录: {latest_plot_dir}")

        # 记录图表生成信息到日志
        logger.info("📊 可视化图表生成完成")
        logger.info(f"图表保存位置: {plot_dir}")
        logger.info(f"最新图表位置: {latest_plot_dir}")

    except Exception as e:
        print(f"⚠️  图表生成过程中出现错误: {e}")
        logger.error(f"图表生成错误: {e}")
        print("训练已完成，但图表生成失败。可以稍后手动运行可视化脚本。")

    # 记录训练完成信息
    training_end_time = datetime.datetime.now()
    total_training_time = training_end_time - datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")

    logger.info("="*80)
    logger.info("🎯 训练完成!")
    logger.info("="*80)
    logger.info(f"最佳验证精度: {best_acc:.2f}%")
    logger.info(f"训练开始时间: {timestamp}")
    logger.info(f"训练结束时间: {training_end_time.strftime('%Y%m%d_%H%M%S')}")
    logger.info(f"总训练时长: {total_training_time}")
    logger.info(f"模型保存位置: {checkpoint_dir}/")
    logger.info(f"日志保存位置: {log_dir}/")
    logger.info(f"配置备份位置: {config_backup_path}")
    logger.info("="*80)


if __name__ == "__main__":
    """
    直接运行配置 - 可以在 PyCharm 中直接点击运行
    """
    main()
