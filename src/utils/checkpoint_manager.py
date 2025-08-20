"""
检查点管理工具
用于训练恢复功能的检查点文件管理
"""

import os
import torch
import glob
import datetime
from typing import Optional, Dict, Any, Tuple, List


class CheckpointManager:
    """
    检查点管理器
    负责检查点文件的查找、验证、加载和保存
    """
    
    def __init__(self, project_root: str):
        """
        初始化检查点管理器
        
        Args:
            project_root (str): 项目根目录路径
        """
        self.project_root = project_root
        self.checkpoint_base_dir = os.path.join(project_root, 'models', 'checkpoints')
        self.latest_dir = os.path.join(self.checkpoint_base_dir, 'latest')
    
    def find_latest_checkpoint(self, train_dir: Optional[str] = None) -> Optional[str]:
        """
        查找最新的检查点文件
        
        Args:
            train_dir (str, optional): 指定训练目录名称
            
        Returns:
            str: 检查点文件路径，如果没找到返回None
        """
        if train_dir:
            # 查找指定训练目录中的检查点
            target_dir = os.path.join(self.checkpoint_base_dir, train_dir)
            if not os.path.exists(target_dir):
                print(f"❌ 指定的训练目录不存在: {train_dir}")
                return None
            
            checkpoint_files = glob.glob(os.path.join(target_dir, "checkpoint_epoch_*.pth"))
            if not checkpoint_files:
                print(f"❌ 在目录 {train_dir} 中没有找到检查点文件")
                return None
            
            # 按epoch数排序，返回最新的
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            print(f"✅ 在指定目录中找到检查点: {os.path.basename(latest_checkpoint)}")
            return latest_checkpoint
        
        else:
            # 自动查找最新的检查点
            # 1. 首先检查latest目录
            latest_checkpoints = glob.glob(os.path.join(self.latest_dir, "checkpoint_epoch_*.pth"))
            if latest_checkpoints:
                latest_checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                latest_checkpoint = latest_checkpoints[-1]
                print(f"✅ 在latest目录中找到检查点: {os.path.basename(latest_checkpoint)}")
                return latest_checkpoint
            
            # 2. 如果latest目录没有，查找所有训练目录
            train_dirs = [d for d in os.listdir(self.checkpoint_base_dir) 
                         if d.startswith('train_') and os.path.isdir(os.path.join(self.checkpoint_base_dir, d))]
            
            if not train_dirs:
                print("❌ 没有找到任何训练目录")
                return None
            
            # 按时间戳排序，找最新的
            train_dirs.sort(reverse=True)
            
            for train_dir in train_dirs:
                train_path = os.path.join(self.checkpoint_base_dir, train_dir)
                checkpoint_files = glob.glob(os.path.join(train_path, "checkpoint_epoch_*.pth"))
                
                if checkpoint_files:
                    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
                    latest_checkpoint = checkpoint_files[-1]
                    print(f"✅ 在训练目录 {train_dir} 中找到检查点: {os.path.basename(latest_checkpoint)}")
                    return latest_checkpoint
            
            print("❌ 没有找到任何检查点文件")
            return None
    
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        验证检查点文件的完整性
        
        Args:
            checkpoint_path (str): 检查点文件路径
            
        Returns:
            bool: 验证是否通过
        """
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return False
        
        try:
            # 尝试加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 检查必需的键
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'best_acc']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                print(f"❌ 检查点文件缺少必需的键: {missing_keys}")
                return False
            
            # 检查epoch是否为正整数
            if not isinstance(checkpoint['epoch'], int) or checkpoint['epoch'] <= 0:
                print(f"❌ 检查点中的epoch值无效: {checkpoint['epoch']}")
                return False
            
            print(f"✅ 检查点文件验证通过")
            print(f"   📊 Epoch: {checkpoint['epoch']}")
            print(f"   🎯 最佳精度: {checkpoint.get('best_acc', 'N/A'):.2f}%")
            print(f"   📈 验证精度: {checkpoint.get('val_acc', 'N/A'):.2f}%")
            
            return True

        except Exception as e:
            print(f"❌ 检查点文件验证失败: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler=None) -> Tuple[int, float]:
        """
        加载检查点并恢复训练状态

        Args:
            checkpoint_path (str): 检查点文件路径
            model: PyTorch模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）

        Returns:
            Tuple[int, float]: (起始epoch, 最佳精度)
        """
        print(f"📥 加载检查点: {checkpoint_path}")

        # 验证检查点
        if not self.validate_checkpoint(checkpoint_path):
            raise ValueError("检查点文件验证失败")

        # 加载检查点数据
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 恢复模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型状态已恢复")

        # 恢复优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ 优化器状态已恢复")

        # 恢复学习率调度器状态（如果存在）
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ 学习率调度器状态已恢复")
        elif scheduler is not None:
            print("⚠️  检查点中没有学习率调度器状态，将使用默认状态")

        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        print(f"🚀 训练将从第 {start_epoch + 1} 轮开始")
        print(f"🎯 当前最佳精度: {best_acc:.2f}%")

        return start_epoch, best_acc

    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的检查点文件

        Returns:
            List[Dict]: 检查点信息列表
        """
        checkpoints = []

        # 扫描所有训练目录
        if not os.path.exists(self.checkpoint_base_dir):
            return checkpoints

        train_dirs = [d for d in os.listdir(self.checkpoint_base_dir)
                     if d.startswith('train_') and os.path.isdir(os.path.join(self.checkpoint_base_dir, d))]

        # 也包括latest目录
        if os.path.exists(self.latest_dir):
            train_dirs.append('latest')

        for train_dir in train_dirs:
            train_path = os.path.join(self.checkpoint_base_dir, train_dir)
            checkpoint_files = glob.glob(os.path.join(train_path, "checkpoint_epoch_*.pth"))

            for checkpoint_file in checkpoint_files:
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')

                    # 提取epoch数
                    epoch = checkpoint.get('epoch', 0)

                    # 获取文件信息
                    file_stat = os.stat(checkpoint_file)
                    file_time = datetime.datetime.fromtimestamp(file_stat.st_mtime)

                    checkpoint_info = {
                        'path': checkpoint_file,
                        'train_dir': train_dir,
                        'epoch': epoch,
                        'best_acc': checkpoint.get('best_acc', 0.0),
                        'val_acc': checkpoint.get('val_acc', 0.0),
                        'file_time': file_time,
                        'file_size': file_stat.st_size
                    }

                    checkpoints.append(checkpoint_info)

                except Exception as e:
                    print(f"⚠️  无法读取检查点 {checkpoint_file}: {e}")

        # 按时间排序
        checkpoints.sort(key=lambda x: x['file_time'], reverse=True)

        return checkpoints

    def print_checkpoint_summary(self):
        """
        打印检查点摘要信息
        """
        checkpoints = self.list_available_checkpoints()

        if not checkpoints:
            print("❌ 没有找到任何检查点文件")
            return

        print(f"\n📋 可用检查点摘要 (共 {len(checkpoints)} 个):")
        print("=" * 80)
        print(f"{'序号':<4} {'训练目录':<20} {'Epoch':<8} {'最佳精度':<10} {'验证精度':<10} {'文件时间':<20}")
        print("-" * 80)

        for i, checkpoint in enumerate(checkpoints[:10], 1):  # 只显示最近10个
            train_dir = checkpoint['train_dir']
            epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            val_acc = checkpoint['val_acc']
            file_time = checkpoint['file_time'].strftime('%Y-%m-%d %H:%M:%S')

            print(f"{i:<4} {train_dir:<20} {epoch:<8} {best_acc:<10.2f} {val_acc:<10.2f} {file_time:<20}")

        if len(checkpoints) > 10:
            print(f"... 还有 {len(checkpoints) - 10} 个检查点未显示")

        print("=" * 80)
