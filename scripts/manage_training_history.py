"""
训练历史管理脚本
用于查看、比较和管理不同训练的结果
"""

import os
import sys
import json
import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def list_training_runs(project_root):
    """列出所有训练运行记录"""
    checkpoint_base = os.path.join(project_root, 'models', 'checkpoints')
    evaluation_base = os.path.join(project_root, 'outputs', 'evaluation')
    log_base = os.path.join(project_root, 'outputs', 'logs')

    print("🚀 训练历史记录")
    print("="*80)
    
    # 列出训练检查点
    if os.path.exists(checkpoint_base):
        train_dirs = [d for d in os.listdir(checkpoint_base) 
                     if d.startswith('train_') and os.path.isdir(os.path.join(checkpoint_base, d))]
        train_dirs.sort(reverse=True)  # 最新的在前
        
        print(f"📁 训练检查点目录 ({len(train_dirs)} 个):")
        for i, train_dir in enumerate(train_dirs[:10]):  # 只显示最近10个
            train_path = os.path.join(checkpoint_base, train_dir)
            best_model_path = os.path.join(train_path, 'best_model.pth')
            
            # 获取目录创建时间
            timestamp = train_dir.replace('train_', '')
            try:
                dt = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = timestamp
            
            status = "✅" if os.path.exists(best_model_path) else "❌"
            print(f"  {i+1:2d}. {train_dir} - {time_str} {status}")
    
    print()
    
    # 列出评估结果
    if os.path.exists(evaluation_base):
        eval_dirs = [d for d in os.listdir(evaluation_base) 
                    if d.startswith('eval_') and os.path.isdir(os.path.join(evaluation_base, d))]
        eval_dirs.sort(reverse=True)  # 最新的在前
        
        print(f"📊 评估结果目录 ({len(eval_dirs)} 个):")
        for i, eval_dir in enumerate(eval_dirs[:10]):  # 只显示最近10个
            eval_path = os.path.join(evaluation_base, eval_dir)
            results_path = os.path.join(eval_path, 'reports', 'evaluation_results.json')
            
            # 获取目录创建时间
            timestamp = eval_dir.replace('eval_', '')
            try:
                dt = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = timestamp
            
            # 尝试读取评估结果
            accuracy = "N/A"
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                        accuracy = f"{results['metrics']['accuracy']:.4f}"
                except:
                    pass
            
            status = "✅" if os.path.exists(results_path) else "❌"
            print(f"  {i+1:2d}. {eval_dir} - {time_str} - 准确率: {accuracy} {status}")


def compare_evaluations(project_root, eval_dirs=None):
    """比较多个评估结果"""
    evaluation_base = os.path.join(project_root, 'outputs', 'evaluation')
    
    if eval_dirs is None:
        # 自动选择最近的几个评估
        all_eval_dirs = [d for d in os.listdir(evaluation_base) 
                        if d.startswith('eval_') and os.path.isdir(os.path.join(evaluation_base, d))]
        all_eval_dirs.sort(reverse=True)
        eval_dirs = all_eval_dirs[:3]  # 最近3个
    
    print(f"\n📊 评估结果对比 (最近{len(eval_dirs)}次)")
    print("="*80)
    
    results_data = []
    for eval_dir in eval_dirs:
        results_path = os.path.join(evaluation_base, eval_dir, 'reports', 'evaluation_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    results_data.append((eval_dir, results))
            except Exception as e:
                print(f"❌ 无法读取 {eval_dir}: {e}")
    
    if not results_data:
        print("❌ 没有找到有效的评估结果")
        return
    
    # 表头
    print(f"{'评估时间':<20} {'准确率':<10} {'F1-Score':<10} {'错误率':<10} {'相邻混淆':<10}")
    print("-" * 80)
    
    # 数据行
    for eval_dir, results in results_data:
        timestamp = eval_dir.replace('eval_', '')
        try:
            dt = datetime.datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            time_str = dt.strftime('%m-%d %H:%M')
        except:
            time_str = timestamp[:12]
        
        metrics = results['metrics']
        accuracy = f"{metrics['accuracy']:.4f}"
        f1_score = f"{metrics['f1_macro']:.4f}"
        error_rate = f"{results['error_info']['error_rate']:.4f}"
        adjacent_confusion = f"{metrics['adjacent_confusion_rate']:.4f}"
        
        print(f"{time_str:<20} {accuracy:<10} {f1_score:<10} {error_rate:<10} {adjacent_confusion:<10}")


def clean_old_results(project_root, keep_recent=5):
    """清理旧的训练和评估结果，保留最近的几个"""
    checkpoint_base = os.path.join(project_root, 'models', 'checkpoints')
    evaluation_base = os.path.join(project_root, 'outputs', 'evaluation')
    
    print(f"\n🧹 清理旧结果 (保留最近{keep_recent}个)")
    print("="*80)
    
    # 清理训练检查点
    if os.path.exists(checkpoint_base):
        train_dirs = [d for d in os.listdir(checkpoint_base) 
                     if d.startswith('train_') and os.path.isdir(os.path.join(checkpoint_base, d))]
        train_dirs.sort(reverse=True)  # 最新的在前
        
        to_delete = train_dirs[keep_recent:]  # 要删除的目录
        
        print(f"📁 训练检查点: 总共{len(train_dirs)}个，删除{len(to_delete)}个")
        for train_dir in to_delete:
            train_path = os.path.join(checkpoint_base, train_dir)
            try:
                import shutil
                shutil.rmtree(train_path)
                print(f"  ✅ 已删除: {train_dir}")
            except Exception as e:
                print(f"  ❌ 删除失败 {train_dir}: {e}")
    
    # 清理评估结果
    if os.path.exists(evaluation_base):
        eval_dirs = [d for d in os.listdir(evaluation_base) 
                    if d.startswith('eval_') and os.path.isdir(os.path.join(evaluation_base, d))]
        eval_dirs.sort(reverse=True)  # 最新的在前
        
        to_delete = eval_dirs[keep_recent:]  # 要删除的目录
        
        print(f"📊 评估结果: 总共{len(eval_dirs)}个，删除{len(to_delete)}个")
        for eval_dir in to_delete:
            eval_path = os.path.join(evaluation_base, eval_dir)
            try:
                import shutil
                shutil.rmtree(eval_path)
                print(f"  ✅ 已删除: {eval_dir}")
            except Exception as e:
                print(f"  ❌ 删除失败 {eval_dir}: {e}")


def view_training_config(project_root, train_dir):
    """查看特定训练的配置参数"""
    log_base = os.path.join(project_root, 'outputs', 'logs')
    config_path = os.path.join(log_base, train_dir, 'config_backup.yaml')
    log_path = os.path.join(log_base, train_dir, 'training.log')

    print(f"\n📋 训练配置详情: {train_dir}")
    print("="*80)

    # 显示配置文件
    if os.path.exists(config_path):
        print("📄 训练配置参数:")
        print("-" * 40)
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 显示关键配置
            print(f"训练轮数: {config['training']['epochs']}")
            print(f"批次大小: {config['dataloader']['batch_size']}")
            print(f"学习率: {config['training']['learning_rate']}")
            print(f"优化器: {config['training']['optimizer']}")
            print(f"学习率调度器: {config['training']['scheduler']['name']}")
            print(f"使用预训练: {config['model']['pretrained']}")
            print(f"类别数量: {config['data']['num_classes']}")
            print(f"输入尺寸: {config['data']['input_shape']}")

        except Exception as e:
            print(f"❌ 无法读取配置文件: {e}")
    else:
        print("❌ 配置文件不存在")

    # 显示训练日志摘要
    if os.path.exists(log_path):
        print(f"\n📝 训练日志摘要:")
        print("-" * 40)
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 提取关键信息
            for line in lines:
                if "最佳验证精度" in line or "训练完成" in line or "总训练时长" in line:
                    print(line.strip())

        except Exception as e:
            print(f"❌ 无法读取日志文件: {e}")
    else:
        print("❌ 日志文件不存在")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("🔧 训练历史管理工具")
    print("="*80)
    print("1. 查看训练历史")
    print("2. 比较评估结果")
    print("3. 查看训练配置")
    print("4. 清理旧结果")
    print("5. 退出")
    print("="*80)
    
    while True:
        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == '1':
            list_training_runs(project_root)
        elif choice == '2':
            compare_evaluations(project_root)
        elif choice == '3':
            # 查看训练配置
            checkpoint_base = os.path.join(project_root, 'models', 'checkpoints')
            if os.path.exists(checkpoint_base):
                train_dirs = [d for d in os.listdir(checkpoint_base)
                             if d.startswith('train_') and os.path.isdir(os.path.join(checkpoint_base, d))]
                train_dirs.sort(reverse=True)

                if train_dirs:
                    print("\n选择要查看的训练:")
                    for i, train_dir in enumerate(train_dirs[:10]):
                        print(f"  {i+1}. {train_dir}")

                    try:
                        idx = int(input("请输入序号: ").strip()) - 1
                        if 0 <= idx < len(train_dirs):
                            view_training_config(project_root, train_dirs[idx])
                        else:
                            print("❌ 无效序号")
                    except:
                        print("❌ 请输入有效数字")
                else:
                    print("❌ 没有找到训练记录")
            else:
                print("❌ 检查点目录不存在")

        elif choice == '4':
            keep_num = input("保留最近几个结果? (默认5): ").strip()
            try:
                keep_num = int(keep_num) if keep_num else 5
            except:
                keep_num = 5

            confirm = input(f"确认清理旧结果，保留最近{keep_num}个? (y/N): ").strip().lower()
            if confirm == 'y':
                clean_old_results(project_root, keep_num)
            else:
                print("❌ 已取消清理操作")
        elif choice == '5':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选择，请输入1-5")


if __name__ == "__main__":
    main()
