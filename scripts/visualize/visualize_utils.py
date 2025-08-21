"""
可视化工具函数
提供通用的可视化辅助功能
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from typing import Optional, Tuple


def setup_plot_style():
    """设置matplotlib绘图样式"""
    # 设置中文字体支持
    try:
        # 尝试设置中文字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        matplotlib.rcParams['axes.unicode_minus'] = False
        chinese_support = True
    except:
        # 如果中文字体不可用，使用英文
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        chinese_support = False
    
    # 设置绘图样式
    plt.style.use('default')
    matplotlib.rcParams['figure.figsize'] = (12, 8)
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    
    return chinese_support


def create_output_directory(base_dir: str, subdir: str = "") -> str:
    """
    创建输出目录
    
    Args:
        base_dir (str): 基础目录路径
        subdir (str): 子目录名称
        
    Returns:
        str: 创建的目录路径
    """
    if subdir:
        output_dir = os.path.join(base_dir, subdir)
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot_with_timestamp(fig, base_path: str, filename: str, 
                            show_plot: bool = True) -> str:
    """
    保存图表并可选择显示
    
    Args:
        fig: matplotlib图表对象
        base_path (str): 基础保存路径
        filename (str): 文件名（不含扩展名）
        show_plot (bool): 是否显示图表
        
    Returns:
        str: 保存的文件路径
    """
    # 确保目录存在
    os.makedirs(base_path, exist_ok=True)
    
    # 生成完整文件路径
    if not filename.endswith('.png'):
        filename += '.png'
    
    full_path = os.path.join(base_path, filename)
    
    # 保存图表
    fig.savefig(full_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"📊 图表已保存: {full_path}")
    
    # 可选择显示图表
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return full_path


def validate_image_path(image_path: str, project_root: str) -> Optional[str]:
    """
    验证图像路径是否有效
    
    Args:
        image_path (str): 图像路径
        project_root (str): 项目根目录
        
    Returns:
        str: 有效的绝对路径，如果无效返回None
    """
    # 如果是相对路径，转换为绝对路径
    if not os.path.isabs(image_path):
        full_path = os.path.join(project_root, image_path)
    else:
        full_path = image_path
    
    # 检查文件是否存在
    if os.path.exists(full_path):
        return full_path
    else:
        print(f"❌ 图像文件不存在: {full_path}")
        return None


def get_project_root() -> str:
    """
    获取项目根目录路径
    
    Returns:
        str: 项目根目录路径
    """
    # 从当前文件位置推断项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts/visualize/ -> scripts/ -> project_root/
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root


def print_tool_header(tool_name: str, description: str):
    """
    打印工具标题信息
    
    Args:
        tool_name (str): 工具名称
        description (str): 工具描述
    """
    print(f"🎨 {tool_name}")
    print("=" * 50)
    print(f"📝 {description}")
    print()


def print_completion_message(output_path: str, additional_info: str = ""):
    """
    打印完成信息
    
    Args:
        output_path (str): 输出文件路径
        additional_info (str): 额外信息
    """
    print(f"\n✅ 可视化完成！")
    print(f"📁 输出文件: {output_path}")
    if additional_info:
        print(f"💡 {additional_info}")


def format_number(num: float, precision: int = 2) -> str:
    """
    格式化数字显示
    
    Args:
        num (float): 要格式化的数字
        precision (int): 小数位数
        
    Returns:
        str: 格式化后的字符串
    """
    if abs(num) >= 1000:
        return f"{num:.{precision}e}"
    else:
        return f"{num:.{precision}f}"


def create_color_palette(n_colors: int) -> list:
    """
    创建颜色调色板
    
    Args:
        n_colors (int): 需要的颜色数量
        
    Returns:
        list: 颜色列表
    """
    if n_colors <= 10:
        # 使用预定义的颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        return colors[:n_colors]
    else:
        # 使用matplotlib的颜色映射
        cmap = plt.cm.get_cmap('tab20')
        return [cmap(i / n_colors) for i in range(n_colors)]


def add_watermark(fig, text: str = "Generated by VGG Training System"):
    """
    为图表添加水印
    
    Args:
        fig: matplotlib图表对象
        text (str): 水印文本
    """
    fig.text(0.99, 0.01, text, fontsize=8, alpha=0.5, 
             ha='right', va='bottom', style='italic')


def setup_subplot_layout(n_plots: int) -> Tuple[int, int]:
    """
    根据图表数量计算最佳的子图布局
    
    Args:
        n_plots (int): 图表数量
        
    Returns:
        Tuple[int, int]: (行数, 列数)
    """
    if n_plots <= 1:
        return (1, 1)
    elif n_plots <= 2:
        return (1, 2)
    elif n_plots <= 4:
        return (2, 2)
    elif n_plots <= 6:
        return (2, 3)
    elif n_plots <= 9:
        return (3, 3)
    else:
        # 对于更多图表，使用近似正方形布局
        cols = int(n_plots ** 0.5) + 1
        rows = (n_plots + cols - 1) // cols
        return (rows, cols)
