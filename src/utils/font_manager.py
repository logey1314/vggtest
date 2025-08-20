"""
字体管理模块
提供统一的matplotlib中文字体设置和管理功能
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

class FontManager:
    """字体管理器类"""
    
    def __init__(self):
        self.has_chinese_font = False
        self.current_font = None
        self.system = platform.system()
        self._setup_font()
    
    def _setup_font(self):
        """设置字体"""
        # 定义各平台的中文字体
        if self.system == "Windows":
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun']
        elif self.system == "Darwin":  # macOS
            chinese_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
        else:  # Linux
            chinese_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 尝试设置中文字体
        for font in chinese_fonts:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                self.has_chinese_font = True
                self.current_font = font
                return
        
        # 如果没有找到中文字体，使用默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.has_chinese_font = False
        self.current_font = 'DejaVu Sans'
    
    def get_font_status(self):
        """获取字体状态信息"""
        return {
            'has_chinese_font': self.has_chinese_font,
            'current_font': self.current_font,
            'system': self.system
        }
    
    def print_font_status(self):
        """打印字体状态"""
        if self.has_chinese_font:
            print(f"✅ 使用中文字体: {self.current_font}")
        else:
            print(f"⚠️  未检测到中文字体，使用: {self.current_font}")
            print("💡 图表将使用英文标题以确保正常显示")

# 全局字体管理器实例
_font_manager = None

def setup_matplotlib_font():
    """
    设置matplotlib字体
    
    Returns:
        bool: 是否支持中文字体
    """
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager.has_chinese_font

def get_font_manager():
    """
    获取字体管理器实例
    
    Returns:
        FontManager: 字体管理器实例
    """
    global _font_manager
    if _font_manager is None:
        _font_manager = FontManager()
    return _font_manager

def has_chinese_font():
    """
    检查是否支持中文字体
    
    Returns:
        bool: 是否支持中文字体
    """
    return get_font_manager().has_chinese_font

def get_text_labels(label_type, use_english=None):
    """
    获取文本标签（中文或英文）
    
    Args:
        label_type (str): 标签类型
        use_english (bool): 是否使用英文，None表示自动检测
        
    Returns:
        dict: 标签字典
    """
    if use_english is None:
        use_english = not has_chinese_font()
    
    # 定义标签字典
    labels = {
        'confusion_matrix': {
            'title': 'Confusion Matrix' if use_english else '混淆矩阵',
            'title_normalized': 'Normalized Confusion Matrix' if use_english else '归一化混淆矩阵',
            'xlabel': 'Predicted Label' if use_english else '预测标签',
            'ylabel': 'True Label' if use_english else '真实标签',
            'count_label': 'Sample Count' if use_english else '样本数量',
            'percentage_label': 'Percentage' if use_english else '百分比'
        },
        'error_analysis': {
            'class_errors': 'Class Error Count' if use_english else '各类别错误数量',
            'error_types': 'Error Type Distribution' if use_english else '错误类型分布',
            'confidence_dist': 'Error Sample Confidence Distribution' if use_english else '错误样本置信度分布',
            'error_matrix': 'Error Transfer Matrix' if use_english else '错误转移矩阵',
            'confidence': 'Confidence' if use_english else '置信度',
            'frequency': 'Frequency' if use_english else '频次',
            'predicted_label': 'Predicted Label' if use_english else '预测标签',
            'true_label': 'True Label' if use_english else '真实标签',
            'adjacent_errors': 'Adjacent Errors' if use_english else '相邻错误',
            'severe_errors': 'Severe Errors' if use_english else '严重错误',
            'error_samples': 'Error Sample Visualization' if use_english else '错误样本可视化',
            'no_confidence': 'No Confidence Info' if use_english else '无置信度信息',
            'mean': 'Mean' if use_english else '平均值'
        },
        'performance': {
            'overall_performance': 'Overall Performance Metrics' if use_english else '整体性能指标',
            'class_precision': 'Class Precision' if use_english else '各类别精确率',
            'class_recall': 'Class Recall' if use_english else '各类别召回率',
            'class_f1': 'Class F1-Score' if use_english else '各类别F1-Score',
            'accuracy': 'Accuracy' if use_english else '准确率',
            'precision': 'Precision' if use_english else '精确率',
            'recall': 'Recall' if use_english else '召回率',
            'f1_score': 'F1-Score' if use_english else 'F1分数',
            'score': 'Score' if use_english else '分数'
        }
    }
    
    return labels.get(label_type, {})

def print_font_installation_guide():
    """打印字体安装指南"""
    system = platform.system()
    print("\n💡 中文字体安装指南:")
    print("=" * 40)
    
    if system == "Linux":
        print("Ubuntu/Debian:")
        print("  sudo apt-get install fonts-wqy-microhei fonts-noto-cjk")
        print("\nCentOS/RHEL:")
        print("  sudo yum install wqy-microhei-fonts")
        print("\nConda:")
        print("  conda install -c conda-forge fonts-conda-ecosystem")
    elif system == "Darwin":
        print("macOS 通常已内置中文字体")
        print("如果显示异常，请检查系统字体设置")
    elif system == "Windows":
        print("Windows 通常已内置中文字体")
        print("如果显示异常，请检查系统字体设置")
    else:
        print("请根据您的操作系统安装相应的中文字体包")
    
    print("=" * 40)
