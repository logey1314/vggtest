"""
中文字体支持测试脚本
用于检测系统是否支持中文字体显示
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from src.utils.font_manager import setup_matplotlib_font, get_font_manager, print_font_installation_guide

def test_chinese_font():
    """测试中文字体支持"""
    print("🔤 中文字体支持测试")
    print("=" * 50)

    # 使用统一的字体管理器
    has_chinese = setup_matplotlib_font()
    font_manager = get_font_manager()

    # 显示字体状态
    status = font_manager.get_font_status()
    print(f"操作系统: {status['system']}")
    print(f"当前字体: {status['current_font']}")
    print(f"中文支持: {'是' if status['has_chinese_font'] else '否'}")

    if has_chinese:
        print(f"\n✅ 支持中文字体")
        print("📊 图表将正常显示中文标题")
        
        # 测试绘制中文图表
        try:
            plt.figure(figsize=(8, 6))
            plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'b-', linewidth=2)
            plt.title('中文字体测试 - 训练损失曲线', fontsize=14, fontweight='bold')
            plt.xlabel('训练轮次')
            plt.ylabel('损失值')
            plt.grid(True, alpha=0.3)

            test_output_dir = "outputs/plots/test"
            os.makedirs(test_output_dir, exist_ok=True)
            save_path = os.path.join(test_output_dir, "chinese_font_test.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 测试图表已保存: {save_path}")
            print("请查看图表确认中文显示是否正常")

        except Exception as e:
            print(f"⚠️  绘制测试图表时出错: {e}")

    else:
        print(f"\n❌ 不支持中文字体")
        print("📊 图表将使用英文标题")
        print_font_installation_guide()
    
    print("\n" + "=" * 50)
    return len(found_fonts) > 0

if __name__ == "__main__":
    test_chinese_font()
