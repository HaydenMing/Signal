import matplotlib.pyplot as plt
import numpy as np

# 数据
reserved_tokens = [0.1, 0.15, 0.2, 0.25, 0.3]  # 按升序排列
# za.py 中的数据
mAP_za = [75.2, 74.5, 79.0, 72.9, 77.3]
R_1_za = [78.5, 76.3, 82.8, 76.8, 79.8]
# Line_Gra.py 中的数据
mAP_line = [74.5, 76.1, 80.3, 77.2, 78.0]
R_1_line = [76.8, 79.3, 85.2, 80.9, 80.1]

# 设置字体为粗体
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# 设置柱状图宽度（减小宽度避免重叠）
bar_width = 0.15  # 原0.35过宽，导致四组柱子无法紧凑排列
index = np.arange(len(reserved_tokens))

# 浅色系配色（保持学术风格）
colors_za = ['#d0d1e6', '#3690c0']  # 浅紫到深蓝渐变
colors_line = ['#a1d99b', '#31a354']  # 绿色系

# 创建高分辨率画布
fig, ax = plt.subplots(figsize=(10, 5), dpi=600)

# 添加网格背景
ax.grid(axis='y', linestyle='--', alpha=1.0)  # 仅添加y轴网格，虚线样式，适当透明
ax.grid(axis='x', linestyle='--', alpha=1.0)  # 仅添加y轴网格，虚线样式，适当透明

# 绘制 za.py 中的各指标柱状图（围绕刻度左半部分排列）
for i, (data, label, color) in enumerate(zip([mAP_za, R_1_za],
                                             ['α:mAP', 'α:R-1'],
                                             colors_za)):
    # 四组柱子总宽度为 4*bar_width，左半部分偏移量为负，右半为正
    position = index - 1.5 * bar_width + i * bar_width  # 修正位置计算
    bars = ax.bar(position, data, width=bar_width, color=color, label=label)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14,  # 适当减小字体避免拥挤
                    fontweight='bold')

# 绘制 Line_Gra.py 中的各指标柱状图（围绕刻度右半部分排列）
for i, (data, label, color) in enumerate(zip([mAP_line, R_1_line],
                                             ['β:mAP', 'β:R-1'],
                                             colors_line)):
    # 右半部分从 0.5*bar_width 开始偏移，与左半部分衔接
    position = index + 0.5 * bar_width + i * bar_width  # 修正位置计算
    bars = ax.bar(position, data, width=bar_width, color=color, label=label)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14,
                    fontweight='bold')

# 设置坐标轴
ax.set_xticks(index)
ax.set_xticklabels(reserved_tokens, fontsize=14)
ax.set_ylim(71, 88)  # 略微扩大y轴范围，避免标签顶到边界

# 设置y轴刻度字体
for tick in ax.get_yticklabels():
    tick.set_fontsize(14)

# 图例设置
legend = ax.legend(fontsize=14)
for text in legend.get_texts():
    text.set_weight("bold")
    text.set_fontsize(18)

# 优化布局
plt.tight_layout()
plt.show()
