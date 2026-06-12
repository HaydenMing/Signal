import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# 数据
head = [1, 4, 8, 16, 32]
mAP = [70.2, 72.2, 77.0, 74.0, 77.0]
R_1 = [72.6, 74.3, 80.6, 76.1, 80.2]
R_5 = [81.8, 82.2, 89.5, 84.1, 88.3]
R10 = [87.4, 86.6, 91.9, 87.9, 91.5]

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
# 配色方案和标记样式
colors = ['#5ba6cf', '#f1c471', '#56c18c', '#e87e50']
markers = ['o', 's', '^', 'D']  # 不同形状的标记

# 创建画布
fig, ax = plt.subplots(figsize=(10, 6), dpi=600)

# 生成等距的x索引（0,1,2,3,4），确保x轴间距相等
x_indices = np.arange(len(head))

# 绘制折线图：同一head位置（同一x索引）绘制不同指标的折线
for i, (data, label, color, marker) in enumerate(zip(
        [mAP, R_1, R_5, R10],
        ['mAP', 'R-1', 'R-5', 'R10'],
        colors,
        markers
)):
    # 使用等距索引作为x坐标，保证同一head下x相同且整体等距
    ax.plot(x_indices, data, marker=marker, color=color, label=label,
            linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')

    # 添加数据点数值标签（同一head下垂直对齐）
    for idx, (x, y) in enumerate(zip(x_indices, data)):
        # 垂直偏移量，不同指标依次向上偏移，避免重叠
        vertical_offset = -15 - (i * 5)

        ax.annotate(f'{y:.1f}',
                    xy=(x, y),  # 标记位置（同一x索引，保证垂直对齐）
                    xytext=(x, y + vertical_offset / 72),  # 标签文本位置，垂直方向偏移
                    textcoords="data",
                    ha='center', va='top',
                    fontsize=15,
                    fontweight='bold')

# 设置x轴（使用等距索引作为刻度位置，显示原始head数值）
ax.set_xticks(x_indices)
ax.set_xticklabels(head, fontsize=15)  # 显示原始head数值但保持等距
ax.set_xlabel('Head', fontsize=15)

# 调整x轴范围，留出适当边距
ax.set_xlim(-0.5, len(head) - 0.5)

# 调整y轴
ax.set_ylim(68, 95)
y_major_locator = MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

# 设置y轴刻度标签
for tick in ax.get_yticklabels():
    tick.set_fontsize(15)

# 添加x轴和y轴网格
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# 图例设置 - 放置在右上角
legend = ax.legend(loc='upper right', fontsize=12)
for text in legend.get_texts():
    text.set_weight("bold")

# 优化布局
plt.tight_layout()

# 显示图表
plt.show()
