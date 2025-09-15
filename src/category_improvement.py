import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# 从你提供的报告中提取每个类别的F1分数
categories = [
    "灰雁", "大天鹅", "绿头鸭", "绿翅鸭", "灰山鹳", "西鹳鹳", "雄鸡", "红喉潜鸟", 
    "苍鹭", "普通鸬鹚", "苍鹰", "欧亚属", "西方秧鸡", "骨顶鸡", "黑翅长脚鹳", 
    "凤头麦鸡", "白腰草鹳", "红脚鹳", "林鹳", "麻雀"
]

baseline_f1 = [
    0.8546, 0.9426, 0.9258, 0.8772, 0.8571, 0.9683, 0.9061, 0.9249,
    0.9658, 0.8949, 0.9035, 0.8810, 0.9032, 0.9209, 0.9664,
    0.9508, 0.9439, 0.9008, 0.9183, 0.9217
]

model_c_f1 = [
    0.9369, 0.9551, 0.9700, 0.9153, 0.8889, 0.9862, 0.9576, 0.9677,
    0.9732, 0.9380, 0.9321, 0.8736, 0.9036, 0.9645, 0.9789,
    0.9675, 0.9352, 0.9277, 0.9486, 0.9296
]

# 计算Model C相对于Baseline的提升
improvement = [c - b for c, b in zip(model_c_f1, baseline_f1)]

# 创建类别和提升值的元组列表，并按提升值排序
category_improvement = list(zip(categories, improvement))
category_improvement.sort(key=lambda x: x[1], reverse=True)

# 分离排序后的类别和提升值
sorted_categories, sorted_improvement = zip(*category_improvement)

# 创建图表
plt.figure(figsize=(12, 10))

# 创建水平柱状图
bars = plt.barh(range(len(sorted_categories)), sorted_improvement, 
                color=['green' if x >= 0 else 'red' for x in sorted_improvement])

# 添加数值标签
for i, (category, value) in enumerate(zip(sorted_categories, sorted_improvement)):
    plt.text(value, i, f'{value:+.3f}', va='center', ha='left' if value >= 0 else 'right')

# 设置标题和标签
plt.title('各类别F1分数提升对比 (Model C vs Baseline)', fontsize=16, fontweight='bold')
plt.xlabel('F1分数提升', fontsize=14)
plt.ylabel('鸟类类别', fontsize=14)

# 设置y轴刻度
plt.yticks(range(len(sorted_categories)), sorted_categories)

# 添加参考线
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=np.mean(improvement), color='blue', linestyle='--', alpha=0.7, 
            label=f'平均提升: {np.mean(improvement):.3f}')

# 添加图例
plt.legend()

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('D:/paper/results/category_improvement.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印一些统计信息
print(f"平均提升: {np.mean(improvement):.4f}")
print(f"最大提升: {max(improvement):.4f} ({categories[improvement.index(max(improvement))]})")
print(f"最小提升: {min(improvement):.4f} ({categories[improvement.index(min(improvement))]})")
print(f"提升为正的类别数: {sum(1 for x in improvement if x > 0)}/{len(improvement)}")