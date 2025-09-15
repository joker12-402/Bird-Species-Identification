import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（确保系统存在的字体）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

# 1. 加载四个模型的训练历史数据
model_paths = {
    "Baseline (MFCC)": "D:/paper/results/baseline_mfcc_150/training_history.json",
    "Model A (MFCC+Temporal)": "D:/paper/results/model_a_mfcc_temporal_150/training_history.json", 
    "Model B (MFCC+Energy)": "D:/paper/results/model_b_mfcc_energy_150/training_history.json",
    "Model C (MFCC+T+E)": "D:/paper/results/model_c_v8_150/training_history.json"
}

histories = {}
early_stop_epochs = {}  # 存储每个模型的早停epoch

for name, path in model_paths.items():
    with open(path, 'r', encoding='utf-8') as f:
        history = json.load(f)
        histories[name] = history
        early_stop_epochs[name] = len(history['train_loss'])

# 2. 确定统一的x轴范围
max_epoch = max(early_stop_epochs.values())

# 3. 创建图表
fig, axes = plt.subplots(2, 2, figsize=(18, 15))
fig.suptitle('模型训练过程对比', fontsize=18, fontweight='bold')

# 4. 样式定义
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', '-.', ':']
linewidths = [2.0, 2.0, 2.0, 2.5]

# 5. 绘制训练损失曲线
ax = axes[0, 0]
for i, (name, history) in enumerate(histories.items()):
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 
            label=name, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.axvline(x=early_stop_epochs[name], color=colors[i], linestyle=':', alpha=0.7)
ax.set_title('训练损失曲线')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xlim(1, max_epoch)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 6. 绘制验证损失曲线
ax = axes[0, 1]
for i, (name, history) in enumerate(histories.items()):
    epochs = range(1, len(history['val_loss']) + 1)
    ax.plot(epochs, history['val_loss'], 
            label=name, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.axvline(x=early_stop_epochs[name], color=colors[i], linestyle=':', alpha=0.7)
ax.set_title('验证损失曲线')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xlim(1, max_epoch)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 7. 绘制验证准确率曲线
ax = axes[1, 0]
for i, (name, history) in enumerate(histories.items()):
    epochs = range(1, len(history['val_acc']) + 1)
    ax.plot(epochs, history['val_acc'], 
            label=name, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.axvline(x=early_stop_epochs[name], color=colors[i], linestyle=':', alpha=0.7)
    final_acc = history['val_acc'][-1]
    # 调整文本位置，避免超出坐标轴
    text_x = min(early_stop_epochs[name] + 2, max_epoch)
    ax.text(text_x, final_acc, f'{final_acc:.4f}', 
            color=colors[i], va='bottom' if i < 2 else 'top')
ax.set_title('验证准确率曲线')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_xlim(1, max_epoch)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 8. 绘制验证F1分数曲线
ax = axes[1, 1]
for i, (name, history) in enumerate(histories.items()):
    epochs = range(1, len(history['val_f1']) + 1)
    ax.plot(epochs, history['val_f1'], 
            label=name, color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.axvline(x=early_stop_epochs[name], color=colors[i], linestyle=':', alpha=0.7)
    final_f1 = history['val_f1'][-1]
    # 调整文本位置，确保在图表内显示
    text_x = min(early_stop_epochs[name] + 2, max_epoch)
    ax.text(text_x, final_f1, f'{final_f1:.4f}', 
            color=colors[i], 
            va='bottom' if final_f1 < 0.945 else 'top')
ax.set_title('验证F1分数曲线')
ax.set_xlabel('Epoch')
ax.set_ylabel('F1 Score')
ax.set_xlim(1, max_epoch)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 9. 调整整体布局，增加右侧间距
plt.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.08, wspace=0.25, hspace=0.3)
plt.savefig('D:/paper/results/model_comparison_curves.png', dpi=300, bbox_inches='tight')
plt.show()