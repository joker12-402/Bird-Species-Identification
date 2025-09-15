# 检查特征样本
import numpy as np
feature_path = "D:/paper/data/features/0009/115524_2.npy"  # 示例路径
feature = np.load(feature_path)
print(f"特征形状: {feature.shape}")  # 应为 (180, 100)
print(f"特征范围: min={np.min(feature)}, max={np.max(feature)}")