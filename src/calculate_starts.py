# calculate_stats.py
import os
import numpy as np
import json
from tqdm import tqdm

def calculate_feature_stats(split_file, features_dir):
    """计算训练集特征的均值和标准差"""
    with open(split_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_features = []
    print("正在计算特征统计量...")
    
    for line in tqdm(lines):
        parts = line.strip().split(',')
        if len(parts) == 2:
            file_path = parts[0]
            feature_path = os.path.join(features_dir, file_path)
            try:
                feature = np.load(feature_path)
                all_features.append(feature)
            except:
                continue
    
    all_features = np.array(all_features)
    mean = np.mean(all_features)
    std = np.std(all_features)
    
    print(f"全局均值: {mean:.6f}")
    print(f"全局标准差: {std:.6f}")
    print(f"特征值范围: [{np.min(all_features):.6f}, {np.max(all_features):.6f}]")
    
    # 保存统计量
    stats = {
        'mean': float(mean),
        'std': float(std),
        'min': float(np.min(all_features)),
        'max': float(np.max(all_features))
    }
    
    with open('feature_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    return mean, std

if __name__ == "__main__":
    split_file = "D:/paper/data/splits/train_split.txt"
    features_dir = "D:/paper/data/features"
    calculate_feature_stats(split_file, features_dir)