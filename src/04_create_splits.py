import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse

def create_data_splits(features_dir, output_dir, test_size=0.2, val_size=0.125, random_state=42):
    """
    创建训练集、验证集和测试集的划分
    
    参数:
        features_dir: 特征文件目录
        output_dir: 输出目录
        test_size: 测试集比例
        val_size: 验证集比例（占训练集的比例）
        random_state: 随机种子
    """
    # 加载元数据
    metadata_path = os.path.join(features_dir, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 加载标签映射
    mapping_path = os.path.join(features_dir, "label_mapping.json")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    # 提取文件路径和标签
    file_paths = [item["file_path"] for item in metadata]
    labels = [item["label"] for item in metadata]
    categories = [item["category"] for item in metadata]
    
    # 打印数据集统计信息
    print("数据集统计信息:")
    print(f"总样本数: {len(file_paths)}")
    print(f"类别数: {len(set(labels))}")
    
    # 计算类别分布
    label_counts = Counter(labels)
    print("\n类别分布:")
    for label, count in sorted(label_counts.items()):
        category_id = label_mapping["label_to_id"][str(label)]
        category_name = label_mapping["label_to_name"][str(label)]
        print(f"类别 {label} ({category_id}-{category_name}): {count} 个样本")
    
    # 计算不平衡比率
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n不平衡比率: {imbalance_ratio:.2f}:1")
    
    # 分层划分数据集
    # 第一次划分：分离测试集
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        file_paths, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    # 第二次划分：分离验证集
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, 
        test_size=val_size, 
        stratify=train_val_labels, 
        random_state=random_state
    )
    
    # 打印划分结果
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_files)} 个样本 ({len(train_files)/len(file_paths)*100:.1f}%)")
    print(f"验证集: {len(val_files)} 个样本 ({len(val_files)/len(file_paths)*100:.1f}%)")
    print(f"测试集: {len(test_files)} 个样本 ({len(test_files)/len(file_paths)*100:.1f}%)")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存划分结果
    def save_split(file_list, label_list, filename):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for file_path, label in zip(file_list, label_list):
                f.write(f"{file_path},{label}\n")
    
    save_split(train_files, train_labels, "train_split.txt")
    save_split(val_files, val_labels, "val_split.txt")
    save_split(test_files, test_labels, "test_split.txt")
    
    # 保存划分的统计信息
    split_info = {
        "total_samples": len(file_paths),
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "test_samples": len(test_files),
        "class_distribution": {
            "train": dict(Counter(train_labels)),
            "val": dict(Counter(val_labels)),
            "test": dict(Counter(test_labels))
        },
        "imbalance_ratio": imbalance_ratio
    }
    
    with open(os.path.join(output_dir, "split_info.json"), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n划分文件已保存到: {output_dir}")
    print("文件列表:")
    print("  - train_split.txt: 训练集文件列表")
    print("  - val_split.txt: 验证集文件列表")
    print("  - test_split.txt: 测试集文件列表")
    print("  - split_info.json: 划分统计信息")
    
    return train_files, val_files, test_files, train_labels, val_labels, test_labels

def verify_splits(features_dir, splits_dir):
    """
    验证划分结果是否正确
    """
    print("验证划分结果...")
    
    # 加载划分文件
    def load_split(filename):
        file_paths = []
        labels = []
        with open(os.path.join(splits_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    file_paths.append(parts[0])
                    labels.append(int(parts[1]))
        return file_paths, labels
    
    train_files, train_labels = load_split("train_split.txt")
    val_files, val_labels = load_split("val_split.txt")
    test_files, test_labels = load_split("test_split.txt")
    
    # 检查是否有重叠
    all_files = set(train_files + val_files + test_files)
    if len(all_files) != len(train_files) + len(val_files) + len(test_files):
        print("警告: 划分文件之间存在重叠!")
    
    # 检查特征文件是否存在
    missing_files = 0
    for file_path in all_files:
        full_path = os.path.join(features_dir, file_path)
        if not os.path.exists(full_path):
            print(f"缺失文件: {full_path}")
            missing_files += 1
    
    if missing_files == 0:
        print("所有特征文件都存在")
    else:
        print(f"警告: 共有 {missing_files} 个特征文件缺失")
    
    # 检查类别分布
    print("\n训练集类别分布:", dict(Counter(train_labels)))
    print("验证集类别分布:", dict(Counter(val_labels)))
    print("测试集类别分布:", dict(Counter(test_labels)))
    
    return missing_files == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建数据集划分')
    parser.add_argument('--features', type=str, default='D:/paper/data/features',
                        help='特征文件目录')
    parser.add_argument('--output', type=str, default='D:/paper/data/splits',
                        help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.125,
                        help='验证集比例（占训练集的比例）')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--verify', action='store_true',
                        help='创建完成后验证划分结果')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("=" * 50)
    print("创建数据集划分")
    print("=" * 50)
    print(f"特征目录: {args.features}")
    print(f"输出目录: {args.output}")
    print(f"测试集比例: {args.test_size}")
    print(f"验证集比例: {args.val_size}")
    print(f"随机种子: {args.random_state}")
    print("=" * 50)
    
    # 创建划分
    create_data_splits(
        features_dir=args.features,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # 验证划分结果
    if args.verify:
        verify_splits(args.features, args.output)

if __name__ == "__main__":
    main()