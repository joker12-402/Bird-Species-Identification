import json
from sklearn.model_selection import train_test_split
import os

# 配置
RAW_METADATA_PATH = 'D:/paper/data/features/metadata.json'  # 原始全量元数据路径
OUTPUT_VAL_METADATA_PATH = 'D:/paper/data/features/val_metadata.json'  # 验证集元数据输出路径
RANDOM_STATE = 42  # 固定随机种子确保可复现

# 1. 加载原始元数据
with open(RAW_METADATA_PATH, 'r', encoding='utf-8') as f:
    full_metadata = json.load(f)
all_indices = list(range(len(full_metadata)))
labels = [item['label'] for item in full_metadata]

# 2. 分层划分（70%训练集，15%验证集，15%测试集）
train_idx, temp_idx = train_test_split(all_indices, test_size=0.3, stratify=labels, random_state=RANDOM_STATE)
val_idx, _ = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=RANDOM_STATE)

# 3. 提取验证集元数据（仅包含file_path和label）
val_metadata = [
    {"file_path": item['file_path'].replace('.wav', '.wav'),  # 确保扩展名一致
     "label": item['label']}
    for i, item in enumerate(full_metadata) if i in val_idx
]

# 4. 保存验证集元数据（自动创建目录）
os.makedirs(os.path.dirname(OUTPUT_VAL_METADATA_PATH), exist_ok=True)
with open(OUTPUT_VAL_METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(val_metadata, f, ensure_ascii=False, indent=2)

print(f"成功生成验证集元数据：{len(val_metadata)} 条记录")
print(f"文件路径：{OUTPUT_VAL_METADATA_PATH}")