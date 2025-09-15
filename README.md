# Bird Species Identification

基于深度学习的鸟类声纹识别系统，使用多特征融合和注意力机制。

## 项目结构
Bird_Species_Identification/
├── src/ # 源代码目录
│ ├── model_150.py # 基线模型
│ ├── model_a_150.py # MFCC+时域模型的 Model A
│ ├── model_b_150.py # MFCC+能量模型的 Model B
│ ├── model_c_no_attention.py # 无注意力机制的 Model C
│ ├── 20.model_c_v8.py # 带注意力机制的 Model C（最佳）
│ ├── 21.model_c_v9.py # 多尺度卷积 + 双端注意力 Model C
│ ├── 22.model_c_v10.py # 简化版多尺度卷积 Model C
│ └── ... # 其他模型文件
├── data/ # 数据目录（数据量大，放到了和Bird_Species_Idenification同目录）
├── results/ # 实验结果（数据量大，放到了和Bird_Species_Idenification同目录）
└── README.md # 项目说明

text

## 模型特点

- **多特征融合**：结合 MFCC、时域特征和能量特征
- **注意力机制**：自适应加权不同特征的重要性
- **多尺度卷积**：提取不同尺度的时频特征

## 实验结果

在鸟类声纹数据集上取得了 94.78% 的准确率。

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 准备数据：将音频文件放入 `data/processed_audio` 目录
3. 运行训练：`python src/20.model_c_v8.py`

## 依赖库

- Python 3.6+
- PyTorch 1.8+
- Librosa
- Scikit-learn
