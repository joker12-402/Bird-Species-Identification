# import os
# import librosa
# import matplotlib.pyplot as plt

# # 1. 统计基本信息
# data_path = 'D:/paper/Bird_Species_Identification/data/raw/BirdsData'
# bird_species = os.listdir(data_path)
# print(f"鸟类物种数量: {len(bird_species)}")
# for species in bird_species:
#     files = os.listdir(os.path.join(data_path, species))
#     print(f"物种 '{species}' 的样本数: {len(files)}")

# # 2. 随机挑选一个音频文件进行可视化
# example_file = 'D:/paper/Bird_Species_Identification/data/raw/BirdsData/0009/111651_1.wav' # 随机选择一个文件的路径
# y, sr = librosa.load(example_file)
# print(f"音频长度: {len(y)/sr} 秒, 采样率: {sr} Hz")

# # 绘制波形图
# plt.figure(figsize=(10, 4))
# librosa.display.waveshow(y, sr=sr)
# plt.title('Waveform')
# plt.tight_layout()
# plt.savefig('D:/paper/Bird_Species_Identification/results/waveform_example.png') # 保存到results文件夹
# plt.show()

import os
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np

# 1. 统计基本信息
data_path = 'D:/paper/data/raw/BirdsData'
bird_species = os.listdir(data_path)
print(f"鸟类物种数量: {len(bird_species)}")

# 统计每个物种的样本数并找出最少的样本数
min_samples = float('inf')
for species in bird_species:
    species_path = os.path.join(data_path, species)
    # 确保是目录而不是文件
    if os.path.isdir(species_path):
        files = [f for f in os.listdir(species_path) if f.endswith('.wav')]  # 只统计.wav文件
        print(f"物种 '{species}' 的样本数: {len(files)}")
        if len(files) < min_samples:
            min_samples = len(files)
    else:
        print(f"'{species}' 不是一个目录")

print(f"\n最少样本数: {min_samples} (这将影响后续的数据平衡)")

# 2. 随机挑选一个音频文件进行可视化
# 首先随机选择一个物种
random_species = random.choice(bird_species)
species_path = os.path.join(data_path, random_species)

# 确保是目录并且有文件
if os.path.isdir(species_path):
    # 获取该物种的所有.wav文件
    audio_files = [f for f in os.listdir(species_path) if f.endswith('.wav')]
    
    if audio_files:
        # 随机选择一个音频文件
        random_audio = random.choice(audio_files)
        example_file = os.path.join(species_path, random_audio)
        
        print(f"\n随机选择的文件: {example_file}")
        
        try:
            # 加载音频文件
            y, sr = librosa.load(example_file, sr=None)  # sr=None 保持原始采样率
            print(f"音频长度: {len(y)/sr:.2f} 秒, 采样率: {sr} Hz")
            
            # 绘制波形图
            plt.figure(figsize=(12, 8))
            
            # 子图1: 波形图
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform - {random_audio}')
            
            # 子图2: 频谱图
            plt.subplot(2, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
            plt.tight_layout()
            plt.savefig('D:/paper/results/waveform_and_spectrogram.png')
            plt.show()
            
        except Exception as e:
            print(f"加载文件时出错: {e}")
    else:
        print(f"物种 '{random_species}' 目录中没有找到.wav文件")
else:
    print(f"'{random_species}' 不是一个有效目录")