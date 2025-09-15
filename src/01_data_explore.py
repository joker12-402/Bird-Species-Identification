# import os
# import random
# import librosa
# import matplotlib.pyplot as plt
# import numpy as np

# plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题


# # 1. 统计基本信息
# data_path = 'D:/paper/data/raw/BirdsData'
# bird_species = os.listdir(data_path)
# print(f"鸟类物种数量: {len(bird_species)}")

# # 统计每个物种的样本数并找出最少的样本数
# min_samples = float('inf')
# for species in bird_species:
#     species_path = os.path.join(data_path, species)
#     # 确保是目录而不是文件
#     if os.path.isdir(species_path):
#         files = [f for f in os.listdir(species_path) if f.endswith('.wav')]  # 只统计.wav文件
#         print(f"物种 '{species}' 的样本数: {len(files)}")
#         if len(files) < min_samples:
#             min_samples = len(files)
#     else:
#         print(f"'{species}' 不是一个目录")

# print(f"\n最少样本数: {min_samples} (这将影响后续的数据平衡)")

# # 新增：批量检查音频时长
# print("\n正在检查音频时长分布...")
# durations = []
# for species in bird_species[:3]:  # 只检查前3个物种以节省时间
#     species_path = os.path.join(data_path, species)
#     if os.path.isdir(species_path):
#         audio_files = [f for f in os.listdir(species_path) if f.endswith('.wav')][:10]  # 每个物种检查10个文件
#         for audio_file in audio_files:
#             file_path = os.path.join(species_path, audio_file)
#             try:
#                 y, sr = librosa.load(file_path, sr=None)
#                 duration = len(y) / sr
#                 durations.append(duration)
#             except Exception as e:
#                 print(f"加载文件 {file_path} 时出错: {e}")

# # 分析时长分布
# if durations:
#     print(f"\n音频时长分析 (基于 {len(durations)} 个样本):")
#     print(f"最短时长: {min(durations):.2f} 秒")
#     print(f"最长时长: {max(durations):.2f} 秒")
#     print(f"平均时长: {np.mean(durations):.2f} 秒")
#     print(f"时长标准差: {np.std(durations):.2f} 秒")
    
#     # 绘制时长分布直方图
#     plt.figure(figsize=(10, 6))
#     plt.hist(durations, bins=20, edgecolor='black')
#     plt.xlabel('时长 (秒)')
#     plt.ylabel('频数')
#     plt.title('音频时长分布')
#     plt.savefig('D:/paper/results/duration_distribution.png')
#     plt.show()
# else:
#     print("无法获取音频时长信息")

# # 2. 随机挑选一个音频文件进行可视化
# # 首先随机选择一个物种
# random_species = random.choice(bird_species)
# species_path = os.path.join(data_path, random_species)

# # 确保是目录并且有文件
# if os.path.isdir(species_path):
#     # 获取该物种的所有.wav文件
#     audio_files = [f for f in os.listdir(species_path) if f.endswith('.wav')]
    
#     if audio_files:
#         # 随机选择一个音频文件
#         random_audio = random.choice(audio_files)
#         example_file = os.path.join(species_path, random_audio)
        
#         print(f"\n随机选择的文件: {example_file}")
        
#         try:
#             # 加载音频文件
#             y, sr = librosa.load(example_file, sr=None)  # sr=None 保持原始采样率
#             duration = len(y)/sr
#             print(f"音频长度: {duration:.2f} 秒, 采样率: {sr} Hz")
            
#             # 绘制波形图
#             plt.figure(figsize=(12, 8))
            
#             # 子图1: 波形图
#             plt.subplot(2, 1, 1)
#             librosa.display.waveshow(y, sr=sr)
#             plt.title(f'Waveform - {random_audio} ({duration:.2f}s)')
            
#             # 子图2: 频谱图
#             plt.subplot(2, 1, 2)
#             D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
#             librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
#             plt.colorbar(format='%+2.0f dB')
#             plt.title('Spectrogram')
            
#             plt.tight_layout()
#             plt.savefig('D:/paper/results/waveform_and_spectrogram.png')
#             plt.show()
            
#         except Exception as e:
#             print(f"加载文件时出错: {e}")
#     else:
#         print(f"物种 '{random_species}' 目录中没有找到.wav文件")
# else:
#     print(f"'{random_species}' 不是一个有效目录")

import os
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体支持
try:
    # 尝试使用系统中已有的中文字体
    font_list = [f.name for f in font_manager.fontManager.ttflist if '宋体' in f.name or 'SimHei' in f.name or 'Microsoft' in f.name]
    if font_list:
        plt.rcParams['font.sans-serif'] = [font_list[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能显示为方框")

# 1. 统计基本信息
data_path = 'D:/paper/data/raw/BirdsData'
bird_species = os.listdir(data_path)
print(f"鸟类物种数量: {len(bird_species)}")

# 统计每个物种的样本数并找出最少的样本数
min_samples = float('inf')
sample_counts = {}
for species in bird_species:
    species_path = os.path.join(data_path, species)
    # 确保是目录而不是文件
    if os.path.isdir(species_path):
        files = [f for f in os.listdir(species_path) if f.endswith('.wav')]  # 只统计.wav文件
        sample_count = len(files)
        print(f"物种 '{species}' 的样本数: {sample_count}")
        sample_counts[species] = sample_count
        if sample_count < min_samples:
            min_samples = sample_count
    else:
        print(f"'{species}' 不是一个目录")

print(f"\n最少样本数: {min_samples} (这将影响后续的数据平衡)")

# 新增：批量检查音频时长和采样率
print("\n正在检查音频时长和采样率分布...")
durations = []
sample_rates = []

# 检查更多样本以获得更全面的统计
for species in bird_species[:5]:  # 检查前5个物种
    species_path = os.path.join(data_path, species)
    if os.path.isdir(species_path):
        audio_files = [f for f in os.listdir(species_path) if f.endswith('.wav')][:15]  # 每个物种检查15个文件
        for audio_file in audio_files:
            file_path = os.path.join(species_path, audio_file)
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = len(y) / sr
                durations.append(duration)
                sample_rates.append(sr)
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")

# 分析时长和采样率分布
if durations:
    print(f"\n音频时长分析 (基于 {len(durations)} 个样本):")
    print(f"最短时长: {min(durations):.2f} 秒")
    print(f"最长时长: {max(durations):.2f} 秒")
    print(f"平均时长: {np.mean(durations):.2f} 秒")
    print(f"时长标准差: {np.std(durations):.4f} 秒")
    
    print(f"\n采样率分析:")
    unique_rates = np.unique(sample_rates)
    print(f"发现 {len(unique_rates)} 种不同的采样率: {unique_rates}")
    
    # 绘制时长分布直方图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(durations, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.title('Audio Duration Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(sample_rates, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Sample Rate (Hz)')
    plt.ylabel('Count')
    plt.title('Sample Rate Distribution')
    
    plt.tight_layout()
    plt.savefig('D:/paper/results/audio_properties.png')
    plt.show()
else:
    print("无法获取音频属性信息")

# 2. 随机挑选一个音频文件进行详细分析
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
        
        print(f"\n详细分析文件: {example_file}")
        
        try:
            # 加载音频文件
            y, sr = librosa.load(example_file, sr=None)
            duration = len(y)/sr
            print(f"音频长度: {duration:.6f} 秒, 采样率: {sr} Hz")
            
            # 计算更多音频属性
            rms = librosa.feature.rms(y=y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            print(f"RMS能量范围: {np.min(rms):.4f} - {np.max(rms):.4f}")
            print(f"频谱质心范围: {np.min(spectral_centroid):.2f} - {np.max(spectral_centroid):.2f} Hz")
            
            # 绘制详细分析图
            plt.figure(figsize=(15, 10))
            
            # 子图1: 波形图
            plt.subplot(3, 1, 1)
            librosa.display.waveshow(y, sr=sr)
            plt.title(f'Waveform - {random_audio} ({duration:.6f}s, {sr}Hz)')
            
            # 子图2: 频谱图
            plt.subplot(3, 1, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            
            # 子图3: 频谱质心
            plt.subplot(3, 1, 3)
            times = librosa.times_like(spectral_centroid, sr=sr)
            plt.plot(times, spectral_centroid, color='r')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectral Centroid')
            plt.ylim([0, sr/2])
            
            plt.tight_layout()
            plt.savefig('D:/paper/results/detailed_audio_analysis.png')
            plt.show()
            
        except Exception as e:
            print(f"分析文件时出错: {e}")
    else:
        print(f"物种 '{random_species}' 目录中没有找到.wav文件")
else:
    print(f"'{random_species}' 不是一个有效目录")

# 3. 数据不平衡分析
print("\n数据不平衡分析:")
# 绘制样本数量分布图
plt.figure(figsize=(12, 6))
species_ids = list(sample_counts.keys())
counts = list(sample_counts.values())

plt.bar(range(len(species_ids)), counts)
plt.xlabel('Species ID')
plt.ylabel('Sample Count')
plt.title('Sample Distribution Across Species')
plt.xticks(range(len(species_ids)), species_ids, rotation=45)
plt.tight_layout()
plt.savefig('D:/paper/results/sample_distribution.png')
plt.show()

# 计算不平衡比率
max_count = max(counts)
min_count = min(counts)
imbalance_ratio = max_count / min_count
print(f"最多样本数: {max_count}")
print(f"最少样本数: {min_count}")
print(f"不平衡比率: {imbalance_ratio:.2f}:1")

# 识别样本数最少的类别
min_species = [s for s, c in sample_counts.items() if c == min_count]
print(f"样本数最少的类别: {min_species} (各{min_count}个样本)")