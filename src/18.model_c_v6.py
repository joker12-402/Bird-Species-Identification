import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter
import random
import time
import torchvision.models as models
import librosa
import scipy.signal as signal
from scipy.fftpack import dct
from scipy import ndimage
import seaborn as sns

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# ------------------------------------------------------------------------------
# 1. 修复的特征注意力机制
# ------------------------------------------------------------------------------
class FeatureAttention(nn.Module):
    """特征注意力机制，用于加权不同特征通道的重要性"""
    def __init__(self, num_features, reduction_ratio=4):
        super(FeatureAttention, self).__init__()
        self.num_features = num_features
        self.reduction_ratio = reduction_ratio
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 注意力机制 - 修复实现
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_ratio, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, num_features, height, width]
        batch_size, num_features, height, width = x.size()
        
        # 全局平均池化
        gap = self.global_avg_pool(x)  # [batch_size, num_features, 1, 1]
        gap = gap.view(batch_size, num_features)  # [batch_size, num_features]
        
        # 计算注意力权重
        attention_weights = self.attention(gap)  # [batch_size, num_features]
        attention_weights = attention_weights.view(batch_size, num_features, 1, 1)  # [batch_size, num_features, 1, 1]
        
        # 应用注意力权重
        weighted_features = x * attention_weights
        
        return weighted_features, attention_weights

# ------------------------------------------------------------------------------
# 2. 改进的模型架构 - 添加Label Smoothing和更多正则化
# ------------------------------------------------------------------------------
class ImprovedBirdNetWithAttention(nn.Module):
    def __init__(self, num_classes, in_channels=3, label_smoothing=0.1):
        super(ImprovedBirdNetWithAttention, self).__init__()
        self.label_smoothing = label_smoothing
        
        # 特征注意力机制
        self.feature_attention = FeatureAttention(in_channels)
        
        # 增强的特征提取网络
        self.features = nn.Sequential(
            # 输入: (in_channels, 128, 128)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 64, 64)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (128, 32, 32)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (256, 16, 16)
            
            # 添加额外卷积层
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (512, 8, 8)
            
            # 添加额外卷积层
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 添加空间注意力
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 增强的分类器，增加Dropout防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 适度Dropout
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 适度Dropout
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 应用特征注意力
        x, attention_weights = self.feature_attention(x)
        
        # 特征提取
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x, attention_weights

    def loss_fn(self, outputs, targets):
        """带Label Smoothing的损失函数"""
        log_probs = F.log_softmax(outputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        return loss.mean()

# ------------------------------------------------------------------------------
# 3. 增强的特征提取函数 - 添加Delta MFCC特征
# ------------------------------------------------------------------------------
def extract_mfcc(audio_path, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512):
    """提取MFCC特征及其一阶二阶差分"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 预加重
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # 提取MFCC
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=sr//2
        )
        
        # 提取一阶差分
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 提取二阶差分
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return mfcc, mfcc_delta, mfcc_delta2
    except Exception as e:
        print(f"提取MFCC特征失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mfcc, 100)), np.zeros((n_mfcc, 100)), np.zeros((n_mfcc, 100))

def extract_temporal_features(audio_path, sr=16000, hop_length=512):
    """提取时域特征（短时能量）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 计算短时能量 (Root Mean Square Energy)
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # 计算过零率
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return rmse, zcr
    except Exception as e:
        print(f"提取时域特征失败: {audio_path}, 错误: {e}")
        return np.zeros(100), np.zeros(100)

def extract_energy_features(audio_path, sr=16000, n_mels=40, n_fft=2048, hop_length=512):
    """提取能量特征（梅尔声谱图）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 提取梅尔声谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=sr//2
        )
        
        # 转换为分贝尺度
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 计算频谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        
        return mel_spec_db, spectral_centroid
    except Exception as e:
        print(f"提取能量特征失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mels, 100)), np.zeros(100)

# ------------------------------------------------------------------------------
# 4. 数据集类 - 增强版，支持更多特征
# ------------------------------------------------------------------------------
class EnhancedBirdSoundDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_size=(128, 128), 
                 is_train=False, augment_prob=0.3):  # 降低数据增强概率
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob
        
        # 提取所有标签用于计算类别权重
        self.labels = [item['label'] for item in self.metadata]
        self.num_classes = len(set(self.labels))
        
        # 预计算特征统计量
        self._compute_feature_stats()
        
    def _compute_feature_stats(self):
        """计算特征统计量"""
        print("计算特征统计量...")
        all_mfcc_features = []
        all_mfcc_delta_features = []
        all_mfcc_delta2_features = []
        all_temporal_features = []
        all_zcr_features = []
        all_energy_features = []
        all_spectral_centroid_features = []
        
        # 随机采样部分数据计算统计量
        sample_size = min(100, len(self.metadata))
        sample_indices = np.random.choice(len(self.metadata), sample_size, replace=False)
        
        for idx in tqdm(sample_indices, desc="计算统计量"):
            mfcc, mfcc_delta, mfcc_delta2, temporal, zcr, energy, spectral_centroid = self._extract_features(idx)
            if mfcc is not None:
                all_mfcc_features.append(mfcc)
                all_mfcc_delta_features.append(mfcc_delta)
                all_mfcc_delta2_features.append(mfcc_delta2)
                all_temporal_features.append(temporal)
                all_zcr_features.append(zcr)
                all_energy_features.append(energy)
                all_spectral_centroid_features.append(spectral_centroid)
        
        if not all_mfcc_features:
            # 使用默认值
            self.mfcc_mean = 0.0
            self.mfcc_std = 1.0
            self.mfcc_delta_mean = 0.0
            self.mfcc_delta_std = 1.0
            self.mfcc_delta2_mean = 0.0
            self.mfcc_delta2_std = 1.0
            self.temporal_mean = 0.0
            self.temporal_std = 1.0
            self.zcr_mean = 0.0
            self.zcr_std = 1.0
            self.energy_mean = 0.0
            self.energy_std = 1.0
            self.spectral_centroid_mean = 0.0
            self.spectral_centroid_std = 1.0
            print("无法提取任何特征，使用默认统计量: 均值=0.0, 标准差=1.0")
            return
        
        # 计算全局均值和标准差
        all_mfcc_features = np.vstack(all_mfcc_features)
        self.mfcc_mean = np.mean(all_mfcc_features)
        self.mfcc_std = np.std(all_mfcc_features)
        
        all_mfcc_delta_features = np.vstack(all_mfcc_delta_features)
        self.mfcc_delta_mean = np.mean(all_mfcc_delta_features)
        self.mfcc_delta_std = np.std(all_mfcc_delta_features)
        
        all_mfcc_delta2_features = np.vstack(all_mfcc_delta2_features)
        self.mfcc_delta2_mean = np.mean(all_mfcc_delta2_features)
        self.mfcc_delta2_std = np.std(all_mfcc_delta2_features)
        
        all_temporal_features = np.vstack(all_temporal_features)
        self.temporal_mean = np.mean(all_temporal_features)
        self.temporal_std = np.std(all_temporal_features)
        
        all_zcr_features = np.vstack(all_zcr_features)
        self.zcr_mean = np.mean(all_zcr_features)
        self.zcr_std = np.std(all_zcr_features)
        
        all_energy_features = np.vstack(all_energy_features)
        self.energy_mean = np.mean(all_energy_features)
        self.energy_std = np.std(all_energy_features)
        
        all_spectral_centroid_features = np.vstack(all_spectral_centroid_features)
        self.spectral_centroid_mean = np.mean(all_spectral_centroid_features)
        self.spectral_centroid_std = np.std(all_spectral_centroid_features)
        
        print(f"MFCC统计量: 均值={self.mfcc_mean:.4f}, 标准差={self.mfcc_std:.4f}")
        print(f"MFCC Delta统计量: 均值={self.mfcc_delta_mean:.4f}, 标准差={self.mfcc_delta_std:.4f}")
        print(f"MFCC Delta2统计量: 均值={self.mfcc_delta2_mean:.4f}, 标准差={self.mfcc_delta2_std:.4f}")
        print(f"时域特征统计量: 均值={self.temporal_mean:.4f}, 标准差={self.temporal_std:.4f}")
        print(f"过零率统计量: 均值={self.zcr_mean:.4f}, 标准差={self.zcr_std:.4f}")
        print(f"能量特征统计量: 均值={self.energy_mean:.4f}, 标准差={self.energy_std:.4f}")
        print(f"频谱质心统计量: 均值={self.spectral_centroid_mean:.4f}, 标准差={self.spectral_centroid_std:.4f}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 提取特征
        mfcc, mfcc_delta, mfcc_delta2, temporal, zcr, energy, spectral_centroid = self._extract_features(idx)
        
        # 特征标准化
        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        mfcc_delta = (mfcc_delta - self.mfcc_delta_mean) / self.mfcc_delta_std
        mfcc_delta2 = (mfcc_delta2 - self.mfcc_delta2_mean) / self.mfcc_delta2_std
        temporal = (temporal - self.temporal_mean) / self.temporal_std
        zcr = (zcr - self.zcr_mean) / self.zcr_std
        energy = (energy - self.energy_mean) / self.energy_std
        spectral_centroid = (spectral_centroid - self.spectral_centroid_mean) / self.spectral_centroid_std
        
        # 调整为固定尺寸
        mfcc = self._resize_features(mfcc, self.target_size)
        mfcc_delta = self._resize_features(mfcc_delta, self.target_size)
        mfcc_delta2 = self._resize_features(mfcc_delta2, self.target_size)
        temporal = self._resize_features(temporal, self.target_size)
        zcr = self._resize_features(zcr, self.target_size)
        energy = self._resize_features(energy, self.target_size)
        spectral_centroid = self._resize_features(spectral_centroid, self.target_size)
        
        # 确保所有特征形状相同
        target_shape = mfcc.shape
        mfcc_delta = np.resize(mfcc_delta, target_shape)
        mfcc_delta2 = np.resize(mfcc_delta2, target_shape)
        temporal = np.resize(temporal, target_shape)
        zcr = np.resize(zcr, target_shape)
        energy = np.resize(energy, target_shape)
        spectral_centroid = np.resize(spectral_centroid, target_shape)
        
        # 堆叠为多通道特征 (使用MFCC、Delta MFCC、能量和时域特征)
        features = np.stack([
            mfcc, 
            mfcc_delta, 
            mfcc_delta2,
            energy,
            temporal,
            zcr,
            spectral_centroid
        ], axis=0).astype(np.float32)
        
        # 训练集数据增强
        if self.is_train and random.random() < self.augment_prob:
            features = self._augment(features)
            
        # 获取标签
        item = self.metadata[idx]
        return torch.from_numpy(features), torch.tensor(item['label'], dtype=torch.long)

    def _extract_features(self, idx):
        """提取多种音频特征"""
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['file_path'].replace('.npy', '.wav'))
        
        try:
            # 使用MFCC特征及其差分
            mfcc, mfcc_delta, mfcc_delta2 = extract_mfcc(audio_path, n_mfcc=40)
            
            # 使用时域特征
            temporal, zcr = extract_temporal_features(audio_path)
            
            # 使用能量特征
            energy, spectral_centroid = extract_energy_features(audio_path, n_mels=40)
            
            # 确保特征长度一致
            max_length = 100
            if mfcc.shape[1] > max_length:
                mfcc = mfcc[:, :max_length]
                mfcc_delta = mfcc_delta[:, :max_length]
                mfcc_delta2 = mfcc_delta2[:, :max_length]
                temporal = temporal[:max_length]
                zcr = zcr[:max_length]
                energy = energy[:, :max_length]
                spectral_centroid = spectral_centroid[:max_length]
            elif mfcc.shape[1] < max_length:
                pad_width = max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                mfcc_delta = np.pad(mfcc_delta, ((0, 0), (0, pad_width)), mode='constant')
                mfcc_delta2 = np.pad(mfcc_delta2, ((0, 0), (0, pad_width)), mode='constant')
                temporal = np.pad(temporal, (0, pad_width), mode='constant')
                zcr = np.pad(zcr, (0, pad_width), mode='constant')
                energy = np.pad(energy, ((0, 0), (0, pad_width)), mode='constant')
                spectral_centroid = np.pad(spectral_centroid, (0, pad_width), mode='constant')
                
            return mfcc, mfcc_delta, mfcc_delta2, temporal, zcr, energy, spectral_centroid
        except Exception as e:
            print(f"提取特征失败: {audio_path}, 错误: {e}")
            # 返回随机特征作为后备
            return (np.random.randn(40, 100), np.random.randn(40, 100), np.random.randn(40, 100),
                    np.random.randn(100), np.random.randn(100), np.random.randn(40, 100), np.random.randn(100))

    def _resize_features(self, features, target_size):
        """调整特征图尺寸"""
        # 使用插值调整尺寸
        if len(features.shape) == 1:
            # 一维特征需要先转换为二维
            features = features.reshape(1, -1)
            zoom_factor = (target_size[0] / features.shape[0], target_size[1] / features.shape[1])
            features = ndimage.zoom(features, zoom_factor, order=1)
            return features
        else:
            # 二维特征
            zoom_factor = (target_size[0] / features.shape[0], target_size[1] / features.shape[1])
            return ndimage.zoom(features, zoom_factor, order=1)

    def _augment(self, features):
        """增强函数 - 适用于多通道输入"""
        augmented = features.copy()
        
        # 时间轴随机偏移
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        
        # 随机噪声注入
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape)
            augmented += noise
            
        # 频率掩码 (仅对MFCC相关通道)
        if random.random() < 0.3:
            freq_mask_width = int(augmented.shape[1] * 0.1)
            freq_mask_start = random.randint(0, augmented.shape[1] - freq_mask_width)
            # 对MFCC、Delta MFCC通道应用掩码
            for i in range(3):  # 前3个通道是MFCC相关
                augmented[i, freq_mask_start:freq_mask_start+freq_mask_width, :] = 0
            
        # 时间掩码
        if random.random() < 0.3:
            time_mask_width = int(augmented.shape[2] * 0.1)
            time_mask_start = random.randint(0, augmented.shape[2] - time_mask_width)
            augmented[:, :, time_mask_start:time_mask_start+time_mask_width] = 0
        
        return np.clip(augmented, -3, 3)

# ------------------------------------------------------------------------------
# 5. 辅助函数
# ------------------------------------------------------------------------------
def calculate_class_weights(labels):
    """计算类别权重"""
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    
    return torch.tensor(weights / weights.sum())

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练单个epoch"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs, attention_weights = model(inputs)
        
        # 使用自定义损失函数（带Label Smoothing）
        if hasattr(model, 'loss_fn'):
            loss = model.loss_fn(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"批次损失": loss.item()})
    
    # 计算epoch指标
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device, label_names):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_attention_weights = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="验证"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs, attention_weights = model(inputs)
            
            # 计算损失
            if hasattr(model, 'loss_fn'):
                loss = model.loss_fn(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu())
    
    # 计算验证指标
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 打印分类报告
    print("\n验证集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))
    
    # 合并所有批次的注意力权重
    if all_attention_weights:
        attention_weights = torch.cat(all_attention_weights, dim=0)
    else:
        attention_weights = None
    
    return val_loss, val_acc, val_f1, all_labels, all_preds, attention_weights

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='训练损失')
    axes[0].plot(history['val_loss'], label='验证损失')
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制准确率曲线
    axes[1].plot(history['train_acc'], label='训练准确率')
    axes[1].plot(history['val_acc'], label='验证准确率')
    axes[1].set_title('训练和验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制F1分数曲线
    axes[2].plot(history['train_f1'], label='训练F1分数')
    axes[2].plot(history['val_f1'], label='验证F1分数')
    axes[2].set_title('训练和验证F1分数')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1分数')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, label_names, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_weights(attention_weights, feature_names, output_dir):
    """绘制注意力权重"""
    if attention_weights is None:
        print("无注意力权重可绘制")
        return
    
    # 计算平均注意力权重
    if len(attention_weights.shape) == 4:  # [batch, channels, 1, 1]
        avg_attention = attention_weights.mean(dim=0).squeeze().numpy()
    elif len(attention_weights.shape) == 2:  # [batch, channels]
        avg_attention = attention_weights.mean(dim=0).numpy()
    else:
        print(f"无法处理的注意力权重形状: {attention_weights.shape}")
        return
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(feature_names)), avg_attention)
    plt.title('特征注意力权重')
    plt.xlabel('特征类型')
    plt.ylabel('注意力权重')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    # 在条形上添加数值标签
    for bar, value in zip(bars, avg_attention):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 6. 主函数 - 使用Cyclic Learning Rate和Warmup
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='基于注意力机制的多种特征的鸟类声纹识别 - 终极优化版')
    parser.add_argument('--audio_dir', type=str, default='D:/paper/data/processed_audio',
                        help='重采样后的音频数据目录')
    parser.add_argument('--metadata', type=str, default='D:/paper/data/features/metadata.json',
                        help='元数据JSON路径')
    parser.add_argument('--label_mapping', type=str, default='D:/paper/data/features/label_mapping.json',
                        help='标签映射JSON路径')
    parser.add_argument('--output_dir', type=str, default='D:/paper/results/ultimate_model_c',
                        help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=70, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存参数配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # 加载标签映射
    with open(args.label_mapping, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    label_to_name = label_mapping['label_to_name']
    num_classes = len(label_to_name)
    label_names = [label_to_name[str(i)] for i in range(num_classes)]
    
    # 加载数据集
    print("加载数据集...")
    full_dataset = EnhancedBirdSoundDataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        is_train=True,
        augment_prob=0.3  # 降低数据增强概率
    )
    
    # 使用分层抽样划分数据集
    print("使用分层抽样划分数据集...")
    labels = full_dataset.labels
    
    # 划分训练集和临时集（验证+测试）
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.3,  # 30%用于验证和测试
        stratify=labels,
        random_state=SEED
    )
    
    # 从临时集中划分验证集和测试集
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # 各占15%
        stratify=temp_labels,
        random_state=SEED
    )
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    # 设置数据集的模式
    def set_dataset_mode(dataset, is_train):
        if hasattr(dataset, 'dataset'):
            set_dataset_mode(dataset.dataset, is_train)
        else:
            dataset.is_train = is_train
    
    set_dataset_mode(train_dataset, True)
    set_dataset_mode(val_dataset, False)
    set_dataset_mode(test_dataset, False)
    
    print(f"数据集划分: 训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 测试集 {len(test_dataset)}")
    print(f"类别数量: {num_classes}")
    
    # 计算类别权重
    class_weights = calculate_class_weights(full_dataset.labels)
    print(f"类别权重: {class_weights.numpy()}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型 - 注意输入通道数为7（新增了Delta MFCC等特征）
    model = ImprovedBirdNetWithAttention(
        num_classes=num_classes, 
        in_channels=7,  # 新增了Delta MFCC等特征
        label_smoothing=args.label_smoothing
    ).to(device)
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 使用AdamW优化器，适度权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用带热身的余弦退火学习率调度
    def lr_lambda(epoch):
        # 前5个epoch进行学习率热身
        if epoch < 5:
            return (epoch + 1) / 5
        # 之后使用余弦退火
        return 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (args.epochs - 5)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # 初始化训练状态变量
    best_val_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # 开始训练
    print(f"\n开始训练终极优化版Model C (带注意力机制的多特征融合)...")
    try:
        for epoch in range(1, args.epochs + 1):
            # 训练
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # 验证
            val_loss, val_acc, val_f1, _, _, attention_weights = validate(
                model, val_loader, criterion, device, label_names
            )
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # 打印epoch摘要
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.4f}, F1={train_f1:.4f}")
            print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}, F1={val_f1:.4f}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"保存最佳模型 (验证F1: {best_val_f1:.4f})")
                early_stop_counter = 0
                
                # 保存注意力权重可视化
                feature_names = ['MFCC', 'MFCC Delta', 'MFCC Delta2', 'Energy', 'Temporal', 'ZCR', 'Spectral Centroid']
                plot_attention_weights(attention_weights, feature_names, args.output_dir)
            else:
                early_stop_counter += 1
                print(f"早停计数: {early_stop_counter}/{args.patience}")
                if early_stop_counter >= args.patience:
                    print(f"早停触发 (连续 {args.patience} 个epoch未提升)")
                    break
            
            # 每5轮更新一次训练曲线
            if epoch % 5 == 0:
                plot_training_history(history, args.output_dir)
    
    except KeyboardInterrupt:
        print("\n检测到中断信号，保存当前训练状态...")
        # 保存当前模型状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_f1': best_val_f1,
            'history': history
        }, os.path.join(args.output_dir, 'checkpoint.pth'))
        print("检查点已保存")
    
    # 最终绘制训练历史
    plot_training_history(history, args.output_dir)
    
    # 测试集评估
    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_acc, test_f1, test_labels, test_preds, attention_weights = validate(
        model, test_loader, criterion, device, label_names
    )
    
    print(f"\n测试集结果:")
    print(f"损失: {test_loss:.4f}, 准确率: {test_acc:.4f}, F1分数: {test_f1:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, label_names, args.output_dir)
    
    # 绘制注意力权重
    feature_names = ['MFCC', 'MFCC Delta', 'MFCC Delta2', 'Energy', 'Temporal', 'ZCR', 'Spectral Centroid']
    plot_attention_weights(attention_weights, feature_names, args.output_dir)
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # 保存模型架构信息
    if attention_weights is not None and len(attention_weights) > 0:
        if len(attention_weights.shape) == 4:  # [batch, channels, 1, 1]
            attention_dict = {
                'MFCC': float(attention_weights[0, 0, 0, 0]),
                'MFCC Delta': float(attention_weights[0, 1, 0, 0]),
                'MFCC Delta2': float(attention_weights[0, 2, 0, 0]),
                'Energy': float(attention_weights[0, 3, 0, 0]),
                'Temporal': float(attention_weights[0, 4, 0, 0]),
                'ZCR': float(attention_weights[0, 5, 0, 0]),
                'Spectral Centroid': float(attention_weights[0, 6, 0, 0])
            }
        elif len(attention_weights.shape) == 2:  # [batch, channels]
            attention_dict = {
                'MFCC': float(attention_weights[0, 0]),
                'MFCC Delta': float(attention_weights[0, 1]),
                'MFCC Delta2': float(attention_weights[0, 2]),
                'Energy': float(attention_weights[0, 3]),
                'Temporal': float(attention_weights[0, 4]),
                'ZCR': float(attention_weights[0, 5]),
                'Spectral Centroid': float(attention_weights[0, 6])
            }
        else:
            attention_dict = {
                'MFCC': 1.0,
                'MFCC Delta': 1.0,
                'MFCC Delta2': 1.0,
                'Energy': 1.0,
                'Temporal': 1.0,
                'ZCR': 1.0,
                'Spectral Centroid': 1.0
            }
    else:
        attention_dict = {
            'MFCC': 1.0,
            'MFCC Delta': 1.0,
            'MFCC Delta2': 1.0,
            'Energy': 1.0,
            'Temporal': 1.0,
            'ZCR': 1.0,
            'Spectral Centroid': 1.0
        }

    model_info = {
        'model_name': 'UltimateBirdNetWithAttention',
        'num_classes': num_classes,
        'input_channels': 7,
        'features': ['MFCC', 'MFCC Delta', 'MFCC Delta2', 'Energy (Mel-spectrogram)', 'Temporal (RMSE)', 'ZCR', 'Spectral Centroid'],
        'attention_weights': attention_dict,
        'class_names': label_names,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_val_f1': best_val_f1,
        'label_smoothing': args.label_smoothing
    }
    with open(os.path.join(args.output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()