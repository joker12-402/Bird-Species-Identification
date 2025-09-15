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
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter, deque
import random
import time
import torchvision.models as models
import librosa
import scipy.signal as signal
from scipy.fftpack import dct
from scipy import ndimage
from sklearn.utils import compute_class_weight

# 设置随机种子，确保实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# 配置中文字体（解决绘图中文显示问题）
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

# ------------------------------------------------------------------------------
# 焦点损失 (Focal Loss)
# ------------------------------------------------------------------------------
# 修改 Focal Loss 的实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-BCE_loss)
        
        # 如果提供了alpha，为每个样本选择对应的类别权重
        if self.alpha is not None:
            # 确保alpha在正确的设备上
            alpha = self.alpha.to(inputs.device)
            # 为每个样本选择对应的类别权重
            alpha_t = alpha[targets]
            # 计算Focal Loss
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            # 如果没有提供alpha，使用标准Focal Loss
            F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# ------------------------------------------------------------------------------
# 基础特征提取函数
# ------------------------------------------------------------------------------
def compute_mfcc(y, sr=16000, n_fft=512, hop_length=160, n_mels=40, n_mfcc=13):
    """基础MFCC实现，不依赖librosa的高级功能"""
    # 预加重
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # 分帧
    frame_length = n_fft
    frames = []
    for i in range(0, len(y) - frame_length, hop_length):
        frames.append(y[i:i+frame_length])
    
    if not frames:
        # 如果音频太短，返回空特征
        return np.zeros((1, n_mfcc))
    
    frames = np.array(frames)
    
    # 加窗
    window = np.hanning(frame_length)
    frames = frames * window
    
    # 计算功率谱
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    
    # 创建Mel滤波器组
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((n_fft + 1) * hz_points / sr)
    
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    # 应用滤波器组
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT变换得到MFCC
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(n_mfcc+1)]
    
    return mfcc

def compute_pncc_basic(y, sr=16000, n_fft=512, hop_length=160, n_filters=40, n_ceps=13):
    """简化版PNCC实现"""
    # 使用基础的MFCC实现
    mfcc = compute_mfcc(y, sr, n_fft, hop_length, n_filters, n_ceps)
    
    # 添加一些PNCC特有的处理
    # 这里简化处理，实际PNCC有更复杂的处理流程
    pncc = mfcc.copy()
    
    # 添加一些噪声鲁棒性处理
    pncc = pncc - np.mean(pncc, axis=0)
    pncc = pncc / (np.std(pncc, axis=0) + 1e-8)
    
    return pncc

# ------------------------------------------------------------------------------
# 特征融合函数
# ------------------------------------------------------------------------------
def extract_features(audio_path, sr=16000, feature_type='mpfc'):
    """
    提取音频特征
    参数:
        audio_path: 音频文件路径
        sr: 采样率
        feature_type: 特征类型 ('mfcc', 'pncc', 'mpfc')
    返回:
        features: 特征矩阵
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"加载音频文件失败: {audio_path}, 错误: {e}")
        # 返回随机特征作为后备
        if feature_type == 'mfcc':
            return np.random.randn(100, 13)
        elif feature_type == 'pncc':
            return np.random.randn(100, 13)
        else:  # mpfc
            return np.random.randn(100, 26)
    
    if feature_type == 'mfcc':
        # 提取MFCC特征
        return compute_mfcc(y, sr=sr)
    
    elif feature_type == 'pncc':
        # 提取PNCC特征
        return compute_pncc_basic(y, sr=sr)
    
    elif feature_type == 'mpfc':
        # 融合MFCC和PNCC特征
        mfcc = compute_mfcc(y, sr=sr)
        pncc = compute_pncc_basic(y, sr=sr)
        
        # 确保特征长度一致
        min_len = min(mfcc.shape[0], pncc.shape[0])
        mfcc = mfcc[:min_len, :]
        pncc = pncc[:min_len, :]
        
        # 特征融合 (拼接)
        fused_features = np.concatenate([mfcc, pncc], axis=1)
        
        return fused_features
    
    else:
        raise ValueError(f"不支持的 feature_type: {feature_type}")

# ------------------------------------------------------------------------------
# 坐标注意力机制 (Coordinate Attention)
# ------------------------------------------------------------------------------
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mid_channels = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        # 水平方向注意力
        x_h = self.pool_h(x)
        x_h = self.conv1(x_h)
        x_h = F.relu(self.bn1(x_h))
        x_h = self.conv_h(x_h)
        x_h = x_h.sigmoid()
        
        # 垂直方向注意力
        x_w = self.pool_w(x)
        x_w = self.conv1(x_w)
        x_w = F.relu(self.bn1(x_w))
        x_w = self.conv_w(x_w)
        x_w = x_w.sigmoid()
        
        # 应用注意力
        out = identity * x_h * x_w
        return out

# ------------------------------------------------------------------------------
# 带坐标注意力和Dropout的ResNet模型
# ------------------------------------------------------------------------------
class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes, model_name='resnet18', pretrained=False, in_channels=1, dropout_rate=0.5):
        super(ResNetWithAttention, self).__init__()
        
        # 加载预定义的ResNet模型
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported model name")
        
        # 修改第一层卷积以适应不同通道输入
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 在特定层后添加坐标注意力
        self.ca1 = CoordinateAttention(64)
        self.ca2 = CoordinateAttention(128)
        self.ca3 = CoordinateAttention(256)
        self.ca4 = CoordinateAttention(512)
        
        # 添加Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 修改最后一层全连接
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.ca1(x)
        x = self.dropout(x)
        
        x = self.resnet.layer2(x)
        x = self.ca2(x)
        x = self.dropout(x)
        
        x = self.resnet.layer3(x)
        x = self.ca3(x)
        x = self.dropout(x)
        
        x = self.resnet.layer4(x)
        x = self.ca4(x)
        x = self.dropout(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        
        return x

# ------------------------------------------------------------------------------
# 数据集类
# ------------------------------------------------------------------------------
class BirdSoundDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, feature_type='mpfc', 
                 transform=None, is_train=False, augment_prob=0.8, 
                 target_size=(224, 224)):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.audio_dir = audio_dir
        self.feature_type = feature_type
        self.transform = transform
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.target_size = target_size
        
        # 提取所有标签用于计算类别权重
        self.labels = [item['label'] for item in self.metadata]
        self.num_classes = len(set(self.labels))
        
        # 预计算特征统计量（如果不存在）
        self.stats_file = f'feature_stats_{feature_type}.json'
        if not os.path.exists(self.stats_file):
            self._compute_feature_stats()
        else:
            # 加载特征统计量
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            self.feature_mean = stats['mean']
            self.feature_std = stats['std']
            print(f"已加载特征统计量: 均值={self.feature_mean:.4f}, 标准差={self.feature_std:.4f}")

    def _compute_feature_stats(self):
        """计算特征统计量"""
        print("计算特征统计量...")
        all_features = []
        
        # 随机采样部分数据计算统计量
        sample_size = min(200, len(self.metadata))  # 减少样本数量
        sample_indices = np.random.choice(len(self.metadata), sample_size, replace=False)
        
        successful_samples = 0
        for idx in tqdm(sample_indices, desc="计算统计量"):
            item = self.metadata[idx]
            audio_path = os.path.join(self.audio_dir, item['file_path'].replace('.npy', '.wav'))
            
            try:
                features = extract_features(audio_path, feature_type=self.feature_type)
                all_features.append(features)
                successful_samples += 1
            except Exception as e:
                print(f"处理文件失败: {audio_path}, 错误: {e}")
                continue
        
        # 检查是否有成功提取的特征
        if successful_samples == 0:
            # 使用默认值
            self.feature_mean = 0.0
            self.feature_std = 1.0
            print("无法提取任何特征，使用默认统计量: 均值=0.0, 标准差=1.0")
            
            # 保存默认统计量
            stats = {
                'mean': 0.0,
                'std': 1.0
            }
            
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f)
            return
        
        # 计算全局均值和标准差
        all_features = np.vstack(all_features)
        feature_mean = np.mean(all_features)
        feature_std = np.std(all_features)
        
        # 保存统计量
        stats = {
            'mean': float(feature_mean),
            'std': float(feature_std)
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f)
        
        print(f"特征统计量已保存: 均值={feature_mean:.4f}, 标准差={feature_std:.4f}")
        print(f"成功处理 {successful_samples}/{sample_size} 个样本")
        
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['file_path'].replace('.npy', '.wav'))
        
        # 提取特征
        try:
            features = extract_features(audio_path, feature_type=self.feature_type)
        except Exception as e:
            print(f"提取特征失败: {audio_path}, 错误: {e}")
            # 返回随机特征作为后备
            features = np.random.randn(100, 13 if self.feature_type == 'mfcc' else 13 if self.feature_type == 'pncc' else 26)
        
        # 特征标准化
        features = (features - self.feature_mean) / self.feature_std
        
        # 调整为固定尺寸
        features = self._resize_features(features, self.target_size)
        
        # 根据特征类型确定通道数
        if self.feature_type == 'mpfc':
            # 对于MPFC特征，将其分成两个通道
            # 假设MPFC特征维度为26，分成两个13维的特征
            half_dim = features.shape[1] // 2
            channel1 = features[:, :half_dim]
            channel2 = features[:, half_dim:]
            
            # 分别调整为目标尺寸
            channel1 = self._resize_features(channel1, self.target_size)
            channel2 = self._resize_features(channel2, self.target_size)
            
            # 合并为两个通道
            features = np.stack([channel1, channel2], axis=0)  # 形状: (2, H, W)
        else:
            # 对于单特征，增加一个通道维度
            features = features[np.newaxis, ...]  # 形状: (1, H, W)
        
        features = features.astype(np.float32)
        
        # 训练集数据增强
        if self.is_train and random.random() < self.augment_prob:
            features = self._augment(features)
        
        # 应用变换
        if self.transform:
            features = self.transform(features)
            
        return torch.from_numpy(features), torch.tensor(item['label'], dtype=torch.long)

    def _resize_features(self, features, target_size):
        """调整特征图尺寸"""
        # 当前尺寸
        orig_height, orig_width = features.shape
        
        # 调整高度
        if orig_height != target_size[0]:
            zoom_factor = target_size[0] / orig_height
            features = ndimage.zoom(features, (zoom_factor, 1), order=1)
        
        # 调整宽度
        if features.shape[1] != target_size[1]:
            # 截断或填充
            if features.shape[1] > target_size[1]:
                features = features[:, :target_size[1]]
            else:
                pad_width = target_size[1] - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        
        return features

    def _augment(self, features):
        """增强函数"""
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
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.05, augmented.shape)
            augmented += noise
        
        # 频率轴局部缩放
        if random.random() < 0.4:
            start = random.randint(0, augmented.shape[1]//2)
            end = start + random.randint(augmented.shape[1]//6, augmented.shape[1]//3)
            end = min(end, augmented.shape[1])
            scale = 0.8 + random.random() * 0.4
            augmented[:, start:end, :] *= scale
        
        # 音量缩放
        if random.random() < 0.4:
            volume_scale = 0.8 + random.random() * 0.4
            augmented *= volume_scale
        
        # 添加背景噪声
        if random.random() < 0.3:
            noise_level = random.random() * 0.1
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented += noise
        
        # 时间拉伸
        if random.random() < 0.3:
            rate = 0.8 + random.random() * 0.4
            orig_len = augmented.shape[2]
            new_len = int(orig_len * rate)
            augmented = ndimage.zoom(augmented, (1, 1, rate), order=1)
            if augmented.shape[2] > orig_len:
                augmented = augmented[:, :, :orig_len]
            else:
                pad_width = orig_len - augmented.shape[2]
                augmented = np.pad(augmented, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        
        return np.clip(augmented, -3, 3)

# ------------------------------------------------------------------------------
# 辅助函数
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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=None):
    """训练单个epoch"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
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
    
    # 更新学习率
    if scheduler:
        scheduler.step()
    
    # 计算epoch指标
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device, label_names=None):
    """验证模型性能"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 打印分类报告
    if label_names:
        print("\n验证集分类报告:")
        print(classification_report(
            all_labels, all_preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            digits=4,
            zero_division=0
        ))
    
    return val_loss, val_acc, val_f1, all_labels, all_preds

def plot_training_history(history, save_path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='训练损失', color='#1f77b4')
    plt.plot(history['val_loss'], label='验证损失', color='#ff7f0e')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='训练准确率', color='#1f77b4')
    plt.plot(history['val_acc'], label='验证准确率', color='#ff7f0e')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='训练F1', color='#1f77b4')
    plt.plot(history['val_f1'], label='验证F1', color='#ff7f0e')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('测试集混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45, ha='right')
    plt.yticks(tick_marks, label_names)
    
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

# ------------------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='高级鸟类物种识别模型训练')
    parser.add_argument('--audio_dir', type=str, default='D:/paper/data/processed_audio',
                        help='重采样后的音频数据目录')
    parser.add_argument('--metadata', type=str, default='D:/paper/data/features/metadata.json',
                        help='元数据JSON路径')
    parser.add_argument('--label_mapping', type=str, default='D:/paper/data/features/label_mapping.json',
                        help='标签映射JSON路径')
    parser.add_argument('--output_dir', type=str, default='D:/paper/results/mpfc_attention_model_v2',
                        help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
                        help='选择模型架构')
    parser.add_argument('--feature_type', type=str, default='mpfc', 
                        choices=['mfcc', 'pncc', 'mpfc'], help='特征类型')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout比率')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')
    
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
    full_dataset = BirdSoundDataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        feature_type=args.feature_type,
        is_train=True,
        augment_prob=0.5
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
    print(f"使用特征类型: {args.feature_type}")
    
    # 计算类别权重
    class_weights = calculate_class_weights(full_dataset.labels)
    print(f"类别权重: {class_weights.numpy()}")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 根据特征类型确定输入通道数
    if args.feature_type == 'mpfc':
        in_channels = 2  # MFCC + PNCC
    else:
        in_channels = 1
    
    # 初始化模型
    model = ResNetWithAttention(
        num_classes=num_classes, 
        model_name=args.model, 
        pretrained=False,
        in_channels=in_channels,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 使用焦点损失
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=args.focal_gamma)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用ReduceLROnPlateau学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # 初始化训练状态变量
    best_val_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # 开始训练
    print(f"\n开始训练{args.model}模型...")
    try:
        for epoch in range(1, args.epochs + 1):
            # 训练
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = validate(
                model, val_loader, criterion, device, label_names
            )
            
            # 更新学习率
            scheduler.step(val_f1)
            
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
    
    # 最终绘制训练历史
    plot_training_history(history, args.output_dir)
    
    # 测试集评估
    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_acc, test_f1, test_labels, test_preds = validate(
        model, test_loader, criterion, device, label_names
    )
    
    print(f"\n测试集结果:")
    print(f"损失: {test_loss:.4f}, 准确率: {test_acc:.4f}, F1分数: {test_f1:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_labels, test_preds, label_names, args.output_dir)
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    # 保存模型架构信息
    model_info = {
        'model_name': args.model,
        'feature_type': args.feature_type,
        'num_classes': num_classes,
        'class_names': label_names,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'best_val_f1': best_val_f1
    }
    with open(os.path.join(args.output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()