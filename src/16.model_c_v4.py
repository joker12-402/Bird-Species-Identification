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
import librosa
import scipy.signal as signal
from scipy import ndimage

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
# 1. 特征注意力机制（增强版）
# ------------------------------------------------------------------------------
class FeatureAttention(nn.Module):
    """改进的特征注意力机制，动态调整不同特征通道权重"""
    def __init__(self, num_features=3, reduction_ratio=2):
        super(FeatureAttention, self).__init__()
        self.num_features = num_features
        self.reduction_ratio = reduction_ratio
        
        # 校验缩减比合理性
        assert num_features % reduction_ratio == 0, \
            f"特征通道数 {num_features} 必须能被缩减比 {reduction_ratio} 整除"
        
        # 全局平均池化 + 全局最大池化融合
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 改进的注意力网络
        self.attention = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 2, num_features // reduction_ratio),  # 融合两种池化结果
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // reduction_ratio),  # 增加批归一化
            nn.Linear(num_features // reduction_ratio, num_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, num_features, height, width]
        batch_size, num_features, height, width = x.size()
        
        # 融合全局平均池化和最大池化
        avg_pool = self.global_avg_pool(x).squeeze(-1).squeeze(-1)  # [batch, num_features]
        max_pool = self.global_max_pool(x).squeeze(-1).squeeze(-1)  # [batch, num_features]
        combined = torch.cat([avg_pool, max_pool], dim=1)  # [batch, 2*num_features]
        
        # 计算注意力权重
        attention_weights = self.attention(combined)  # [batch, num_features]
        attention_weights = attention_weights.view(batch_size, num_features, 1, 1)
        
        # 应用注意力权重
        weighted_features = x * attention_weights
        return weighted_features, attention_weights

# ------------------------------------------------------------------------------
# 2. 改进的Model C架构
# ------------------------------------------------------------------------------
class ImprovedBirdNetWithAttention(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(ImprovedBirdNetWithAttention, self).__init__()
        
        # 特征注意力机制（优先关注能量特征）
        self.feature_attention = FeatureAttention(num_features=in_channels, reduction_ratio=1)
        
        # 特征提取 backbone（加深网络并增强正则化）
        self.features = nn.Sequential(
            # 输入: (3, 128, 128)
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),  # 空间dropout
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 增加卷积层
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 增加卷积层
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # 空间注意力模块
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 分类器（增强正则化）
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # 提高dropout比例
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),  # 增加批归一化
            nn.Dropout(0.6),
            nn.Linear(1024, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 应用特征注意力
        x, attention_weights = self.feature_attention(x)
        
        # 特征提取与分类
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x, attention_weights

# ------------------------------------------------------------------------------
# 3. 特征提取函数（支持三种特征）
# ------------------------------------------------------------------------------
def extract_mfcc(audio_path, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512):
    """提取MFCC特征"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # 预加重
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmax=sr//2
        )
        return mfcc
    except Exception as e:
        print(f"MFCC提取失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mfcc, 100))

def extract_temporal_features(audio_path, sr=16000, hop_length=512):
    """提取时域特征（RMSE+过零率）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)  # 均方根能量
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)  # 过零率
        return np.vstack([rmse, zcr])  # 融合为2维特征
    except Exception as e:
        print(f"时域特征提取失败: {audio_path}, 错误: {e}")
        return np.zeros((2, 100))

def extract_energy_features(audio_path, sr=16000, n_mels=40, hop_length=512):
    """提取能量特征（梅尔频谱）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        return librosa.power_to_db(mel_spectrogram, ref=np.max)  # 转换为分贝
    except Exception as e:
        print(f"能量特征提取失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mels, 100))

# ------------------------------------------------------------------------------
# 4. 增强版数据集（三特征融合）
# ------------------------------------------------------------------------------
class EnhancedBirdSoundDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_size=(128, 128), 
                 is_train=False, augment_prob=0.6):  # 提高增强概率
        # 读取元数据时指定UTF-8编码
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.labels = [item['label'] for item in self.metadata]
        self.num_classes = len(set(self.labels))
        self._compute_feature_stats()  # 计算三特征联合统计量
        
    def _compute_feature_stats(self):
        """计算三种特征的联合统计量"""
        print("计算特征统计量...")
        all_features = []
        sample_size = min(100, len(self.metadata))
        sample_indices = np.random.choice(len(self.metadata), sample_size, replace=False)
        
        for idx in tqdm(sample_indices, desc="计算统计量"):
            mfcc, temporal, energy = self._extract_raw_features(idx)
            if mfcc is not None and temporal is not None and energy is not None:
                all_features.append(mfcc.flatten())
                all_features.append(temporal.flatten())
                all_features.append(energy.flatten())
        
        if not all_features:
            self.feature_mean = 0.0
            self.feature_std = 1.0
            print("使用默认统计量")
            return
        
        all_features = np.concatenate(all_features)
        self.feature_mean = np.mean(all_features)
        self.feature_std = np.std(all_features)
        print(f"特征统计量: 均值={self.feature_mean:.4f}, 标准差={self.feature_std:.4f}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 提取三种特征
        mfcc, temporal, energy = self._extract_raw_features(idx)
        
        # 调整尺寸并标准化
        mfcc = self._resize_features(mfcc, self.target_size)
        temporal = self._resize_features(temporal, self.target_size)
        energy = self._resize_features(energy, self.target_size)
        
        # 合并为3通道特征
        features = np.stack([mfcc, temporal, energy], axis=0).astype(np.float32)
        features = (features - self.feature_mean) / self.feature_std
        
        # 训练集增强
        if self.is_train and random.random() < self.augment_prob:
            features = self._augment(features)
            
        item = self.metadata[idx]
        return torch.from_numpy(features), torch.tensor(item['label'], dtype=torch.long)

    def _extract_raw_features(self, idx):
        """提取原始三特征"""
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['file_path'].replace('.npy', '.wav'))
        
        mfcc = extract_mfcc(audio_path)
        temporal = extract_temporal_features(audio_path)
        energy = extract_energy_features(audio_path)
        
        # 统一特征长度
        for feat in [mfcc, temporal, energy]:
            if feat.shape[1] > 100:
                feat = feat[:, :100]
            elif feat.shape[1] < 100:
                pad_width = 100 - feat.shape[1]
                feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
        
        return mfcc, temporal, energy

    def _resize_features(self, features, target_size):
        """调整特征尺寸"""
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        zoom_factor = (target_size[0]/features.shape[0], target_size[1]/features.shape[1])
        return ndimage.zoom(features, zoom_factor, order=1)

    def _augment(self, features):
        """增强函数（针对三通道优化）"""
        augmented = features.copy()
        
        # 时间轴偏移
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        
        # 噪声注入
        if random.random() < 0.5:  # 提高噪声概率
            noise = np.random.normal(0, 0.03, augmented.shape)
            augmented += noise
            
        # 频率掩码（仅对MFCC和能量通道）
        if random.random() < 0.4:
            freq_mask_width = int(augmented.shape[1] * 0.1)
            freq_mask_start = random.randint(0, augmented.shape[1] - freq_mask_width)
            augmented[0, freq_mask_start:freq_mask_start+freq_mask_width, :] = 0  # MFCC
            augmented[2, freq_mask_start:freq_mask_start+freq_mask_width, :] = 0  # 能量
            
        # 时间掩码（全通道）
        if random.random() < 0.4:
            time_mask_width = int(augmented.shape[2] * 0.1)
            time_mask_start = random.randint(0, augmented.shape[2] - time_mask_width)
            augmented[:, :, time_mask_start:time_mask_start+time_mask_width] = 0
        
        return np.clip(augmented, -3, 3)

# ------------------------------------------------------------------------------
# 5. 训练与验证函数
# ------------------------------------------------------------------------------
def calculate_class_weights(labels):
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    weights = np.zeros(num_classes, dtype=np.float32)
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    return torch.tensor(weights / weights.sum())

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"批次损失": loss.item()})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device, label_names=None):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_attention_weights = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attention_weights = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 分析注意力权重（确保能量特征被充分利用）
    avg_attention = np.mean(all_attention_weights, axis=0)
    print(f"平均注意力权重 - MFCC: {avg_attention[0,0,0]:.4f}, "
          f"时域: {avg_attention[1,0,0]:.4f}, 能量: {avg_attention[2,0,0]:.4f}")
    
    if label_names:
        print("\n验证集分类报告:")
        print(classification_report(
            all_labels, all_preds, labels=range(len(label_names)),
            target_names=label_names, digits=4, zero_division=0
        ))
    
    return val_loss, val_acc, val_f1, all_labels, all_preds, avg_attention

# ------------------------------------------------------------------------------
# 6. 辅助可视化函数
# ------------------------------------------------------------------------------
def plot_training_history(history, save_path):
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(133)
    plt.plot(history['train_f1'], label='训练F1')
    plt.plot(history['val_f1'], label='验证F1')
    plt.title('F1分数曲线')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
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
# 7. 主函数（优化训练策略）
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='优化版Model C (MFCC+时域+能量+注意力)')
    parser.add_argument('--audio_dir', type=str, default='D:/paper/data/processed_audio')
    parser.add_argument('--metadata', type=str, default='D:/paper/data/features/metadata.json')
    parser.add_argument('--label_mapping', type=str, default='D:/paper/data/features/label_mapping.json')
    parser.add_argument('--output_dir', type=str, default='D:/paper/results/model_c_v4')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)  # 增大批次
    parser.add_argument('--lr', type=float, default=0.0008)  # 降低初始学习率
    parser.add_argument('--patience', type=int, default=8)  # 减少早停耐心值
    parser.add_argument('--reduction_ratio', type=int, default=1)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置时指定UTF-8编码，并确保中文正常显示
    with open(os.path.join(args.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    
    # 加载标签时指定UTF-8编码
    with open(args.label_mapping, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    label_to_name = label_mapping['label_to_name']
    num_classes = len(label_to_name)
    label_names = [label_to_name[str(i)] for i in range(num_classes)]
    
    # 加载数据集
    full_dataset = EnhancedBirdSoundDataset(
        metadata_path=args.metadata,
        audio_dir=args.audio_dir,
        is_train=True,
        augment_prob=0.6  # 增强概率提高到60%
    )
    
    # 划分数据集
    labels = full_dataset.labels
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)), test_size=0.3, stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED
    )
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    # 设置数据集模式
    def set_dataset_mode(dataset, is_train):
        if hasattr(dataset, 'dataset'):
            set_dataset_mode(dataset.dataset, is_train)
        else:
            dataset.is_train = is_train
    
    set_dataset_mode(train_dataset, True)
    set_dataset_mode(val_dataset, False)
    set_dataset_mode(test_dataset, False)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型与优化器
    model = ImprovedBirdNetWithAttention(
        num_classes=num_classes, 
        in_channels=3
    ).to(device)
    
    class_weights = calculate_class_weights(full_dataset.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器与学习率调度（使用带重启的余弦退火）
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=5e-4  # 增加权重衰减
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # 训练过程
    best_val_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    print("\n开始训练优化版Model C...")
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            val_loss, val_acc, val_f1, _, _, _ = validate(
                model, val_loader, criterion, device, label_names
            )
            
            scheduler.step()  # 余弦退火+重启
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
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
                if early_stop_counter >= args.patience:
                    print(f"早停触发 (连续 {args.patience} 个epoch未提升)")
                    break
            
            if epoch % 5 == 0:
                plot_training_history(history, args.output_dir)
        
    except KeyboardInterrupt:
        print("\n保存中断状态...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, os.path.join(args.output_dir, 'checkpoint.pth'))
    
    # 最终评估
    plot_training_history(history, args.output_dir)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_loss, test_acc, test_f1, test_labels, test_preds, _ = validate(
        model, test_loader, criterion, device, label_names
    )
    
    print(f"\n测试集结果:")
    print(f"损失: {test_loss:.4f}, 准确率: {test_acc:.4f}, F1分数: {test_f1:.4f}")
    
    plot_confusion_matrix(test_labels, test_preds, label_names, args.output_dir)
    
    # 保存结果时指定UTF-8编码
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"所有结果已保存至: {args.output_dir}")

if __name__ == '__main__':
    main()
