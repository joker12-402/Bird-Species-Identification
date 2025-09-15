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
# 1. 改进的模型架构 - 适配三通道输入
# ------------------------------------------------------------------------------
class ImprovedBirdNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):  # 修改为3通道输入
        super(ImprovedBirdNet, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
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
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 2. 特征提取函数 - 增强版，提取MFCC、时域和能量特征
# ------------------------------------------------------------------------------
def extract_mfcc(audio_path, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512):
    """提取MFCC特征"""
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
        
        return mfcc
    except Exception as e:
        print(f"提取MFCC特征失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mfcc, 100))

def extract_temporal_features(audio_path, sr=16000, hop_length=512):
    """提取时域特征（短时能量）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # 计算短时能量 (Root Mean Square Energy)
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        return rmse
    except Exception as e:
        print(f"提取时域特征失败: {audio_path}, 错误: {e}")
        return np.zeros(100)

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
        
        return mel_spec_db
    except Exception as e:
        print(f"提取能量特征失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mels, 100))

# ------------------------------------------------------------------------------
# 3. 数据集类 - 增强版，支持三通道输入
# ------------------------------------------------------------------------------
class EnhancedBirdSoundDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, target_size=(128, 128), 
                 is_train=False, augment_prob=0.4):
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
        all_temporal_features = []
        all_energy_features = []
        
        # 随机采样部分数据计算统计量
        sample_size = min(100, len(self.metadata))
        sample_indices = np.random.choice(len(self.metadata), sample_size, replace=False)
        
        for idx in tqdm(sample_indices, desc="计算统计量"):
            mfcc, temporal, energy = self._extract_features(idx)
            if mfcc is not None:
                all_mfcc_features.append(mfcc)
                all_temporal_features.append(temporal)
                all_energy_features.append(energy)
        
        if not all_mfcc_features:
            # 使用默认值
            self.mfcc_mean = 0.0
            self.mfcc_std = 1.0
            self.temporal_mean = 0.0
            self.temporal_std = 1.0
            self.energy_mean = 0.0
            self.energy_std = 1.0
            print("无法提取任何特征，使用默认统计量: 均值=0.0, 标准差=1.0")
            return
        
        # 计算全局均值和标准差
        all_mfcc_features = np.vstack(all_mfcc_features)
        self.mfcc_mean = np.mean(all_mfcc_features)
        self.mfcc_std = np.std(all_mfcc_features)
        
        all_temporal_features = np.vstack(all_temporal_features)
        self.temporal_mean = np.mean(all_temporal_features)
        self.temporal_std = np.std(all_temporal_features)
        
        all_energy_features = np.vstack(all_energy_features)
        self.energy_mean = np.mean(all_energy_features)
        self.energy_std = np.std(all_energy_features)
        
        print(f"MFCC统计量: 均值={self.mfcc_mean:.4f}, 标准差={self.mfcc_std:.4f}")
        print(f"时域特征统计量: 均值={self.temporal_mean:.4f}, 标准差={self.temporal_std:.4f}")
        print(f"能量特征统计量: 均值={self.energy_mean:.4f}, 标准差={self.energy_std:.4f}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 提取特征
        mfcc_features, temporal_features, energy_features = self._extract_features(idx)
        
        # 特征标准化
        mfcc_features = (mfcc_features - self.mfcc_mean) / self.mfcc_std
        temporal_features = (temporal_features - self.temporal_mean) / self.temporal_std
        energy_features = (energy_features - self.energy_mean) / self.energy_std
        
        # 调整为固定尺寸
        mfcc_features = self._resize_features(mfcc_features, self.target_size)
        temporal_features = self._resize_features(temporal_features, self.target_size)
        energy_features = self._resize_features(energy_features, self.target_size)
        
        # 确保所有特征形状相同
        if mfcc_features.shape != temporal_features.shape:
            temporal_features = np.resize(temporal_features, mfcc_features.shape)
        if mfcc_features.shape != energy_features.shape:
            energy_features = np.resize(energy_features, mfcc_features.shape)
        
        # 堆叠为三通道特征
        features = np.stack([mfcc_features, temporal_features, energy_features], axis=0).astype(np.float32)
        
        # 训练集数据增强
        if self.is_train and random.random() < self.augment_prob:
            features = self._augment(features)
            
        # 获取标签
        item = self.metadata[idx]
        return torch.from_numpy(features), torch.tensor(item['label'], dtype=torch.long)

    def _extract_features(self, idx):
        """提取MFCC、时域和能量特征"""
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item['file_path'].replace('.npy', '.wav'))
        
        try:
            # 使用MFCC特征
            mfcc = extract_mfcc(audio_path, n_mfcc=40)
            
            # 使用时域特征
            temporal = extract_temporal_features(audio_path)
            
            # 使用能量特征
            energy = extract_energy_features(audio_path, n_mels=40)
            
            # 确保特征长度一致
            if mfcc.shape[1] > 100:
                mfcc = mfcc[:, :100]
                temporal = temporal[:100]
                energy = energy[:, :100]
            elif mfcc.shape[1] < 100:
                pad_width = 100 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                temporal = np.pad(temporal, (0, pad_width), mode='constant')
                energy = np.pad(energy, ((0, 0), (0, pad_width)), mode='constant')
                
            return mfcc, temporal, energy
        except Exception as e:
            print(f"提取特征失败: {audio_path}, 错误: {e}")
            # 返回随机特征作为后备
            return np.random.randn(40, 100), np.random.randn(100), np.random.randn(40, 100)

    def _resize_features(self, features, target_size):
        """调整特征图尺寸"""
        # 使用插值调整尺寸
        if len(features.shape) == 1:
            # 时域特征是一维的，需要先转换为二维
            features = features.reshape(1, -1)
            zoom_factor = (target_size[0] / features.shape[0], target_size[1] / features.shape[1])
            features = ndimage.zoom(features, zoom_factor, order=1)
            return features
        else:
            # MFCC和能量特征是二维的
            zoom_factor = (target_size[0] / features.shape[0], target_size[1] / features.shape[1])
            return ndimage.zoom(features, zoom_factor, order=1)

    def _augment(self, features):
        """增强函数 - 适用于三通道输入"""
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
            
        # 频率掩码 (仅对MFCC通道)
        if random.random() < 0.3:
            freq_mask_width = int(augmented.shape[1] * 0.1)
            freq_mask_start = random.randint(0, augmented.shape[1] - freq_mask_width)
            augmented[0, freq_mask_start:freq_mask_start+freq_mask_width, :] = 0  # 仅对MFCC通道
            
        # 时间掩码
        if random.random() < 0.3:
            time_mask_width = int(augmented.shape[2] * 0.1)
            time_mask_start = random.randint(0, augmented.shape[2] - time_mask_width)
            augmented[:, :, time_mask_start:time_mask_start+time_mask_width] = 0
        
        return np.clip(augmented, -3, 3)

# ------------------------------------------------------------------------------
# 4. 辅助函数
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
# 5. 主函数
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='基于MFCC+时域特征+能量特征的鸟类声纹识别 - Model C')
    parser.add_argument('--audio_dir', type=str, default='D:/paper/data/processed_audio',
                        help='重采样后的音频数据目录')
    parser.add_argument('--metadata', type=str, default='D:/paper/data/features/metadata.json',
                        help='元数据JSON路径')
    parser.add_argument('--label_mapping', type=str, default='D:/paper/data/features/label_mapping.json',
                        help='标签映射JSON路径')
    parser.add_argument('--output_dir', type=str, default='D:/paper/results/model_c_mfcc_temporal_energy_100',
                        help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
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
        augment_prob=0.4
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
    
    # 初始化模型 - 注意输入通道数为3
    model = ImprovedBirdNet(num_classes=num_classes, in_channels=3).to(device)
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
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
    print(f"\n开始训练Model C (MFCC + 时域特征 + 能量特征)...")
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
        'model_name': 'ImprovedBirdNet_ModelC',
        'num_classes': num_classes,
        'input_channels': 3,
        'features': ['MFCC', 'Temporal (RMSE)', 'Energy (Mel-spectrogram)'],
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