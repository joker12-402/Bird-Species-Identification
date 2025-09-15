import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter, deque
import random
import time

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
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 10

# ------------------------------------------------------------------------------
# 1. 数据集类：处理特征加载和数据增强（添加了特征标准化）
# ------------------------------------------------------------------------------
class BirdSoundDataset(Dataset):
    def __init__(self, metadata_path, features_dir, transform=None, is_train=False, augment_prob=0.8):
        """
        鸟类音频特征数据集
        :param metadata_path: 元数据JSON文件路径
        :param features_dir: 特征文件根目录
        :param transform: 数据预处理变换
        :param is_train: 是否为训练集（控制数据增强）
        :param augment_prob: 数据增强概率
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.features_dir = features_dir
        self.transform = transform
        self.is_train = is_train
        self.augment_prob = augment_prob
        
        # 提取所有标签用于计算类别权重
        self.labels = [item['label'] for item in self.metadata]
        self.num_classes = len(set(self.labels))
        
        # 加载特征统计量（特征标准化）
        try:
            with open('feature_stats.json', 'r') as f:
                stats = json.load(f)
            self.feature_mean = stats['mean']
            self.feature_std = stats['std']
            print(f"已加载特征统计量: 均值={self.feature_mean:.4f}, 标准差={self.feature_std:.4f}")
            print(f"特征范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            
            # 检查标准差是否过小（可能导致数值问题）
            if self.feature_std < 1e-6:
                print(f"警告: 标准差非常小 ({self.feature_std:.6f})，这可能导致数值不稳定")
                self.feature_std = 1.0  # 防止除以零
                
        except FileNotFoundError:
            print("错误: 未找到 feature_stats.json 文件，请先运行 calculate_stats.py")
            raise
        except Exception as e:
            print(f"加载特征统计量时出错: {e}")
            raise

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        feature_path = os.path.join(self.features_dir, item['file_path'])
        
        # 加载特征
        try:
            features = np.load(feature_path)
        except Exception as e:
            print(f"加载特征文件失败: {feature_path}, 错误: {e}")
            # 返回一个随机特征作为替代（仅用于调试）
            features = np.random.randn(180, 100)
        
        # 特征标准化 - 关键步骤！
        features = (features - self.feature_mean) / self.feature_std
        
        features = features[np.newaxis, ...].astype(np.float32)
        
        # 训练集数据增强
        if self.is_train and random.random() < self.augment_prob:
            features = self._augment(features)
        
        # 应用变换
        if self.transform:
            features = self.transform(features)
            
        return torch.from_numpy(features), torch.tensor(item['label'], dtype=torch.long)

    def _augment(self, features):
        """增强函数：时间偏移、噪声注入、频率缩放、音量调整"""
        augmented = features.copy()
        
        # 1. 时间轴随机偏移（±15%）
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        
        # 2. 随机噪声注入
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.05, augmented.shape)  # 减小噪声强度
            augmented += noise
        
        # 3. 频率轴局部缩放
        if random.random() < 0.4:
            start = random.randint(0, augmented.shape[1]//2)
            end = start + random.randint(augmented.shape[1]//6, augmented.shape[1]//3)
            end = min(end, augmented.shape[1])
            scale = 0.8 + random.random() * 0.4  # 缩小缩放范围
            augmented[:, start:end, :] *= scale
        
        # 4. 音量缩放
        if random.random() < 0.4:
            volume_scale = 0.8 + random.random() * 0.4  # 缩小缩放范围
            augmented *= volume_scale
        
        return np.clip(augmented, -3, 3)  # 标准化后范围缩小，调整clip范围

# ------------------------------------------------------------------------------
# 2. 简化模型架构
# ------------------------------------------------------------------------------
class SimpleBirdNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleBirdNet, self).__init__()
        
        self.features = nn.Sequential(
            # 输入: (1, 180, 100)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (32, 90, 50)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 45, 25)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (128, 22, 12)
        )
        
        # 计算全连接层输入尺寸
        self._init_weights()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 22 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def _init_weights(self):
        # 初始化权重
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
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------------------------------------------------------------
# 3. 焦点损失实现（解决数据不平衡）
# ------------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ------------------------------------------------------------------------------
# 4. 辅助函数：类别权重计算、训练/验证、可视化
# ------------------------------------------------------------------------------
def calculate_class_weights(labels):
    """计算类别权重（样本越少权重越高）"""
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    
    return torch.tensor(weights / weights.sum())  # 归一化

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练单个epoch"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    batch_count = 0
    
    pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 每10个batch打印一次损失
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"Epoch {epoch:2d} | Batch {batch_count:3d} | 损失: {loss.item():.4f}")
        
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
        # 确保标签数量一致
        unique_labels = np.unique(all_labels + list(range(len(label_names))))
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
    # 绘制平滑曲线（展示趋势）
    if len(history['val_acc']) >= 3:
        val_smoothed = np.convolve(history['val_acc'], np.ones(3)/3, mode='valid')
        plt.plot(range(1, len(val_smoothed)+1), val_smoothed, '--', color='#2ca02c', label='验证准确率(平滑)')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='训练F1', color='#1f77b4')
    plt.plot(history['val_f1'], label='验证F1', color='#ff7f0e')
    # 绘制平滑曲线（展示趋势）
    if len(history['val_f1']) >= 3:
        f1_smoothed = np.convolve(history['val_f1'], np.ones(3)/3, mode='valid')
        plt.plot(range(1, len(f1_smoothed)+1), f1_smoothed, '--', color='#2ca02c', label='验证F1(平滑)')
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
    # 确保标签数量一致
    unique_labels = np.unique(y_true + y_pred + list(range(len(label_names))))
    
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
# 5. 主函数：训练流程控制
# ------------------------------------------------------------------------------
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='鸟类物种识别模型训练')
    parser.add_argument('--features_dir', type=str, default='D:/paper/data/features',
                        help='特征数据目录')
    parser.add_argument('--metadata', type=str, default='D:/paper/data/features/metadata.json',
                        help='元数据JSON路径')
    parser.add_argument('--label_mapping', type=str, default='D:/paper/data/features/label_mapping.json',
                        help='标签映射JSON路径')
    parser.add_argument('--output_dir', type=str, default='D:/paper/results/model',
                        help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='初始学习率')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--augment_prob', type=float, default=0.5, help='数据增强概率')  # 降低增强概率
    parser.add_argument('--gamma', type=float, default=1.5, help='Focal Loss的gamma参数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--smooth_window', type=int, default=3, help='性能平滑窗口大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        features_dir=args.features_dir,
        is_train=True,
        augment_prob=args.augment_prob
    )
    
    # 使用分层抽样确保每个类别在训练集、验证集和测试集中都有代表
    print("使用分层抽样划分数据集...")
    labels = full_dataset.labels
    
    # 首先划分训练集和临时集（验证+测试）
    train_idx, temp_idx = train_test_split(
        range(len(full_dataset)),
        test_size=args.val_split + args.test_split,
        stratify=labels,
        random_state=SEED
    )
    
    # 然后从临时集中划分验证集和测试集
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=args.test_split / (args.val_split + args.test_split),
        stratify=temp_labels,
        random_state=SEED
    )
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    # 设置数据集的模式
    # 注意：由于我们使用的是Subset，需要直接修改原始数据集的属性
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
    
    # 初始化模型、损失函数和优化器
    model = SimpleBirdNet(num_classes=num_classes).to(device)
    class_weights = class_weights.to(device)
    
    # 使用标准交叉熵损失而不是Focal Loss，以便调试
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 使用AdamW优化器，更稳定
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 初始化训练状态变量
    start_epoch = 1
    best_val_f1 = 0.0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
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
    
    # 开始训练
    print("\n开始训练...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # 训练
            train_loss, train_acc, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # 验证
            val_loss, val_acc, val_f1, _, _ = validate(
                model, val_loader, criterion, device, label_names
            )
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # 更新学习率
            scheduler.step(val_f1)
            
            # 打印epoch摘要
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.4f}, F1={train_f1:.4f}")
            print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}, F1={val_f1:.4f}")
            
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
    
    print(f"\n所有结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()