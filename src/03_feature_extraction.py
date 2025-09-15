import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import argparse
import json
from scipy.signal import lfilter, butter
from scipy.fftpack import dct

# 鸟类ID到名称的映射
BIRD_MAPPING = {
    '0009': '灰雁',
    '0017': '大天鹅',
    '0034': '绿头鸭',
    '0036': '绿翅鸭',
    '0074': '灰山鹳',
    '0077': '西鹳鹳',
    '0114': '雄鸡',
    '0121': '红喉潜鸟',
    '0180': '苍鹭',
    '0202': '普通鸬鹚',
    '0235': '苍鹰',
    '0257': '欧亚属',
    '0265': '西方秧鸡',
    '0281': '骨顶鸡',
    '0298': '黑翅长脚鹳',
    '0300': '凤头麦鸡',
    '0364': '白腰草鹳',
    '0368': '红脚鹳',
    '0370': '林鹳',
    '1331': '麻雀'
}

# 创建ID到数字标签的映射
ID_TO_LABEL = {bird_id: idx for idx, bird_id in enumerate(sorted(BIRD_MAPPING.keys()))}
LABEL_TO_ID = {idx: bird_id for bird_id, idx in ID_TO_LABEL.items()}
LABEL_TO_NAME = {idx: BIRD_MAPPING[bird_id] for bird_id, idx in ID_TO_LABEL.items()}

def butterworth_filter(signal, sr, cutoff=8000, order=5):
    """
    巴特沃斯低通滤波器（修复版）
    增加参数校验和边界处理，确保临界频率在合法范围
    """
    # 校验采样率有效性
    if sr <= 0:
        raise ValueError(f"无效的采样率: {sr} Hz，必须为正数")
    
    # 计算奈奎斯特频率
    nyquist = 0.5 * sr
    
    # 调整截止频率，确保不超过奈奎斯特频率
    if cutoff >= nyquist:
        adjusted_cutoff = nyquist * 0.99  # 安全边界，避免等于奈奎斯特频率
        print(f"警告：截止频率({cutoff}Hz)超过奈奎斯特频率({nyquist}Hz)，已调整为{adjusted_cutoff:.1f}Hz")
        cutoff = adjusted_cutoff
    elif cutoff <= 0:
        adjusted_cutoff = 100  # 设置最小合理截止频率
        print(f"警告：无效的截止频率({cutoff}Hz)，已调整为{adjusted_cutoff}Hz")
        cutoff = adjusted_cutoff
    
    # 计算归一化临界频率并确保在(0,1)范围内
    Wn = cutoff / nyquist
    if not (0 < Wn < 1):
        Wn = max(0.01, min(0.99, Wn))  # 强制限制在合法范围
        print(f"警告：归一化频率Wn={Wn:.3f}超出范围，已调整为合法值")
    
    # 生成滤波器并应用
    b, a = butter(order, Wn, btype='low', analog=False)
    return lfilter(b, a, signal)

def extract_pncc(audio_path, n_fft=3200, hop_length=320, n_mfcc=13, n_bands=40, fmin=50):
    """提取PNCC特征（增加异常处理）"""
    try:
        # 加载音频并强制转为16000Hz
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 检查音频是否为空
        if len(y) == 0:
            raise ValueError("音频文件内容为空")
        
        # 预加重处理
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # 应用巴特沃斯滤波（使用修复后的函数）
        y = butterworth_filter(y, sr)
        
        # 调整n_fft以适应音频长度
        n_fft = min(n_fft, len(y))
        if n_fft < 1024:  # 确保n_fft不会过小
            n_fft = 1024
            print(f"警告：音频过短，已将n_fft调整为{1024}")
        
        # 计算短时傅里叶变换和功率谱
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hamming')
        power_spec = np.abs(stft) ** 2
        
        # 生成梅尔滤波组（确保参数合理）
        mel_basis = librosa.filters.mel(
            sr=sr, 
            n_fft=n_fft, 
            n_mels=n_bands, 
            fmin=fmin, 
            fmax=0.5*sr,  # 明确使用奈奎斯特频率
            htk=True
        )
        mel_spec = np.dot(mel_basis, power_spec)
        
        # 功率归一化（PNCC核心步骤）
        geometric_mean = np.exp(np.mean(np.log(mel_spec + 1e-10), axis=1, keepdims=True))
        k = 1.0
        pncc_spec = np.log(np.maximum(mel_spec, k * geometric_mean) + 1e-10)
        
        # 倒谱均值减
        pncc_spec = pncc_spec - np.mean(pncc_spec, axis=1, keepdims=True)
        
        # DCT变换获取倒谱系数
        pncc = dct(pncc_spec, type=2, norm='ortho', axis=0)[:n_mfcc]
        
        # 去除直流分量
        if n_mfcc > 0:
            pncc = pncc[1:]
        
        return pncc
    
    except Exception as e:
        raise RuntimeError(f"PNCC特征提取失败: {str(e)}") from e

def extract_mfcc(audio_path, n_mfcc=13, n_fft=3200, hop_length=320):
    """提取MFCC特征及其一阶、二阶差分"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        if len(y) == 0:
            raise ValueError("音频文件内容为空")
        
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
            fmax=0.5*sr  # 使用奈奎斯特频率
        )
        
        # 计算一阶和二阶差分
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return mfcc, mfcc_delta, mfcc_delta2
    
    except Exception as e:
        raise RuntimeError(f"MFCC特征提取失败: {str(e)}") from e

def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=3200, hop_length=320):
    """提取Mel频谱图特征"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        if len(y) == 0:
            raise ValueError("音频文件内容为空")
        
        # 预加重
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        
        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length,
            fmax=0.5*sr
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    except Exception as e:
        raise RuntimeError(f"梅尔频谱提取失败: {str(e)}") from e

def extract_rms_energy(audio_path, hop_length=320):
    """提取RMS能量特征"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        if len(y) == 0:
            raise ValueError("音频文件内容为空")
        
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        return rms
    
    except Exception as e:
        raise RuntimeError(f"RMS能量提取失败: {str(e)}") from e

def extract_mtpfe_features(audio_path):
    """提取MTPFE特征组合"""
    try:
        mel_spec = extract_mel_spectrogram(audio_path)
        mfcc, mfcc_delta, mfcc_delta2 = extract_mfcc(audio_path)
        pncc = extract_pncc(audio_path)
        rms = extract_rms_energy(audio_path)
        
        # 拼接所有特征
        temporal_features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        energy_features = np.concatenate([pncc, rms], axis=0)
        mtpfe_features = np.concatenate([mel_spec, temporal_features, energy_features], axis=0)
        
        # 统一特征长度为100
        if mtpfe_features.shape[1] > 100:
            mtpfe_features = mtpfe_features[:, :100]
        elif mtpfe_features.shape[1] < 100:
            pad_width = 100 - mtpfe_features.shape[1]
            mtpfe_features = np.pad(mtpfe_features, ((0, 0), (0, pad_width)), mode='constant')
        
        return mtpfe_features
    
    except Exception as e:
        raise RuntimeError(f"MTPFE特征组合失败: {str(e)}") from e

def process_audio_file(audio_path, output_dir, category):
    """处理单个音频文件并保存特征"""
    try:
        features = extract_mtpfe_features(audio_path)
        
        # 确保输出目录存在
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_category_dir, f"{base_name}.npy")
        
        # 保存特征
        np.save(output_path, features)
        
        return {
            "file_path": os.path.relpath(output_path, output_dir),
            "category": category,
            "label": ID_TO_LABEL[category],
            "species": BIRD_MAPPING[category]
        }
    
    except Exception as e:
        print(f"处理文件 {audio_path} 时出错: {e}")
        return None

def extract_features(input_dir, output_dir):
    """提取整个数据集的特征"""
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"找到 {len(categories)} 个类别")
    
    metadata = []
    processed_count = 0
    error_count = 0
    
    # 计算总文件数
    total_files = sum(
        len([f for f in os.listdir(os.path.join(input_dir, cat)) if f.endswith('.wav')]) 
        for cat in categories
    )
    
    # 创建进度条
    pbar = tqdm(total=total_files, desc="提取特征")
    
    # 遍历每个类别
    for category in categories:
        if category not in ID_TO_LABEL:
            print(f"警告: 类别 {category} 不在映射表中，跳过")
            continue
            
        category_path = os.path.join(input_dir, category)
        audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        
        # 处理每个音频文件
        for audio_file in audio_files:
            input_path = os.path.join(category_path, audio_file)
            result = process_audio_file(input_path, output_dir, category)
            
            if result:
                metadata.append(result)
                processed_count += 1
            else:
                error_count += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # 保存标签映射
    mapping_path = os.path.join(output_dir, "label_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({
            "id_to_label": ID_TO_LABEL,
            "label_to_id": LABEL_TO_ID,
            "label_to_name": LABEL_TO_NAME
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n特征提取完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"元数据保存至: {metadata_path}")
    print(f"标签映射保存至: {mapping_path}")
    
    return metadata

def verify_features(output_dir, sample_size=5):
    """验证提取的特征"""
    print("\n验证特征提取结果...")
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print("元数据文件不存在，无法验证")
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not metadata:
        print("没有提取到任何特征数据")
        return False
    
    # 随机抽样检查
    if len(metadata) > sample_size:
        sample_files = np.random.choice(metadata, sample_size, replace=False)
    else:
        sample_files = metadata
    
    # 检查特征文件
    all_valid = True
    for item in sample_files:
        feature_path = os.path.join(output_dir, item["file_path"])
        try:
            features = np.load(feature_path)
            if features.shape != (180, 100):
                print(f"特征形状异常: {item['file_path']} 形状={features.shape} (预期(180,100))")
                all_valid = False
            else:
                print(f"{item['file_path']}: 形状正确={features.shape}, 标签={item['label']}({item['species']})")
        except Exception as e:
            print(f"无法加载特征文件 {feature_path}: {e}")
            all_valid = False
    
    return all_valid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='提取鸟类音频的MTPFE特征（包含PNCC）')
    parser.add_argument('--input', type=str, default='D:/paper/data/processed_audio',
                        help='输入数据目录路径（重采样后的音频）')
    parser.add_argument('--output', type=str, default='D:/paper/data/features',
                        help='输出数据目录路径')
    parser.add_argument('--verify', action='store_true',
                        help='处理完成后验证结果')
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("=" * 60)
    print("鸟类音频特征提取 (MTPFE) - 适配16000Hz采样率")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"目标特征维度: (180, 100)")
    print(f"采样率: 16000Hz")
    print("=" * 60)
    
    # 执行特征提取
    metadata = extract_features(args.input, args.output)
    
    # 验证处理结果
    if args.verify:
        verify_features(output_dir=args.output)

if __name__ == "__main__":
    main()
    