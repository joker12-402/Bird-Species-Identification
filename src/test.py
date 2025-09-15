import torch

# 检查是否有可用的GPU
if torch.cuda.is_available():
    print("PyTorch 支持 GPU 加速")
    print(f"GPU 设备数量: {torch.cuda.device_count()}")
    print(f"当前使用的 GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("PyTorch 仅支持 CPU 版本（或未检测到可用 GPU）")