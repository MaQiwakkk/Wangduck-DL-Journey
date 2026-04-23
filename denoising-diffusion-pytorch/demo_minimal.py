import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# 指定加速设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False  # M4 暂时建议设为 False，避免兼容性问题
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000
).to(device)

# 模拟 8 张 128x128 的随机图片
training_images = torch.rand(8, 3, 128, 128).to(device)

# 跑一次前向传播（训练一步）
loss = diffusion(training_images)
loss.backward()
print(f"Loss: {loss.item()}")

# 尝试生成（采样）
print("Sampling...")
sampled_images = diffusion.sample(batch_size=4)
print(f"Sampled shape: {sampled_images.shape}")
