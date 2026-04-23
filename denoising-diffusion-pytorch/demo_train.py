import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 1. 初始化 U-Net
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4),
    flash_attn = False  # M4 建议设为 False 避坑
)

# 2. 初始化扩散逻辑
diffusion = GaussianDiffusion(
    model,
    image_size = 64,          # 裂缝图片会被自动缩放到这个尺寸
    timesteps = 1000,          # 训练总步数T=1000
    sampling_timesteps = 250    # 采样时用 DDIM 加速，只要 250 步，快 4 倍
)

# 3. 初始化训练器 (管家)
trainer = Trainer(
    diffusion,
    './train_demo_dataset',            # 指向你刚刚放图片的文件夹
    train_batch_size = 2,      # M4 建议从 4 开始，如果显存够大再加
    train_lr = 8e-5,           # 学习率
    train_num_steps = 2000,   # 先设 1 万步看看效果，后续可以增加
    gradient_accumulate_every = 8,
    ema_decay = 0.995,
    amp = False,               # MPS 下设为 False 更稳定
    calculate_fid = False,     # 100 张图，没必要算 FID
    results_folder = './results_crack' # 结果存到这个新文件夹
)

if __name__ == '__main__':
    trainer.train()