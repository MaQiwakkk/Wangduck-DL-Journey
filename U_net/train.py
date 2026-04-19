import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import MyDataset
from unet import UNet


def train():
    # 1. 配置设备 (GPU / CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # 2. 路径设置
    train_data_path = './crack_segmentation_dataset/train'
    weight_save_path = './model_weights'

    if not os.path.exists(weight_save_path):
        os.makedirs(weight_save_path)

    # 3. 加载数据集
    dataset = MyDataset(train_data_path)
    # batch_size 根据你的显存大小调整 (显存小就改成 2 或 1)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 初始化网络 (二分类问题，num_classes=2)
    net = UNet(num_classes=2).to(device)

    # 断点训练，设定要从第几轮的权重开始续训
    resume_epoch = 6
    resume_weight_path = os.path.join(weight_save_path, f'unet_epoch_{resume_epoch}.pth')

    if resume_epoch > 0 and os.path.exists(resume_weight_path):
        net.load_state_dict(torch.load(resume_weight_path, map_location=device))
        print(f"成功加载第 {resume_epoch} 轮的权重，将从第 {resume_epoch + 1} 轮开始继续训练！")
    else:
        print("未找到指定权重或设为从头训练，初始化全新网络...")
        resume_epoch = 0  # 如果没找到，强制从0开始

    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    # 多分类/二分类标准损失函数：交叉熵
    criterion = nn.CrossEntropyLoss()

    # 6. 开始训练
    epochs = 20
    for epoch in range(resume_epoch, epochs):
        net.train()
        epoch_loss = 0

        for i, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)

            # 数据预处理
            # CrossEntropyLoss 要求标签是 LongTensor，并且值为 0 或 1。
            # 图片读取出来的 mask 背景是 0，裂缝是 255。我们需要把大于 0 的都变成 1。
            mask = mask.long()
            mask = torch.where(mask > 0, 1, 0)  # 检查 mask 张量中每一个位置的像素值。如果大于0变成1，否则还是0.

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            out = net(image)  # shape: [B, 2, 256, 256]

            # 计算损失
            loss = criterion(out, mask)

            # 反向传播 与 权重更新
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 打印当前 Epoch 平均 Loss
        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch + 1} 结束, Average Loss: {avg_loss:.4f} ---")

        # 保存每一轮的权重
        torch.save(net.state_dict(), os.path.join(weight_save_path, f'unet_epoch_{epoch + 1}.pth'))


if __name__ == '__main__':
    train()
