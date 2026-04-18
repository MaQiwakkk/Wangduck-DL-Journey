import torch
from torch import nn


def copy_and_crop(tensor_to_crop, target_tensor):
    """
    实现原图中的灰色箭头: 'copy and crop'
    tensor_to_crop: 编码器特征图 E_i (大尺寸)
    target_tensor: 解码器上采样特征图 U_i (小尺寸)
    """
    _, _, H_E, W_E = tensor_to_crop.shape
    _, _, H_U, W_U = target_tensor.shape

    # 计算需要裁剪的像素偏移量 (delta h, delta w)
    delta_h = H_E - H_U
    delta_w = W_E - W_U

    # 在空间维度进行中心切片
    return tensor_to_crop[:, :,
    delta_h // 2: delta_h // 2 + H_U,
    delta_w // 2: delta_w // 2 + W_U]


class DoubleConv(nn.Module):
    """原图中的两个连续蓝色箭头: conv 3x3, ReLU (无 padding)"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet_Original(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        # ---------------- 编码器 Contracting Path ----------------
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 红色箭头

        self.down_conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------- 瓶颈层 Bottleneck ----------------
        self.bottleneck = DoubleConv(512, 1024)

        # ---------------- 解码器 Expanding Path ----------------
        # 绿色箭头: up-conv 2x2
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # 拼接后通道数为 512(crop) + 512(up) = 1024
        self.up_conv1 = DoubleConv(1024, 512)

        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)

        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)

        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # ---------------- 输出层 ----------------
        # 蓝绿色箭头: conv 1x1
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码器提取特征并保留跨层连接信息
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(self.pool1(x1))
        x3 = self.down_conv3(self.pool2(x2))
        x4 = self.down_conv4(self.pool3(x3))

        b = self.bottleneck(self.pool4(x4))

        # 解码器，包含升采样、裁剪对齐、拼接融合
        u1 = self.up_trans1(b)
        x4_crop = copy_and_crop(x4, u1)
        u1_out = self.up_conv1(torch.cat([x4_crop, u1], dim=1))

        u2 = self.up_trans2(u1_out)
        x3_crop = copy_and_crop(x3, u2)
        u2_out = self.up_conv2(torch.cat([x3_crop, u2], dim=1))

        u3 = self.up_trans3(u2_out)
        x2_crop = copy_and_crop(x2, u3)
        u3_out = self.up_conv3(torch.cat([x2_crop, u3], dim=1))

        u4 = self.up_trans4(u3_out)
        x1_crop = copy_and_crop(x1, u4)
        u4_out = self.up_conv4(torch.cat([x1_crop, u4], dim=1))

        return self.out(u4_out)


if __name__ == '__main__':
    # 严格按照原始U-Net结构标注的输入尺寸：572x572，单通道输入
    x = torch.randn(1, 1, 572, 572)
    net = UNet_Original(in_channels=1, num_classes=2)
    out = net(x)

    # 按照图示，验证输出的尺寸是否为 388x388 的 segmentation map
    print(f"输入特征维度: {x.shape}")
    print(f"输出特征维度: {out.shape}")