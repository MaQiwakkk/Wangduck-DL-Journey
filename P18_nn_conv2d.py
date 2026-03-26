import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./torchvision_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()

writer = SummaryWriter("logs_P18")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    #print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    #print(output.shape)  # torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))  # reshape为3通道
    writer.add_images("output", output, step)
    step = step + 1
