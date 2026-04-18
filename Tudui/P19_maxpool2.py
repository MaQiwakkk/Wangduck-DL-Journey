import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # ceil_mode=False就是floor模式

    def forward(self, input):
        output = self.maxpool1(input)  # 触发 Python 的魔法方法 __call__
        return output


tudui = Tudui()
writer = SummaryWriter("logs_P19")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = tudui(imgs)
    writer.add_images("output_pool", output, step)

    step = step + 1

writer.close()