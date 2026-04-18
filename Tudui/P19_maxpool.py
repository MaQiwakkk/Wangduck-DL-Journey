import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

input = torch.reshape(input, (-1, 1, 5, 5))


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)  # ceil_mode=False就是floor模式

    def forward(self, input):
        output = self.maxpool1(input)  # 触发 Python 的魔法方法 __call__
        return output

tudui =Tudui()
output = tudui(input)
print(output)
