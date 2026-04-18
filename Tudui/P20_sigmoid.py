import torch
import torchvision.datasets
from torch import nn
from torch.nn import  Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)



class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

tudui = Tudui()
writer = SummaryWriter("logs_P20")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    output = tudui(imgs)
    writer.add_images("output_sigmoid", output, step)

    step = step + 1

writer.close()



