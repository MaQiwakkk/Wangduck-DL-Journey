import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs_P24")


class Mymdoule(nn.Module):
    def __init__(self):
        super(Mymdoule, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10))

    def forward(self, x):
        x = self.model1(x)
        return x


# 创建模型实例
mymodule = Mymdoule()

# 交叉熵损失
loss = nn.CrossEntropyLoss()

# 设置优化器
optim = torch.optim.SGD(mymodule.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = mymodule(imgs)
        result_loss = loss(outputs, targets)
        # 梯度清零
        optim.zero_grad()
        # 反向传播，计算每一个节点的梯度
        result_loss.backward()
        # 对每一个参数调优
        optim.step()

        running_loss = running_loss + result_loss.item()
    print(running_loss)
    writer.add_scalar("Training Loss", running_loss, epoch)

writer.close()
