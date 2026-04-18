import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("torchvision_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)


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
for data in dataloader:
    imgs, targets = data
    outputs = mymodule(imgs)
    # print(outputs)
    # print(targets)
    result_loss = loss(outputs, targets)
    # print(result_loss)

    result_loss.backward()
