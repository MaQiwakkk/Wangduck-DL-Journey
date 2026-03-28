import torchvision
from torch import nn
from torchvision.models import VGG16_Weights

vgg16_false = torchvision.models.vgg16(weights=None)  # 不加载预训练权重 pretrained=False是过时写法
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)  # 加载预训练权重 pretrained=True是过时写法

print(vgg16_false)
print("===========================================================")
print(vgg16_true)


## 为了应用于 CIFAR10 这个10分类数据集，要对vgg进行一些修改，有两种方法：
# 1.直接修改vgg16的最后一个线性层4096->1000 变为4096->10
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

# 2.在原有的vgg16模型后面加一个1000 -> 10的线性层
vgg16_true.add_module("add_linear", nn.Linear(1000, 10))
# 或者：
# vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)
