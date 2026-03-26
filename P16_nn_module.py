import torch
from torch import nn


class Mymodule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

mymodule = Mymodule()
x = torch.tensor(1.0)
output = mymodule(x) # 有父类的__call__，这个类的实例可以像函数一样被“调用”
print(output)
