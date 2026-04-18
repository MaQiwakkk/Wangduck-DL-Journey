import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kenerl = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# reshape一下格式，前两位加入 batch size， 通道数
input = torch.reshape(input, (1, 1, 5, 5))
kenerl = torch.reshape(kenerl, (1, 1, 3, 3))

print(input)
print(kenerl)

output1 = F.conv2d(input, kenerl, stride=1)
print(output1)

output2 = F.conv2d(input, kenerl, stride=2)
print(output2)

# 边缘扩容padding = 1
output3 = F.conv2d(input, kenerl, stride=1, padding=1)
print(output3)
