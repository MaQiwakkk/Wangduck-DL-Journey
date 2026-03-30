import torch
import torchvision

# 先创建一个模型，以vgg16为例
vgg16 = torchvision.models.vgg16(weights=None)

# 保存方式1，模型结构 + 模型参数
torch.save(vgg16,"vgg16_save1.pth")

# 保存方式2，只保留模型参数（所占空间小，官方推荐）
torch.save(vgg16.state_dict(),"vgg16_save2.pth")


