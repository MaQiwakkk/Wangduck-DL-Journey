import torch
import torchvision

# 加载方式1，需要手动关闭安全检查。
model1 = torch.load("vgg16_save1.pth", weights_only=False)
print(model1)

# 加载方式2
model2 = torchvision.models.vgg16(weights=None)  # 先实例化模型
model2.load_state_dict(torch.load("vgg16_save2.pth"))
print(model2)
