import torch
import torchvision
from torch.utils.data import DataLoader
from CIFAR10_model import *
import time

# 如果有英伟达显卡就用cuda，如果是Mac就用mps，都没有就用cpu
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"当前使用的设备是: {device}")

# 导入CIFAR10数据集
train_data = torchvision.datasets.CIFAR10("torchvision_dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("torchvision_dataset", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# 打印长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
cm = CIFAR10_mdoule()
cm = cm.to(device)  # 调用GPU算力

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # 调用GPU算力

# 优化器
a = 0.01  # 学习率
optimizer = torch.optim.SGD(cm.parameters(), lr=a)

# 记录训练和测试的总次数
total_train_step = 0
total_test_step = 0
# 训练轮数
epoch = 20

for i in range(epoch):
    start_time = time.time()
    ## 训练步骤
    cm.train()
    print("-----第{}轮训练开始-----".format(i + 1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)  # 调用GPU算力
        targets = targets.to(device)  # 调用GPU算力
        outputs = cm(imgs)
        loss = loss_fn(outputs, targets)

        # 梯度清零
        optimizer.zero_grad()  # 因为 PyTorch 默认会累加梯度，如果不清零，上一轮的误差会干扰这一轮的计算
        # 反向传播
        loss.backward()  # 根据当前的 Loss，计算模型中每一个参数（W 和 b）对误差的“贡献度”（即梯度）。
        # 参数更新
        optimizer.step()  # 优化器根据刚才算出的梯度，结合学习率，真正动手去修改模型内部的参数。
        # 统计训练次数
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，损失：{}".format(total_train_step, loss.item()))

    ## 测试步骤
    cm.eval()
    total_test_loss = 0  # 记录10000张测试集上的整体损失
    total_correct = 0  # 记录准确个数
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 调用GPU算力
            targets = targets.to(device)  # 调用GPU算力
            outputs = cm(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss = total_test_loss + loss.item()  # 更新整体损失

            correct = (outputs.argmax(1) == targets).sum().item()  # 一个batch size内正确个数
            total_correct = total_correct + correct  # 更新总个数
    print("测试集上的Loss：{}".format(total_test_loss))
    print("测试集上的Acc：{:.2f}%".format(total_correct / test_data_size * 100))
    end_time = time.time()
    print("第 {} 轮耗时：{:.2f} 秒".format(i + 1, end_time - start_time))
