from torch.utils.tensorboard import SummaryWriter
# add_scalar()函数的使用

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=x+10", i + 10, i)

writer.close()
