# TODO nothing to do
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./torchvision_dataset", train=True, transform=dataset_transform,
                                         download=True)
test_set = torchvision.datasets.CIFAR10(root="./torchvision_dataset", train=False, transform=dataset_transform,
                                        download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("logs_P14")
for i in range(100):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
# tensorboard --logdir=xxx
