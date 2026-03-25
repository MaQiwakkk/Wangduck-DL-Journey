from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img.shape)
# print(tensor_img[0][0][0])

writer = SummaryWriter("logs_P9")
writer.add_image("tensor_img", tensor_img)
writer.close()
# tensorboard --logdir=logs

