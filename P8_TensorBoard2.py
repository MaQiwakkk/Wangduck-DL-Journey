from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# add_image()函数的使用

writer = SummaryWriter("logs_P8")
image_path = "dataset2/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(img_array.shape)
writer.add_image("test", img_array, 1, dataformats='HWC')
writer.close()

# tensorboard --logdir=logs