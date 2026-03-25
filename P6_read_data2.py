import os
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):
    def __init__(self, root_dir, img_dir, label_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir

        # 拼接出两个完整的路径
        self.img_path = os.path.join(self.root_dir, self.img_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)

        # 获取所有图片的文件名列表
        self.img_list = os.listdir(self.img_path)

    def __getitem__(self, idx):
        # 1. 获取图片
        img_name = self.img_list[idx]
        img_item_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_item_path)

        # 2. 获取对应的标签文件（文件名一样，只是后缀是 .txt）
        label_name = img_name.split('.')[0] + ".txt"
        label_item_path = os.path.join(self.label_path, label_name)

        # 3. 读取 TXT 里的内容
        with open(label_item_path, 'r') as f:
            label = f.read().strip()  # 读取并去掉换行符

            return img, label

    def __len__(self):
        return len(self.img_list)


root_dir = "dataset2/train"
ants_img_dir = "ants_image"
ants_label_dir = "ants_label"
ants_dataset = MyData(root_dir, ants_img_dir, ants_label_dir)

img, label = ants_dataset[0]

print(label)
img.show()
