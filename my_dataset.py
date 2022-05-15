import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


def split_train_valid(root_path, val_ratio=0.2):

    class_names = [class_name for class_name in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, class_name))]
    class_indices = dict((v, k) for k, v in enumerate(class_names))

    json_str = json.dumps(dict((ind, name) for name, ind in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as file:
        file.write(json_str)
    img_format = ['jpg', 'png', 'JPG', 'PNG']

    train_img_path = []
    train_img_label = []
    valid_img_path = []
    valid_img_label = []
    total_img_num = 0
    for name in class_names:
        class_path = os.path.join(root_path, name)
        assert os.path.exists(class_path), f'{class_path} path is not exists...'

        img_path_list = [os.path.join(class_path, path) for path in os.listdir(class_path)
                         if path.split('.')[-1] in img_format]

        total_img_num += len(img_path_list)
        img_class = class_indices[name]
        class_num = len(img_path_list)
        val_list = random.sample(img_path_list, k=int(class_num * val_ratio))

        for img_path in img_path_list:

            if img_path in val_list:
                valid_img_path.append(img_path)
                valid_img_label.append(img_class)
            else:
                train_img_path.append(img_path)
                train_img_label.append(img_class)

    print("{} images were found in the dataset.".format(total_img_num))
    print("{} images for training.".format(len(train_img_path)))
    print("{} images for validation.".format(len(valid_img_path)))

    return train_img_path, train_img_label, valid_img_path, valid_img_label


class MyDataset(Dataset):

    def __init__(self, img_path_list, img_label_list, transform):
        super(MyDataset, self).__init__()

        self.img_path_list = img_path_list
        self.img_label_list = img_label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, item):
        image = Image.open(self.img_path_list[item])
        label = self.img_label_list[item]

        if self.transform:
            image = self.transform(image)
        return image, label

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels)

        return imgs, labels