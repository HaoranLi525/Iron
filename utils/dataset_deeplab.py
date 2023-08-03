import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None):
    assert split in ['train', 'eval', 'test']
    image_1000_root = data_root + '/iron_1000/image/'
    image_500_root = data_root + '/iron_500/image/'
    image_1000_list = [image_1000_root + f for f in os.listdir(image_1000_root) if f.endswith('.png')]
    image_500_list = [image_500_root + f for f in os.listdir(image_500_root) if f.endswith('.png')]
    if split == 'train':
        print("Totally {} samples in {} set.".format(len(image_1000_list)+len(image_500_list[0:200]), split))
    else:
        print("Totally {} samples in {} set.".format(len(image_500_list[200:]), split))
    return image_1000_list, image_500_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, transform=None):
        self.split = split
        self.image_1000_list, self.image_500_list = make_dataset(split, data_root)
        if self.split == 'train':
            self.data_list = self.image_1000_list + self.image_500_list[0:200]
        else:
            self.data_list = self.image_500_list[200:]

        self.transform = transform
        self.name = 'Iron'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        self.data_list = sorted(self.data_list)
        image_path = self.data_list[index]
        label_path = './dataset/' + image_path.split('/')[-3] + '/label/' + image_path.split('/')[-1]
        image = Image.open(image_path).convert('RGB')  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return  image, label