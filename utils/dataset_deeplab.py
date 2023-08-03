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
    image_root = data_root + '/' + split + '/image/'
    image_list = [image_root + f for f in os.listdir(image_root) if f.endswith('.png')]
    print("Totally {} samples in {} set.".format(len(image_list), split))
    return image_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root)
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
