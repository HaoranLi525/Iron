import os
import os.path
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None):
    assert split in ['train', 'eval', 'test']
    image_109N1_root = data_root + split + '/109N1/'
    image_5H9_root = data_root + split + '/5H9/'
    image_109N1_list = [image_109N1_root + f for f in os.listdir(image_109N1_root) if f.endswith('.tif')]
    image_5H9_list = [image_5H9_root + f for f in os.listdir(image_5H9_root) if f.endswith('.tif')]
    print("Totally {} samples in {} set.".format(len(image_109N1_list)+len(image_5H9_list), split))
    return image_109N1_list, image_5H9_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, transform=None):
        print(data_root)
        self.split = split
        self.image_109N1_list, self.image_5H9_list = make_dataset(split, data_root)
        self.data_list = self.image_109N1_list + self.image_5H9_list
        self.transform = transform
        self.name = 'Astrocyte'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        self.data_list = sorted(self.data_list)
        image_path = self.data_list[index]
        image = cv2.imread(image_path, 1)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        image = image.transpose((2,0,1))
        if image_path.split('/')[-2] == '109N1':
            label = 0
        elif image_path.split('/')[-2] == '5H9':
            label = 1

        if self.transform is not None:
            image = self.transform(image)

        return  image, label