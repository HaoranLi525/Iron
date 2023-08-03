import os
import os.path as osp
import json
import labelme
import matplotlib.pyplot as plt
import cv2
from PIL import Image     

data_root = '/home/guest/Downloads/Iron/dataset'
png_500_folder = '/home/guest/Downloads/Iron/dataset/iron_500/label/'
img_500_folder = '/home/guest/Downloads/Iron/dataset/iron_500/image/'
png_1000_folder = '/home/guest/Downloads/Iron/dataset/iron_1000/label/'
img_1000_folder = '/home/guest/Downloads/Iron/dataset/iron_1000/image/'

image_1000_root = data_root + '/iron_1000/'
image_500_root = data_root + '/iron_500/'
image_1000_list = [image_1000_root + f for f in os.listdir(image_1000_root) if f.endswith('.json')]
image_500_list = [image_500_root + f for f in os.listdir(image_500_root) if f.endswith('.json')]
image_1000_list = sorted(image_1000_list)
image_500_list = sorted(image_500_list)
i = 1
j = 1
'''
for json_file in image_500_list:
    if i <= 304:
        pass
    else:
        os.system("labelme_json_to_dataset {}".format(json_file))
        image_path = osp.join(image_500_root, json_file.split(".")[-2] + "_json/img.png")
        label_path = osp.join(image_500_root, json_file.split(".")[-2] + "_json/label.png")
        img_save_path = osp.join(img_500_folder, json_file.split(".")[-2].split("/")[-1] + ".png")
        png_save_path = osp.join(png_500_folder, json_file.split(".")[-2].split("/")[-1] + ".png")
        label_png = Image.open(label_path)
        label_png.save(png_save_path)
        image_png = cv2.imread(image_path)
        cv2.imwrite(img_save_path, image_png)
        print('Iron 500 File %d Convert success!\n'%i, flush=True)
    i = i + 1
'''

for json_file in image_1000_list:
    if j <= 997:
        pass
    else:
        os.system("labelme_json_to_dataset {}".format(json_file))
        image_path = osp.join(image_1000_root, json_file.split(".")[-2] + "_json/img.png")
        label_path = osp.join(image_1000_root, json_file.split(".")[-2] + "_json/label.png")
        img_save_path = osp.join(img_1000_folder, json_file.split(".")[-2].split("/")[-1] + ".png")
        png_save_path = osp.join(png_1000_folder, json_file.split(".")[-2].split("/")[-1] + ".png")
        label_png = Image.open(label_path)
        label_png.save(png_save_path)
        image_png = cv2.imread(image_path)
        cv2.imwrite(img_save_path, image_png)
        print('Iron 1000 File %d Convert success!\n'%j, flush=True)
    j = j + 1