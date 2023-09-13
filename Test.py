import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import transforms
from torch import nn
import numpy as np
from datetime import datetime
from lib.Deeplab import Deeplab
from utils import dataset_deeplab as dataset
from utils.metrics import StreamSegMetrics
from utils.utils import clip_gradient, adjust_lr
from utils import transform as tr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='number of iron types')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='/home/guest/Downloads/Iron/savedModel/Deeplab/Net_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--data_root', type=str, default='./dataset', help='images root')
parser.add_argument('--save_path', type=str, default='./savedModel/Deeplab/', help='the path to save model and log')
opt = parser.parse_args()

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0', flush=True)
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1', flush=True)
cudnn.benchmark = True

# build the model
model = Deeplab(num_classes=opt.num_classes, backbone='mobilenet', output_stride=16,sync_bn=True, freeze_bn=False)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {str(total_params)[:-6] + '.' + str(total_params)[-6:-4] + 'M'}", flush=True)

model = model.cuda()

if opt.load is not None:
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load, flush=True)

global best_accuracy, best_epoch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
correct_sample = 0
total_sample = 0
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

image = Image.open(opt.data_root).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)
total_number = 0
with torch.no_grad():
    image = image.to(device, dtype=torch.float32)
    res, _, _ = model(image)
    preds = res.detach().max(dim=1)[1].cpu().numpy()
    result_list = []
    for i in np.unique(preds):
        if i == 0:
            result_list.append(0)
        else:
            result_list.append(np.sum(preds==i))
    for value in result_list:
        total_number += value
    for i in range(10):
        if i == 0:
            pass
        elif i < len(result_list):
            percentage = result_list[i]/total_number
            print('{}料型占比:{:.2f}'.format(i, percentage))
        else:
            print('{}料型占比:{:.2f}'.format(i, 0))
