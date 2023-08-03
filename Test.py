import os
import torch
import torch.nn.functional as F
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--val_batchsize', type=int, default=4, help='evaluation batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data_loader')
parser.add_argument('--num_classes', type=int, default=10, help='number of iron types')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
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

val_data = dataset.SemData(split='eval', data_root=opt.data_root, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.val_batchsize, shuffle=False, num_workers=opt.num_workers, pin_memory=True, sampler=None)

global best_accuracy, best_epoch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = StreamSegMetrics(10)
metrics.reset()
model.eval()
correct_sample = 0
total_sample = 0
with torch.no_grad():
    for i, (image, gt) in enumerate(test_loader, start=1):
        image = image.to(device, dtype=torch.float32)
        gt = gt.to(device, dtype=torch.long)
        res, _, _ = model(image)
        preds = res.detach().max(dim=1)[1].cpu().numpy()
        targets = gt.cpu().numpy()
        metrics.update(targets, preds)

    score = metrics.get_results()
    mIou = score['Mean IoU']
    acc = score['Overall Acc']

print('mIou: {:.4f}, acc: {:.4f}.'.format(mIou, acc, ))