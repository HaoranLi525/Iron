import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from lib.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from lib.aspp import build_aspp
from lib.decoder import build_decoder
from lib.backbone import build_backbone

class Deeplab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=True, freeze_bn=False):
        super(Deeplab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
    
    def forward(self, input=None):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        feature = x
        x1, x2, feature_last = self.decoder(x, low_level_feat)
        x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x1, x2, feature_last