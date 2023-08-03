import torch
from torch import nn
import torch.nn.functional as F
import lib.ResNet as models
import numpy as np

class BioNet(nn.Module):
    def __init__(self, layer=50, dropout=0.4, num_classes=1, pretrained=True, args=None):
        super(BioNet, self).__init__()
        assert layer in [50, 101, 152]
        self.args = args
        BatchNorm = nn.BatchNorm2d
        output_stride = 16
        classes = num_classes
        if layer == 50:
            resnet = models.resnet50(pretrained=pretrained, progress=True)
        elif layer == 101:
            resnet = models.resnet101(pretrained=pretrained, progress=True)
        else:
            resnet = models.resnet152(pretrained=pretrained, progress=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pred = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, classes),
        )
    
    def forward(self, x=None):
        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.reshape(x_4.size(0), -1)
        x_pred = self.pred(x_4)

        return x_pred