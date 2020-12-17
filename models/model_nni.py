# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
from addict import Dict
from torch import nn
import torch.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head

from models.backbone.resnet import BasicBlock
from models.basic import ConvBnRelu
import math
import torch


class ModelNNI(nn.Module):
    def __init__(self, model_config: dict, layers=[2, 2, 2, 2], in_channels=3, inner_channels=256, k=50):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        self.name = f'resnet18_fpn_db'

        self.inplanes = 64
        self.backon_out_channels = []
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(self.backon_out_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(self.backon_out_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(self.backon_out_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(self.backon_out_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out

        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels // 4, 3, padding=1),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels // 4, self.out_channels // 4, 2, 2),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.out_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(self.out_channels)
        self.thresh.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self.backon_out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x, thred=None, is_thred=False, is_binary=True):
        _, _, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # Top-down
        p5 = self.reduce_conv_c5(x5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(x4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(x3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(x2))
        p2 = self.smooth_p2(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)

        shrink_maps = self.binarize(x)
        if is_thred and thred:
            threshold_maps = thred.unsqueeze(1)
        else:
            threshold_maps = self.thresh(x)
        if self.training and is_binary:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)

        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


class Unsample_Add(nn.Module):
    def __init__(self, ):
        super(Unsample_Add, self).__init__()

    def forward(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'resnest50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = ModelNNI(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model.name)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
