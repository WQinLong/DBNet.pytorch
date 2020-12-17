# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
from addict import Dict
from torch import nn
import torch
import torch.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.normalize = Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x, thred=None, is_thred=False, is_binary=True):
        _, _, H, W = x.size()
        x = self.normalize(x)
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out, thred, is_thred=is_thred, is_binary=is_binary)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


class Normalize(nn.Module):
    # https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/modules/normalize.py
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float().reshape(1, len(mean), 1, 1).contiguous())
        self.register_buffer("std", torch.tensor(std).float().reshape(1, len(std), 1, 1).reciprocal().contiguous())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input.type_as(self.mean) - self.mean) * self.std


class MySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize([0.221 * 255], [0.242 * 255])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image, target):
        image = self.normalize(image)
        output = self.backbone(image)

        if target is not None:
            loss = self.loss(output, target.long())
            return loss

        return output


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'resnest50', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},  # 分割头，FPN or FPEM_FFM
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = Model(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(time.time() - tic)
    print(y.shape)
    print(model.name)
    print(model)
    # torch.save(model.state_dict(), 'PAN.pth')
