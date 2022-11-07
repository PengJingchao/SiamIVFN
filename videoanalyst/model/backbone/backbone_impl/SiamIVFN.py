# -*- coding: utf-8 -*
from loguru import logger

import torch
import torch.nn as nn

from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.common_opr.common_block import conv_bn_relu
from videoanalyst.model.module_base import ModuleBase


@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class SiamIVFN(ModuleBase):
    r"""
    SiamIVFN, input is 4 channel.

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self):
        super(SiamIVFN, self).__init__()
        # CFFN
        # RGB
        self.viconv1 = conv_bn_relu(3, 96, stride=2, kszie=11, pad=0)
        self.vipool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.viconv2 = conv_bn_relu(96, 192, 1, 5, 0)
        self.vipool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.viconv3 = conv_bn_relu(256, 192, 1, 3, 0)
        self.viconv4 = conv_bn_relu(384, 96, 1, 3, 0)
        # Share
        self.shareconv2 = conv_bn_relu(96, 64, 1, 5, 0)
        self.sharepool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.shareconv3 = conv_bn_relu(256, 192, 1, 3, 0)
        self.shareconv4 = conv_bn_relu(384, 288, 1, 3, 0)
        self.shareconv5 = conv_bn_relu(384, 256, 1, 3, 0, has_relu=False)
        # T
        self.irconv1 = conv_bn_relu(1, 96, stride=2, kszie=11, pad=0)
        self.irpool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.irconv2 = conv_bn_relu(96, 192, 1, 5, 0)
        self.irpool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.irconv3 = conv_bn_relu(256, 192, 1, 3, 0)
        self.irconv4 = conv_bn_relu(384, 96, 1, 3, 0)
        # CAN
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512 // 16, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512 // 16, 512, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CFFN
        x_t = x[:, 3:4, :, :]
        x = x[:, 0:3, :, :]

        x = self.viconv1(x)
        x_t = self.irconv1(x_t)
        x = self.vipool1(x)
        x_t = self.irpool1(x_t)

        x_share = self.shareconv2(x)
        x_share = self.sharepool2(x_share)
        x = self.viconv2(x)
        x = self.vipool2(x)
        x_t_share = self.shareconv2(x_t)
        x_t_share = self.sharepool2(x_t_share)
        x_t = self.irconv2(x_t)
        x_t = self.irpool2(x_t)
        x = torch.cat((x, x_share), 1)
        x_t = torch.cat((x_t, x_t_share), 1)

        x_share = self.shareconv3(x)
        x = self.viconv3(x)
        x_t_share = self.shareconv3(x_t)
        x_t = self.irconv3(x_t)
        x = torch.cat((x, x_share), 1)
        x_t = torch.cat((x_t, x_t_share), 1)

        x_share = self.shareconv4(x)
        x = self.viconv4(x)
        x_t_share = self.shareconv4(x_t)
        x_t = self.irconv4(x_t)
        x = torch.cat((x, x_share), 1)
        x_t = torch.cat((x_t, x_t_share), 1)

        x = self.shareconv5(x)
        x_t = self.shareconv5(x_t)

        # CAN
        x = torch.cat((x, x_t), 1)
        x_residual = x.clone()
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x += x_residual

        return x
