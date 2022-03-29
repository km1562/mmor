# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 10:40
# @Author  : wkm
# @QQmail  : 690772123@qq.com

import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS
from mmocr.utils import check_argument
from .head_mixin import HeadMixin
from mmcv.cnn import ConvModule

class Channel_Attention(nn.Module):
  def __init__(self, inchannels=256, ratio=1):
    super(Channel_Attention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

    self.fc = nn.Sequential(
      nn.Conv2d(inchannels, inchannels // ratio, 1, bias=False),
      nn.ReLU(),
      nn.Conv2d(inchannels // ratio, inchannels, 1, bias=False)
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.fc(self.avg_pool(x))  # (B, C, 1, 1)
    max_out = self.fc(self.max_pool(x))  # (B, C, 1, 1)
    out = avg_out + max_out  # (B, C, 1, 1)
    return self.sigmoid(out)  # (B, C, 1, 1)


class Spatial_Channel(nn.Module):
  def __init__(self, kernel_size=7):
    super(Spatial_Channel, self).__init__()
    self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    max_out = torch.max(x, dim=1, keepdim=True).values
    avg_out = torch.mean(x, dim=1, keepdim=True)
    x = torch.cat([max_out, avg_out], dim=1)
    x = self.conv1(x)  # (B, 1, H, W)
    return self.sigmoid(x)  # (B, 1, H, W)


class CBAM(BaseModule, nn.Module):
  def __init__(self,
               inchannels=256,
               ratio=1,
               init_cfg=dict(
                 type='Xavier', layer='Conv2d', distribution='uniform')
               ):
    super().__init__(init_cfg=init_cfg)
    self.channel_attention = Channel_Attention(inchannels, ratio)
    self.spatial_channel = Spatial_Channel()

  def forward(self, x):
    channel_x = self.channel_attention(x)  # (B, C, 1, 1)
    x = torch.multiply(x, channel_x)  # (B, C, H, W)
    spatial_x = self.spatial_channel(x)  # (B, 1, H, W)
    x = torch.multiply(x, spatial_x)  # (B, C, H, W)
    return x