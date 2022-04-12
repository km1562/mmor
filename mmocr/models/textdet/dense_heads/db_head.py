# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

from mmocr.models.builder import HEADS
from .head_mixin import HeadMixin
from mmcv.cnn import ConvModule
from .CBAM import CBAM

from torch.nn import Softmax
from torch.nn import functional as F

@HEADS.register_module()
class DBHead(HeadMixin, BaseModule):
    """The class for DBNet head.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): The number of input channels of the db head.
        with_bias (bool): Whether add bias in Conv2d layer.
        downsample_ratio (float): The downsample ratio of ground truths.
        loss (dict): Config of loss for dbnet.
        postprocessor (dict): Config of postprocessor for dbnet.
    """

    def __init__(
            self,
            in_channels,
            use_asapp=False,
            with_bias=False,
            downsample_ratio=1.0,
            loss=dict(type='DBLoss'),
            postprocessor=dict(type='DBPostprocessor', text_repr_type='quad'),
            init_cfg=[
                dict(type='Kaiming', layer='Conv'),
                dict(type='Constant', layer='BatchNorm', val=1., bias=1e-4)
            ],
            train_cfg=None,
            test_cfg=None,
            **kwargs):
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)
        BaseModule.__init__(self, init_cfg=init_cfg)
        HeadMixin.__init__(self, loss, postprocessor)

        assert isinstance(in_channels, int)

        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.downsample_ratio = downsample_ratio

        self.binarize = Sequential(
            nn.Conv2d(
                in_channels, in_channels // 4, 3, bias=with_bias, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())

        self.threshold = self._init_thr(in_channels)

        self.use_resasapp = use_asapp
        if self.use_resasapp:
            self.resasapp = Init_ASPP_ADD(in_channels=self.in_channels, depth=self.in_channels)

    def diff_binarize(self, prob_map, thr_map, k):
        return torch.reciprocal(1.0 + torch.exp(-k * (prob_map - thr_map)))

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape (batch_size, hidden_size, h, w).

        Returns:
            Tensor: A tensor of the same shape as input.
        """
        if self.use_resasapp:
            inputs = self.resasapp(inputs)

        prob_map = self.binarize(inputs)
        thr_map = self.threshold(inputs)
        binary_map = self.diff_binarize(prob_map, thr_map, k=50)
        outputs = torch.cat((prob_map, thr_map, binary_map), dim=1)
        return outputs

    def _init_thr(self, inner_channels, bias=False):
        in_channels = inner_channels
        seq = Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2), nn.Sigmoid())
        return seq

class Init_ASPP_ADD(BaseModule, nn.Module):

    #改进的asapp，跟残差网络一样叠加了一个块
    def __init__(self, in_channels,
                 depth=256,
                 init_cfg=dict(
                 type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv = ConvModule(in_channels, depth, 1, stride=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))
        # self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = ConvModule(in_channels, depth, 1, stride=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))
        # 不同空洞率的卷积
        # self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block6 = ConvModule(in_channels, depth, 3, stride=1, padding=6, dilation=6, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block12 = ConvModule(in_channels, depth, 3, stride=1, padding=12, dilation=12, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.atrous_block18 = ConvModule(in_channels, depth, 3, stride=1, padding=18, dilation=18, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

        self.conv_origin = ConvModule(in_channels, depth, 3, stride=1, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))

    def forward(self, x):
        # print("use asapp")
        oringin = self.conv_origin(x)
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        # 不同空洞率的卷积
        atrous_block1 = self.atrous_block1(x) + oringin
        atrous_block6 = self.atrous_block6(x) + oringin
        atrous_block12 = self.atrous_block12(x) + oringin
        atrous_block18 = self.atrous_block18(x) + oringin
        # 汇合所有尺度的特征
        x = torch.cat([image_features, atrous_block1, atrous_block6,atrous_block12, atrous_block18], dim=1)
        # 利用1X1卷积融合特征输出
        x = self.conv_1x1_output(x)
        return x
