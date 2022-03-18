# Copyright (c) OpenMMLab. All rights reserved.
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

@HEADS.register_module()
class PANHead(HeadMixin, BaseModule):
    """The class for PANet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        postprocessor (dict): Config of postprocessor for PANet.
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_ratio=0.25,
                 loss=dict(type='PANLoss'),
                 postprocessor=dict(
                     type='PANPostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 use_resasapp=False,
                 use_coordconv=False,
                 init_cfg=dict(
                     type='Normal',
                     mean=0,
                     std=0.01,
                     override=dict(name='out_conv')),
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

        assert check_argument.is_type_list(in_channels, int)
        assert isinstance(out_channels, int)

        assert 0 <= downsample_ratio <= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_ratio = downsample_ratio
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.out_conv = nn.Conv2d(
            in_channels=np.sum(np.array(in_channels)),
            out_channels=out_channels,
            kernel_size=1)

        self.use_resasapp = use_resasapp
        if self.use_resasapp:
            self.asapp = Init_ASPP_ADD(
                in_channels=np.sum(np.array(in_channels)),
                depth=out_channels,
            )

        self.use_coordconv = use_coordconv
        if self.use_coordconv:
            self.asapp = MaskHead(
                in_channels=np.sum(np.array(in_channels)),
                out_channels=out_channels,
            )


    def forward(self, inputs):
        r"""
        Args:
            inputs (list[Tensor] | Tensor): Each tensor has the shape of
                :math:`(N, C_i, W, H)`, where :math:`\sum_iC_i=C_{in}` and
                :math:`C_{in}` is ``input_channels``.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, W, H)` where
            :math:`C_{out}` is ``output_channels``.
        """
        if isinstance(inputs, tuple):
            outputs = torch.cat(inputs, dim=1)
        else:
            outputs = inputs

        if self.use_resasapp:
            outputs = self.asapp(outputs)
        else:
            outputs = self.out_conv(outputs)

        return outputs

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

#CoordConv
class MaskHead(BaseModule, nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        out_channels = out_channels
        in_channels = in_channels + 2

        convs = []
        convs.append(ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU')))
        for i in range(3):
            convs.append(ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU')))
        self.mask_convs = nn.Sequential(*convs)

    def forward(self, features):
        x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)  # W
        y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)  # H
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([features.shape[0], 1, -1, -1])
        x = x.expand([features.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_features = torch.cat([features, coord_feat], dim=1)
        mask_features = self.mask_convs(ins_features)
        return mask_features