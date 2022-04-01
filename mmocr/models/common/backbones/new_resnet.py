# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 21:51
# @Author  : wkm
# @QQmail  : 690772123@qq.com

import timm
from mmocr.models.builder import (BACKBONES, UPSAMPLE_LAYERS,
                                  build_activation_layer, build_upsample_layer)
import torch.nn as nn
from mmcv.runner import BaseModule

@BACKBONES.register_module()
class gluon_resnet50_v1d(BaseModule, nn.Module):
    def __init__(self,
                 out_indices,
                 init_cfg,
                 pretrained=True,
                 features_only=False,
                 ):
        super().__init__(init_cfg=init_cfg)
        self.out_indices = out_indices
        m = timm.create_model('gluon_resnet50_v1d', features_only=True, pretrained=True)
        self.m = m

    def forward(self, x):
        feature = self.m(x)
        feature = [feature[indies] for indies in self.out_indices]
        return feature

