# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule,ModuleList, auto_fp16

import timm
from mmocr.models.builder import HEADS
from mmocr.utils import check_argument
from .head_mixin import HeadMixin
from mmcv.cnn import ConvModule
from .CBAM import CBAM

from torch.nn import Softmax


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
                 use_contextblokc=False,
                 use_cbam=False,
                 use_non_local_after=False,
                 use_criss_cro_att_after=False,
                 use_resasapp_add_255=False,
                 use_Spa_Tran=False,
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

        self.use_Spa_Tran = use_Spa_Tran
        if self.use_Spa_Tran:
            self.Spa_Tran = SpatialTransformer()


        # self.out_conv = ConvModule(np.sum(np.array(in_channels)), out_channels, 3, stride=1, padding=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))

        self.use_resasapp = use_resasapp
        if self.use_resasapp:
            self.asapp = Init_ASPP_ADD(
                in_channels=np.sum(np.array(in_channels)),
                depth=out_channels,
            )
        else:
            self.out_conv = nn.Conv2d(
                in_channels=np.sum(np.array(in_channels)),
                out_channels=out_channels,
                kernel_size=1)

        self.use_resasapp_add_255 = use_resasapp_add_255
        if self.use_resasapp_add_255:
            self.asapp_add_255 = Init_ASPP_ADD(
                in_channels=np.sum(np.array(in_channels)),
                depth=np.sum(np.array(in_channels)),
            )

        self.use_coordconv = use_coordconv
        if self.use_coordconv:
            self.coord_conv = MaskHead(
                in_channels=out_channels,
                out_channels=out_channels,
            )

        self.use_contextblokc = use_contextblokc
        if self.use_contextblokc:
            self.contextblokc = ContextBlock(
                inplanes=out_channels,
                ratio=1,
                pooling_type='att',
            )

        self.use_CBAM = use_cbam
        if self.use_CBAM:
            self.CBAM = CBAM(
                inchannels=out_channels,
            )

        self.use_non_local_after = use_non_local_after
        if self.use_non_local_after:
            self.non_loca_after = NonLocal(channel=out_channels)

        self.use_criss_cro_att_after = use_criss_cro_att_after
        if self.use_criss_cro_att_after:
            self.criss_cro_att_after = CrissCrossAttention(in_dim=out_channels)



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

        if self.use_resasapp_add_255:
            outputs = self.asapp_add_255(outputs)

        if self.use_Spa_Tran:
            outputs= self.Spa_Tran(outputs)

        if self.use_resasapp:
            outputs = self.asapp(outputs)
        else:
            outputs = self.out_conv(inputs)

        if self.use_contextblokc:
            outputs = self.contextblokc(outputs)

        if self.use_CBAM:
            outputs = self.CBAM(outputs)

        if self.use_coordconv:
            outputs = self.coord_conv(outputs)

        if self.use_non_local_after:
            outputs = self.non_loca_after(outputs)

        if self.use_criss_cro_att_after:
            outputs = self.criss_cro_att_after(outputs)



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
        self.atrous_block1 = ConvModule(in_channels, depth, 1, stride=1, norm_cfg=dict(type='BN'),  act_cfg=dict(type='LeakyReLU'))
        # 不同空洞率的卷积
        # self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block6 = ConvModule(in_channels, depth, 3, stride=1, padding=6, dilation=6, norm_cfg=dict(type='BN'),  act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block12 = ConvModule(in_channels, depth, 3, stride=1, padding=12, dilation=12, norm_cfg=dict(type='BN'),  act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.atrous_block18 = ConvModule(in_channels, depth, 3, stride=1, padding=18, dilation=18, norm_cfg=dict(type='BN'),  act_cfg=dict(type='LeakyReLU'))
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

        self.conv_origin = ConvModule(in_channels, depth, 3, stride=1, padding=1, norm_cfg=dict(type='BN'),  act_cfg=dict(type='LeakyReLU'))

    @auto_fp16()
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

class Init_ASPP_ADD_GN(BaseModule, nn.Module):

    #改进的asapp，跟残差网络一样叠加了一个块
    def __init__(self, in_channels,
                 depth=256,
                 init_cfg=dict(
                 type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.conv = ConvModule(in_channels, depth, 1, stride=1, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))
        # self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = ConvModule(in_channels, depth, 1, stride=1, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))
        # 不同空洞率的卷积
        # self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block6 = ConvModule(in_channels, depth, 3, stride=1, padding=6, dilation=6, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block12 = ConvModule(in_channels, depth, 3, stride=1, padding=12, dilation=12, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))

        # self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.atrous_block18 = ConvModule(in_channels, depth, 3, stride=1, padding=18, dilation=18, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

        self.conv_origin = ConvModule(in_channels, depth, 3, stride=1, padding=1, norm_cfg=dict(type='GN',num_groups=32), act_cfg=dict(type='LeakyReLU'))

    def forward(self, x):
        # print("use asapp")
        oringin = self.conv_origin(x)
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        # image_features = F.upsample(image_features, size=size, mode='bilinear')
        image_features = F.interpolate(image_features, size=size, mode='bilinear')
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

    @auto_fp16()
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


class ContextBlock(BaseModule, nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,pooling_type='att',
                 fusion_types=('channel_add', ),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            # self.conv_mask = ConvModule(inplanes, 1, kernel_size=1, norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU'))
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # ConvModule(self.inplanes, self.planes, kernel_size=1,norm_cfg=dict(type='BN'), act_cfg=dict(type='LeakyReLU')),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        # pirnt("use contextblock")
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class NonLocal(BaseModule, nn.Module):
    def __init__(self, channel,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, )
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, )

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c//2, -1)
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c//2, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(BaseModule, nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                 ):
        super().__init__(init_cfg=init_cfg)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class VIT(nn.Module):
    def __init__(self,
                 init_cfg,
                 model_name,
                 pretrained,
                 ):
        super().__init__(init_cfg=init_cfg)
        m = timm.create_model(model_name, pretraine=True)

    def forward(self,x):
        feature = self.m(x)
        return feature

class SpatialTransformer(nn.Module):
    # def __init__(self, spatial_dims):
    def __init__(self,):
        super(SpatialTransformer, self).__init__()

        self.conv = nn.Conv2d(256,32, 3, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(4)
        self.fc1 = nn.Linear(32*4*4, 1024) # 可根据自己的网络参数具体设置
        self.fc2 = nn.Linear(1024, 6)


    def forward(self, x):
        _, in_ch, h, w = x.shape
        batch_images = x #保存一份原始数据

        x = self.conv(x)
        x = self.max_pool(x)

        x = x.view(-1, 32*4*4)
        # 利用FC结构学习到6个参数
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 3) # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), in_ch, h, w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points)
        return rois


