# Copyright (c) OpenMMLab. All rights reserved.
import cv2.cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16, Sequential

from mmocr.models.builder import NECKS

def swish(x):
    return x * x.sigmoid()

@NECKS.register_module()
class FPNF(BaseModule):
    """FPN-like fusion module in Shape Robust Text Detection with Progressive
    Scale Expansion Network.

    Args:
        in_channels (list[int]): A list of number of input channels.
        out_channels (int): The number of output channels.
        fusion_type (str): Type of the final feature fusion layer. Available
            options are "concat" and "add".
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 fusion_type='concat',
                 use_asf=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.backbone_end_level = len(in_channels)
        for i in range(self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

            if i < self.backbone_end_level - 1:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)

        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = 1024
        elif self.fusion_type == 'add':
            feature_channels = 256
        else:
            raise NotImplementedError

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        # self.weight_feature = nn.Parameter(
        #     torch.ones(self.backbone_end_level, dtype=torch.float32,
        #                requires_grad=True)
        # )

        self.use_asf = use_asf
        if self.use_asf:
            self.ASF = ASF(feature_channels, asf_type=self.use_asf)

    @auto_fp16()
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C_i, H_i, W_i)`. It usually expects 4 tensors
                (C2-C5 features) from ResNet.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            # step 2: smooth level i-1
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # upsample and cont
        bottom_shape = laterals[0].shape[2:]
        for i in range(1, used_backbone_levels):
            laterals[i] = F.interpolate(
                laterals[i], size=bottom_shape, mode='nearest')

        # weights = F.relu(self.weight_feature)
        # norm_weights = weights / (weights.sum() + 0.0001)
        #
        # #这种直接覆盖回原来的变量，会有问题吗？
        # for i, lateral in enumerate(laterals):
        #     norm_weight = norm_weights[i]
        #     lateral = lateral * norm_weight
        #     lateral = swish(lateral)
        #     laterals[i] = lateral

        if self.fusion_type == 'concat':
            out = torch.cat(laterals, 1)
        elif self.fusion_type == 'add':
            out = laterals[0]
            for i in range(1, used_backbone_levels):
                out += laterals[i]
        else:
            raise NotImplementedError

        if self.use_asf:
            out = self.ASF(out)

        out = self.output_convs(out)

        return out

@NECKS.register_module()
class BiFPN(BaseModule):
    def __init__(
        self,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat',
        use_asf=False,
        num_repeated=1,
        use_CBAM=False,
        init_cfg=dict(
            type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)

        self.repeated_bifpn = nn.ModuleList()
        self.in_channels = in_channels
        print(num_repeated)
        for i in range(num_repeated):
            if i != 0:
                in_channels = [out_channels] * len(in_channels)

            self.repeated_bifpn.append(BiFPN_PARTS(in_channels,out_channels))

        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = out_channels * 4
        elif self.fusion_type == 'add':
            feature_channels = out_channels
        else:
            raise NotImplementedError


        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.fpn_convs = ModuleList()
        self.backbone_end_level = len(in_channels)
        for i in range(self.backbone_end_level - 1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        self.use_asf = use_asf
        if self.use_asf:
            self.ASF = ASF(feature_channels, asf_type=self.use_asf)

        self.use_CBAM = use_CBAM
        if self.use_CBAM:
            self.CBAM_One = CBAM(inchannels=256)
            self.CBAM_Two = CBAM(inchannels=512)

    def forward(self, features):
        for bifpn in self.repeated_bifpn:
            features = bifpn(features)

        output_feats = features
        # build top-down path
        used_backbone_levels = len(output_feats)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            prev_shape = output_feats[i - 1].shape[2:]
            # output_feats[i - 1] += F.interpolate(
            #     output_feats[i], size=prev_shape, mode='nearest')

            output_feats[i - 1] = output_feats[i - 1] + F.interpolate(
                output_feats[i], size=prev_shape, mode='nearest')
            # step 2: smooth level i-1
            output_feats[i - 1] = self.fpn_convs[i - 1](output_feats[i - 1])

        # upsample and cont
        bottom_shape = output_feats[0].shape[2:]
        for i in range(1, used_backbone_levels):
            output_feats[i] = F.interpolate(
                output_feats[i], size=bottom_shape, mode='nearest')
        if self.fusion_type == 'concat':
            out = torch.cat(output_feats, 1)
        elif self.fusion_type == 'add':
            out = output_feats[0]
            for i in range(1, used_backbone_levels):
                out = out + output_feats[i]

        if self.use_asf:
            out = self.ASF(out)

        out = self.output_convs(out)
        return out


@NECKS.register_module()
class SingleBiFPN(BaseModule):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            fusion_type='concat',
            use_asf=False,
            use_CBAM=False,
            use_SE=False,
            use_CBAM_backward=False,
            mode='bilinear',
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform')):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super().__init__(init_cfg=init_cfg)
        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.mode = mode
        self.out_channels = out_channels
        self.backbone_end_level = len(in_channels)
        # build 5-levels bifpn
        if len(in_channels) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels) == 6:
            self.nodes = [
                {'feat_level': 4, 'inputs_offsets': [4, 5]},
                {'feat_level': 3, 'inputs_offsets': [3, 6]},
                {'feat_level': 2, 'inputs_offsets': [2, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 8]},
                {'feat_level': 0, 'inputs_offsets': [0, 9]},
                {'feat_level': 1, 'inputs_offsets': [1, 9, 10]},
                {'feat_level': 2, 'inputs_offsets': [2, 8, 11]},
                {'feat_level': 3, 'inputs_offsets': [3, 7, 12]},
                {'feat_level': 4, 'inputs_offsets': [4, 6, 13]},
                {'feat_level': 5, 'inputs_offsets': [5, 14]},
            ]
        elif len(in_channels) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        elif len(in_channels) == 4:
            self.nodes = [  #这里的结构跟bifpn是不一样的，bifpn输入输出是一样的，这里少了一层，比如输入6层，会输出5层的！
                {'feat_level': 2, 'inputs_offsets': [2, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 4]},
                {'feat_level': 0, 'inputs_offsets': [0, 5]},
                {'feat_level': 0, 'inputs_offsets': [5, 6]},
                {'feat_level': 1, 'inputs_offsets': [1, 5, 7]},
                {'feat_level': 2, 'inputs_offsets': [2, 4, 8]},
                {'feat_level': 3, 'inputs_offsets': [9, 3]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels]

        num_output_connections = [0 for _ in in_channels]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    # lateral_conv = Conv2d(
                    #     in_channels,
                    #     out_channels,
                    #     kernel_size=1,
                    #     norm=get_norm(norm, out_channels)
                    # )

                    lateral_conv = ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)

                    self.add_module(  #命名是 不同的input_node+不同的层
                        "lateral_{}_feat_level{}".format(input_offset, feat_level), lateral_conv

                    )
            node_info.append(out_channels)
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            # self.add_module(name, Conv2d(
            #     out_channels,
            #     out_channels,
            #     kernel_size=3,
            #     padding=1,
            #     norm=get_norm(norm, out_channels),
            #     bias=(norm == "")
            # ))
            self.add_module(name, ConvModule(  #每一个新生成的节点进行一次卷积操作
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            )


        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = out_channels * 4
        elif self.fusion_type == 'add':
            feature_channels = out_channels
        else:
            raise NotImplementedError

        self.fpn_convs = ModuleList()
        for i in range(self.backbone_end_level - 1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        self.use_asf = use_asf
        if self.use_asf:
            self.ASF = ASF(feature_channels, asf_type=self.use_asf)

        # elif self.asf_type == 'max_mean_asf':
        #     self.ASF = MAX_MEAN_ASF(feature_channels)
        # self.down_conv = ConvModule(
        #     out_channels,
        #     out_channels,
        #     3,
        #     padding =1,
        #     stride =2,
        #     conv_cfg=None,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg,
        #     inplace=False)
        self.use_CBAM = use_CBAM
        if self.use_CBAM:
            self.CBAM_One = CBAM(inchannels=256)
            self.CBAM_Two = CBAM(inchannels=512)

        self.use_CBAM_backward = use_CBAM_backward
        if self.use_CBAM_backward:
            self.CBAM_backward_One = CBAM(inchannels=256)
            self.CBAM_backward_TWO = CBAM(inchannels=512)

        self.use_SE = use_SE
        if self.use_SE:
            self.SE_Block = SE_Block(256)


    @auto_fp16()
    def forward(self, feats):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """


        feats = [_ for _ in feats]

        if self.use_CBAM:
            # print("feats[0].shape: ",feats[0].shape)
            feats[0] = self.CBAM_One(feats[0])
            feats[1] = self.CBAM_Two(feats[1])

        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_feat_level{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                    # input_node = self.down_conv(input_node)

                #向上的分支
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            # mode="nearest"
                            mode=self.mode
                            # mode="bilinear"
                        )
                elif (h == 1 and target_h == 1) and w > target_w:
                    height_stride_size = 1
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == 1, width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 2, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                    # input_node = self.down_conv(input_node)
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()

        # return output_feats
        # build top-down path
        #
        # print("________________________")
        # print(output_feats.shape)

        # if self.use_CBAM_backward:
        #     output_feats[0] = self.CBAM_backward_One(output_feats[0])
        #     output_feats[1] = self.CBAM_backward_One(output_feats[1])

        used_backbone_levels = len(output_feats)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            prev_shape = output_feats[i - 1].shape[2:]
            # output_feats[i - 1] += F.interpolate(
            #     output_feats[i], size=prev_shape, mode='nearest')

            output_feats[i - 1] = output_feats[i - 1] + F.interpolate(
                output_feats[i], size=prev_shape, mode=self.mode)
            # step 2: smooth level i-1
            output_feats[i - 1] = self.fpn_convs[i - 1](output_feats[i - 1])

        # upsample and cont
        bottom_shape = output_feats[0].shape[2:]
        for i in range(1, used_backbone_levels):
            output_feats[i] = F.interpolate(
                output_feats[i], size=bottom_shape, mode=self.mode)

        if self.fusion_type == 'concat':
            out = torch.cat(output_feats, 1)
        elif self.fusion_type == 'add':
            out = output_feats[0]
            for i in range(1, used_backbone_levels):
                out = out + output_feats[i]

        if self.use_asf:
            out = self.ASF(out)

        out = self.output_convs(out)

        if self.use_SE:
            out = self.SE_Block(out)

        return out

class ASF(BaseModule):
    def __init__(self,
                 in_channels=1024,
                 asf_type='',
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)

        # Saptial Attention branch
        self.max_spatial_attention = MAX_Spatial_Attention(in_channels)
        if asf_type == 'min':
            self.second_spatial_attention = MIN_Spatial_Attention(in_channels)
        elif asf_type == 'mean':
            self.second_spatial_attention = MEAN_Spatial_Attention(in_channels)
        else:
            raise raiseNotImplementedError("attetntion type must use mean or min ")

    def forward(self, inputs):
        #MAX_Saptial Attention
        max_sp_atten_weights = self.max_spatial_attention(inputs)    # [batch, 1, H, W]

        inputs = max_sp_atten_weights * inputs                                     # [batch, 1, H, W] * [batch, 1024, H, W]

        second_sp_atten_weights = self.second_spatial_attention(inputs)    # [batch, 1, H, W]

        out = second_sp_atten_weights * inputs

        return out  #[batch, 1024, H, W]

class MAX_Spatial_Attention(BaseModule):
    def __init__(self,
                 in_channels=1024,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.max_sp_conv = ConvModule(in_channels,in_channels,3,padding=1,stride=1,inplace=False)

        self.conv_relu_conv_sigmoid = nn.Sequential(
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,act_cfg=act_cfg,),
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, inplace=False,),
            nn.Sigmoid(),
        )

        self.conv_sigmoid = nn.Sequential(
            # nn.Conv2d(input_channels, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=in_channels,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        #conv
        max_sp_atten_feat  = self.max_sp_conv(inputs)                                        # [batch, 1024, H, W]
        # 维度衰减
        max_sp_atten_feat = torch.max(max_sp_atten_feat, dim=1, keepdim=True).values         # [batch, 1024, H, W] --> [batch, 1, H, W]

        #生成权重
        atten_weighted = self.conv_relu_conv_sigmoid(max_sp_atten_feat)      # [batch, 1, H, W]
        atten_weighted = atten_weighted + inputs                             # [batch, 1, H, W] + [batch, 1024, H, W]
        atten_weighted = self.conv_sigmoid(atten_weighted)                   # [batch, 1024, H, W] -> [batch, 1, H, W]

        return atten_weighted

class MIN_Spatial_Attention(BaseModule):
    def __init__(self,
                 in_channels=1024,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')
        self.min_sp_conv = ConvModule(in_channels, in_channels, 3, padding=1, stride=1, inplace=False)
        self.conv_relu_conv_sigmoid = nn.Sequential(
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,act_cfg=act_cfg,),
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, inplace=False,),
            nn.Sigmoid(),
        )

        self.conv_sigmoid = nn.Sequential(
            # nn.Conv2d(input_channels, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=in_channels,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        ##conv
        min_sp_atten_feat  = self.min_sp_conv(inputs)

        # 维度衰减
        min_sp_atten_feat = -torch.max(-min_sp_atten_feat, dim=1, keepdim=True).values         # [batch, 1024, H, W] --> [batch, 1, H, W]

        #生成权重
        atten_weighted = self.conv_relu_conv_sigmoid(min_sp_atten_feat)      # [batch, 1, H, W]
        atten_weighted = inputs - atten_weighted                             # [batch, 1, H, W] + [batch, 1024, H, W]
        atten_weighted = self.conv_sigmoid(atten_weighted)                   # [batch, 1024, H, W] -> [batch, 1, H, W]

        return atten_weighted


class MEAN_Spatial_Attention(BaseModule):
    def __init__(self,
                 in_channels=1024,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')
        self.mean_sp_conv = ConvModule(in_channels, in_channels, 3, padding=1, stride=1, inplace=False)
        self.conv_relu_conv_sigmoid = nn.Sequential(
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,act_cfg=act_cfg,),
            # nn.Conv2d(1, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1, inplace=False,),
            nn.Sigmoid(),
        )

        self.conv_sigmoid = nn.Sequential(
            # nn.Conv2d(input_channels, 1, 3, padding=1, stride=1),
            ConvModule(in_channels=in_channels,out_channels=1,kernel_size=3,padding=1,stride=1,inplace=False,),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        ##conv
        mean_sp_atten_feat  = self.mean_sp_conv(inputs)

        # 维度衰减
        mean_sp_atten_feat = torch.mean(mean_sp_atten_feat, dim=1, keepdim=True)         # [batch, 1024, H, W] --> [batch, 1, H, W]

        #生成权重
        atten_weighted = self.conv_relu_conv_sigmoid(mean_sp_atten_feat)      # [batch, 1, H, W]
        atten_weighted = inputs - atten_weighted                             # [batch, 1, H, W] + [batch, 1024, H, W]
        atten_weighted = self.conv_sigmoid(atten_weighted)                   # [batch, 1024, H, W] -> [batch, 1, H, W]

        return atten_weighted



@NECKS.register_module()
class BiFPN_PARTS(BaseModule):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(
            self,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            fusion_type='concat',
            use_asf=False,
            mode='nearest',
            init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform')):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
        """
        super().__init__(init_cfg=init_cfg)
        conv_cfg = None
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        self.mode = mode
        self.out_channels = out_channels
        self.backbone_end_level = len(in_channels)
        # build 5-levels bifpn
        if len(in_channels) == 5:
            self.nodes = [
                {'feat_level': 3, 'inputs_offsets': [3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
                {'feat_level': 1, 'inputs_offsets': [1, 6]},
                {'feat_level': 0, 'inputs_offsets': [0, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 7, 8]},
                {'feat_level': 2, 'inputs_offsets': [2, 6, 9]},
                {'feat_level': 3, 'inputs_offsets': [3, 5, 10]},
                {'feat_level': 4, 'inputs_offsets': [4, 11]},
            ]
        elif len(in_channels) == 6:
            self.nodes = [
                {'feat_level': 4, 'inputs_offsets': [4, 5]},
                {'feat_level': 3, 'inputs_offsets': [3, 6]},
                {'feat_level': 2, 'inputs_offsets': [2, 7]},
                {'feat_level': 1, 'inputs_offsets': [1, 8]},
                {'feat_level': 0, 'inputs_offsets': [0, 9]},
                {'feat_level': 1, 'inputs_offsets': [1, 9, 10]},
                {'feat_level': 2, 'inputs_offsets': [2, 8, 11]},
                {'feat_level': 3, 'inputs_offsets': [3, 7, 12]},
                {'feat_level': 4, 'inputs_offsets': [4, 6, 13]},
                {'feat_level': 5, 'inputs_offsets': [5, 14]},
            ]
        elif len(in_channels) == 3:
            self.nodes = [
                {'feat_level': 1, 'inputs_offsets': [1, 2]},
                {'feat_level': 0, 'inputs_offsets': [0, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 3, 4]},
                {'feat_level': 2, 'inputs_offsets': [2, 5]},
            ]
        elif len(in_channels) == 4:
            self.nodes = [  #这里的结构跟bifpn是不一样的，bifpn输入输出是一样的，这里少了一层，比如输入6层，会输出5层的！
                {'feat_level': 2, 'inputs_offsets': [2, 3]},
                {'feat_level': 1, 'inputs_offsets': [1, 4]},
                {'feat_level': 0, 'inputs_offsets': [0, 5]},
                {'feat_level': 0, 'inputs_offsets': [5, 6]},
                {'feat_level': 1, 'inputs_offsets': [1, 5, 7]},
                {'feat_level': 2, 'inputs_offsets': [2, 4, 8]},
                {'feat_level': 3, 'inputs_offsets': [9, 3]},
            ]
        else:
            raise NotImplementedError

        node_info = [_ for _ in in_channels]

        num_output_connections = [0 for _ in in_channels]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1

                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    # lateral_conv = Conv2d(
                    #     in_channels,
                    #     out_channels,
                    #     kernel_size=1,
                    #     norm=get_norm(norm, out_channels)
                    # )

                    lateral_conv = ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)

                    self.add_module(  #命名是 不同的input_node+不同的层
                        "lateral_{}_feat_level{}".format(input_offset, feat_level), lateral_conv

                    )
            node_info.append(out_channels)
            num_output_connections.append(0)

            # generate attention weights
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            self.__setattr__(name, nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True
                ))

            # generate convolutions after combination
            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            # self.add_module(name, Conv2d(
            #     out_channels,
            #     out_channels,
            #     kernel_size=3,
            #     padding=1,
            #     norm=get_norm(norm, out_channels),
            #     bias=(norm == "")
            # ))
            self.add_module(name, ConvModule(  #每一个新生成的节点进行一次卷积操作
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            )


        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            feature_channels = out_channels * 4
        elif self.fusion_type == 'add':
            feature_channels = out_channels
        else:
            raise NotImplementedError

        self.fpn_convs = ModuleList()
        for i in range(self.backbone_end_level - 1):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

        self.use_asf = use_asf
        if self.use_asf:
            self.ASF = ASF(feature_channels, asf_type=self.use_asf)

        # elif self.asf_type == 'max_mean_asf':
        #     self.ASF = MAX_MEAN_ASF(feature_channels)
        # self.down_conv = ConvModule(
        #     out_channels,
        #     out_channels,
        #     3,
        #     padding =1,
        #     stride =2,
        #     conv_cfg=None,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg,
        #     inplace=False)




    @auto_fp16()
    def forward(self, feats):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "p5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["n2", "n3", ..., "n6"].
        """


        feats = [_ for _ in feats]
        num_levels = len(feats)
        num_output_connections = [0 for _ in feats]
        for fnode in self.nodes:
            feat_level = fnode["feat_level"]
            inputs_offsets = fnode["inputs_offsets"]
            inputs_offsets_str = "_".join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                num_output_connections[input_offset] += 1
                input_node = feats[input_offset]

                # reduction
                if input_node.size(1) != self.out_channels:
                    name = "lateral_{}_feat_level{}".format(input_offset, feat_level)
                    input_node = self.__getattr__(name)(input_node)

                # maybe downsample
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 1, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                    # input_node = self.down_conv(input_node)

                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode="nearest"
                            # mode="bilinear"
                        )
                elif (h == 1 and target_h == 1) and w > target_w:
                    height_stride_size = 1
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == 1, width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=(height_stride_size + 2, width_stride_size + 1),
                        stride=(height_stride_size, width_stride_size), padding=1
                    )
                    # input_node = self.down_conv(input_node)
                else:
                    raise NotImplementedError()
                input_nodes.append(input_node)

            # attention
            name = "weights_f{}_{}".format(feat_level, inputs_offsets_str)
            weights = F.relu(self.__getattr__(name))
            norm_weights = weights / (weights.sum() + 0.0001)

            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = swish(new_node)

            name = "outputs_f{}_{}".format(feat_level, inputs_offsets_str)
            feats.append(self.__getattr__(name)(new_node))

            num_output_connections.append(0)

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()

        return output_feats
        # build top-down path
        # used_backbone_levels = len(output_feats)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # step 1: upsample to level i-1 size and add level i-1
        #     prev_shape = output_feats[i - 1].shape[2:]
        #     # output_feats[i - 1] += F.interpolate(
        #     #     output_feats[i], size=prev_shape, mode='nearest')
        #
        #     output_feats[i - 1] = output_feats[i - 1] + F.interpolate(
        #         output_feats[i], size=prev_shape, mode=self.mode)
        #     # step 2: smooth level i-1
        #     output_feats[i - 1] = self.fpn_convs[i - 1](output_feats[i - 1])
        #
        # # upsample and cont
        # bottom_shape = output_feats[0].shape[2:]
        # for i in range(1, used_backbone_levels):
        #     output_feats[i] = F.interpolate(
        #         output_feats[i], size=bottom_shape, mode=self.mode)
        # if self.fusion_type == 'concat':
        #     out = torch.cat(output_feats, 1)
        # elif self.fusion_type == 'add':
        #     out = output_feats[0]
        #     for i in range(1, used_backbone_levels):
        #         out = out + output_feats[i]
        #
        # if self.use_asf:
        #     out = self.ASF(out)
        #
        # out = self.output_convs(out)
        # return out


class Channel_Attention(BaseModule):
    def __init__(self, inchannels=256, ratio=16,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = Sequential(
            # nn.Conv2d(inchannels, inchannels // ratio, 1, bias=False),
            ConvModule(inchannels, inchannels // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            # nn.Conv2d(inchannels // ratio, inchannels, 1, bias=False)
            ConvModule(inchannels // ratio, inchannels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  #(B, C, 1, 1)
        max_out = self.fc(self.max_pool(x))  #(B, C, 1, 1)
        out = avg_out + max_out  #(B, C, 1, 1)
        return self.sigmoid(out) #(B, C, 1, 1)

class Spatial_Channel(BaseModule):
    def __init__(self, kernel_size=7, init_cfg=dict(
                type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        # self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2,bias=False) #这里梯度就不更新了。
        self.conv1 = ConvModule(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = torch.max(x, dim=1, keepdim=True).values
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x) # (B, 1, H, W)
        return self.sigmoid(x) #(B, 1, H, W)

class CBAM(BaseModule):
    def __init__(self, inchannels=256, ratio=16,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        self.channel_attention = Channel_Attention(inchannels, ratio)
        self.spatial_channel = Spatial_Channel()

    def forward(self, x):
        channel_x = self.channel_attention(x)  #(B, C, 1, 1)
        x = torch.multiply(x, channel_x) #(B, C, H, W)
        spatial_x = self.spatial_channel(x) #(B, 1, H, W)
        x = torch.multiply(x, spatial_x) #(B, C, H, W)
        return x

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上