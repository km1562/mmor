# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 21:31
# @Author  : wkm
# @QQmail  : 690772123@qq.com


from mmocr.models.builder import LOSSES
from . import PANLoss
from mmocr.utils import check_argument
import cv2

@LOSSES.register_module()
class PSELoss(PANLoss):
    r"""The class for implementing PSENet loss. This is partially adapted from
    https://github.com/whai362/PSENet.

    PSENet: `Shape Robust Text Detection with
    Progressive Scale Expansion Network <https://arxiv.org/abs/1806.02559>`_.

    Args:
        alpha (float): Text loss coefficient, and :math:`1-\alpha` is the
            kernel loss coefficient.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss. Available options are
            "mean" and "sum".
    """

    def __init__(self,
                 # use_log_cosh_dice_loss=False,
                 alpha=0.7,
                 ohem_ratio=3,
                 reduction='mean',
                 kernel_sample_type='adaptive'):
        super().__init__()
        assert reduction in ['mean', 'sum'
                             ], "reduction must be either of ['mean','sum']"
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.kernel_sample_type = kernel_sample_type
        self.use_log_cosh_dice_loss = use_log_cosh_dice_loss

    # def forward(self, score_maps, downsample_ratio, gt_kernels, gt_mask):
    def forward(self, boundary_maps, gt_kernel, gt_mask):
        """Compute PSENet loss.

        Args:
            score_maps (tensor): The output tensor with size of Nx6xHxW.
            downsample_ratio (float): The downsample ratio between score_maps
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text`` and ``loss_kernel``.
        """

        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        # assert isinstance(downsample_ratio, float)
        losses = []

        # pred_texts = score_maps[:, 0, :, :]
        # pred_kernels = score_maps[:, 1:, :, :]
        # feature_sz = score_maps.size()

        # print("__downsample_ratio__")
        # print(downsample_ratio)
        # gt_kernels = [item.rescale(downsample_ratio) for item in gt_kernels]
        # gt_kernels = self.bitmasks2tensor(gt_kernels, feature_sz[2:])
        # gt_kernels = [item.to(score_maps.device) for item in gt_kernels]

        # gt_mask = [item.rescale(downsample_ratio) for item in gt_mask]

        dprint_shape(gt_mask)
        gt_mask = self.bitmasks2tensor(gt_mask, feature_sz[2:])
        gt_mask = [item.to(score_maps.device) for item in gt_mask]

        # compute text loss
        # sampled_masks_text = self.ohem_batch(pred_texts.detach(),
        #                                      gt_kernels[0], gt_mask[0])

        #TODO gt_mask做一个拉普拉斯
        boundary_gt_mask = [cv2.Laplacian(item, -1) for item in gt_mask]

        #TODO 可视化看对不对

        #for遍历，做一个边界loss
        for idx in range()


        log_cosh_dice_loss
        if self.use_log_cosh_dice_loss:
            loss_texts = self.log_cosh_dice_loss(pred_texts, gt_kernels[0],
                                                sampled_masks_text)
        else:
            loss_texts = self.dice_loss_with_logits(pred_texts, gt_kernels[0],
                                                sampled_masks_text)



        losses.append(self.alpha * loss_texts)

        # compute kernel loss
        if self.kernel_sample_type == 'hard':
            sampled_masks_kernel = (gt_kernels[0] > 0.5).float() * (
                gt_mask[0].float())
        elif self.kernel_sample_type == 'adaptive':
            sampled_masks_kernel = (pred_texts > 0).float() * (
                gt_mask[0].float())
        else:
            raise NotImplementedError

        num_kernel = pred_kernels.shape[1]
        assert num_kernel == len(gt_kernels) - 1
        loss_list = []
        for idx in range(num_kernel):

            # log_cosh_dice_loss
            if self.use_log_cosh_dice_loss:
                loss_kernels = self.log_cosh_dice_loss(
                pred_kernels[:, idx, :, :], gt_kernels[1 + idx],
                sampled_masks_kernel)
            else:
                loss_kernels = self.dice_loss_with_logits(
                    pred_kernels[:, idx, :, :], gt_kernels[1 + idx],
                    sampled_masks_kernel)
            loss_list.append(loss_kernels)

        losses.append((1 - self.alpha) * sum(loss_list) / len(loss_list))

        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError
        results = dict(loss_text=losses[0], loss_kernel=losses[1])
        return results


@LOSSES.register_module()
class PSE_Attack_Loss(PANLoss):
    r"""The class for implementing PSENet loss. This is partially adapted from
    https://github.com/whai362/PSENet.

    PSENet: `Shape Robust Text Detection with
    Progressive Scale Expansion Network <https://arxiv.org/abs/1806.02559>`_.

    Args:
        alpha (float): Text loss coefficient, and :math:`1-\alpha` is the
            kernel loss coefficient.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss. Available options are
            "mean" and "sum".
    """

    def __init__(self,
                 use_log_cosh_dice_loss=False,
                 alpha=0.7,
                 ohem_ratio=3,
                 reduction='mean',
                 kernel_sample_type='adaptive'):
        super().__init__()
        assert reduction in ['mean', 'sum'
                             ], "reduction must be either of ['mean','sum']"
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.kernel_sample_type = kernel_sample_type
        self.use_log_cosh_dice_loss = use_log_cosh_dice_loss

    def forward(self, score_maps, downsample_ratio, gt_kernels, gt_mask):
        """Compute PSENet loss.

        Args:
            score_maps (tensor): The output tensor with size of Nx6xHxW.
            downsample_ratio (float): The downsample ratio between score_maps
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text`` and ``loss_kernel``.
        """

        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)
        losses = []

        pred_texts = score_maps[:, 0, :, :]
        pred_kernels = score_maps[:, 1:, :, :]
        feature_sz = score_maps.size()

        # print("__downsample_ratio__")
        # print(downsample_ratio)
        gt_kernels = [item.rescale(downsample_ratio) for item in gt_kernels]
        gt_kernels = self.bitmasks2tensor(gt_kernels, feature_sz[2:])
        gt_kernels = [item.to(score_maps.device) for item in gt_kernels]

        gt_mask = [item.rescale(downsample_ratio) for item in gt_mask]
        gt_mask = self.bitmasks2tensor(gt_mask, feature_sz[2:])
        gt_mask = [item.to(score_maps.device) for item in gt_mask]

        # compute text loss
        sampled_masks_text = self.ohem_batch(pred_texts.detach(),
                                             gt_kernels[0], gt_mask[0])

        #attack loss
        loss_texts = self.attack_loss(pred_texts, gt_kernels[0],
                                                sampled_masks_text, )


        losses.append(self.alpha * loss_texts)

        # compute kernel loss
        if self.kernel_sample_type == 'hard':
            sampled_masks_kernel = (gt_kernels[0] > 0.5).float() * (
                gt_mask[0].float())
        elif self.kernel_sample_type == 'adaptive':
            sampled_masks_kernel = (pred_texts > 0).float() * (
                gt_mask[0].float())
        else:
            raise NotImplementedError

        num_kernel = pred_kernels.shape[1]
        assert num_kernel == len(gt_kernels) - 1
        loss_list = []
        for idx in range(num_kernel):
            loss_kernels = self.attack_loss(
                pred_kernels[:, idx, :, :], gt_kernels[1 + idx],
                sampled_masks_kernel)
            loss_list.append(loss_kernels)

        losses.append((1 - self.alpha) * sum(loss_list) / len(loss_list))

        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError
        results = dict(loss_text=losses[0], loss_kernel=losses[1])
        return results

    def attack_loss(self, pred, target, mask, threshold=0.5, c=0.1):

        pred = torch.sigmoid(pred)
        hyparameter_pixel_diff = -c * (pred - threshold)

        loss = 1 / (1 + torch.exp(hyparameter_pixel_diff))
        return loss

def dprint_shape(args):
    print(str(args) + "'s information", args.shape)