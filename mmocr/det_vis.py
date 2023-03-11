# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 17:05
# @Author  : wkm
# @QQmail  : 690772123@qq.com

from mmocr.utils.ocr import MMOCR

# 导入模型到内存
ocr = MMOCR(det='PS_CTW', det_config='result/psenet_r50_bifpnf_resasapp_coordconv_after_600e_ctw1500_8sampler_2/psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500_8sampler.py', recog=None, det_ckpt='result/psenet_r50_bifpnf_resasapp_coordconv_after_600e_ctw1500_8sampler_2/epoch_290.pth')

# 推理
results = ocr.readtext('/home/datasets/textGroup/ctw1500/imgs/test/*', output='demo/paper/demo', export='demo/paper/demo', batch_mode=True, )