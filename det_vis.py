# /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 17:05
# @Author  : wkm
# @QQmail  : 690772123@qq.com

from mmocr.utils.ocr import MMOCR

# 导入模型到内存
# ocr = MMOCR(det='PS_CTW', det_config='result/ctw1500/psenet_r50_bifpnf_resasapp_coordconv_after_600e_ctw1500_8sampler_2/psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500_8sampler.py', recog=None, det_ckpt='result/ctw1500/psenet_r50_bifpnf_resasapp_coordconv_after_600e_ctw1500_8sampler_2/epoch_290.pth',
#  device='cuda:3')

ocr = MMOCR(det='PS_CTW', det_config='configs/textdet/psenet/ctw1500/gene_psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500_8sampler.py', recog=None, det_ckpt='attack_result/ctw1500/attack_psenet_r50_bifpnf_resasapp_coordconv_600e_ctw1500_8sampler/epoch_50.pth',
 device='cuda:3')

# 推理
# results = ocr.readtext('/home/datasets/textGroup/ctw1500/imgs/attac_testing/', output='demo/attack/demo', export='demo/attack/demo/result_json')
# results = ocr.readtext('/home/datasets/textGroup/ctw1500/imgs/PSENetattac_training/', output='demo/PSE_attack_640/demo', export='demo/PSE_attack_640/demo/result_json')
results = ocr.readtext('/home/datasets/textGroup/ctw1500/imgs/PSENetattac_training_0.2var/', output='demo/PSENetattac_training_0.2var/', export='demo/PSENetattac_training_0.2var/result_json')