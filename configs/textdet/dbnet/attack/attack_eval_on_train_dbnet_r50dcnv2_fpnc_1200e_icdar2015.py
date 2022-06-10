_base_ = [
    '../../../_base_/runtime_10e.py',
    '../../../_base_/schedules/schedule_sgd_1200e.py',
    '../../../_base_/det_models/dbnet_r50dcnv2_fpnc.py',
    '../../../_base_/det_datasets/icdar2015.py',
    '../../../_base_/det_pipelines/not_aug_dbnet_pipeline.py'
]
######################## dataset ##################################
dataset_type = 'IcdarDataset'
data_root = 'data/icdar2015'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs/DBNetattac_training_0.5',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs/',
    pipeline=None)

train_list = [train]
test_list = [train] #在train数据集上评估
######################## dataset ##################################

######################## pipeling ##################################
# for dbnet_r50dcnv2_fpnc
img_norm_cfg_r50dcnv2 = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline_r50dcnv2 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg_r50dcnv2),
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.00001],
              # dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
              # dict(cls='Affine', rotate=[-10, 10]), ]
              ['Resize', (640,640)]
        ]
    ),
    # dict(type='EastRandomCrop', target_size=(640, 640)),
    # dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    # dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]

test_pipeline_4068_1024 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            # dict(type='Resize', img_scale=(2944, 736), keep_ratio=True),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg_r50dcnv2),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
######################## pipeling ##################################
#
#
# train_list = {{_base_.train_list}}
# test_list = {{_base_.test_list}}

train_pipeline_r50dcnv2 = train_pipeline_r50dcnv2
test_pipeline_4068_1024 = test_pipeline_4068_1024

load_from = 'checkpoints/textdet/dbnet/res50dcnv2_synthtext.pth'

data = dict(
    samples_per_gpu=20,
    workers_per_gpu=20,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline_r50dcnv2),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_4068_1024))

evaluation = dict(interval=100, metric='hmean-iou')
