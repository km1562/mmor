_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/schedules/schedule_adam_step_600e.py',
    '../../../../_base_/det_models/psenet_r50_bifpnf.py',  #bifpn，以后不要这样了，直接在本地覆盖掉
    '../../../../_base_/det_datasets/attack_ctw1500.py',
    '../../../../_base_/det_pipelines/no_aug_psenet_pipeline.py'
]

########################## model ##########################
model_poly = dict(
    type='PSENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='SingleBiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        in_channels=[256],
        out_channels=7,
        use_coordconv=True,
        use_resasapp=True,
        loss=dict(type='PSELoss', use_log_cosh_dice_loss=False,),
        postprocessor=dict(type='PSEPostprocessor', text_repr_type='poly')),
    train_cfg=None,
    test_cfg=None)

model_quad = dict(
    type='PSENet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNF',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        in_channels=[256],
        out_channels=7,
        loss=dict(type='PSELoss'),
        postprocessor=dict(type='PSEPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)
########################## model ##########################

model = model_poly

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

######################## dataset ##########################
dataset_type = 'IcdarDataset'
data_root = 'data/ctw1500'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_training.json',
    img_prefix=f'{data_root}/imgs/PSENetattac_training_0.5/',
    pipeline=None)

test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train]

test_list = [train]

######################## dataset ##########################

#######################  pse pipelin  #######################
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    # dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    # dict(
    #     type='ScaleAspectJitter',
    #     img_scale=[(3000, 736)],
    #     ratio_range=(0.5, 3),
    #     aspect_ratio_range=(1, 1),
    #     multiscale_mode='value',
    #     long_size_bound=1280,
    #     short_size_bound=640,
    #     resize_type='long_short_bound',
    #     keep_ratio=False),
    # dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='PSENetTargets'),
    dict(type='RandomFlip', flip_ratio=0.0000001, direction='horizontal'), #不能关？
    # dict(type='RandomRotateTextDet'),
    # dict(
    #     type='RandomCropInstances',
    #     target_size=(640, 640),
    #     instance_key='gt_kernels'),
    # dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]

# for ctw1500
img_scale_test_ctw1500 = (1280, 1280)
test_pipeline_ctw1500 = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',  #不能关？
        img_scale=img_scale_test_ctw1500,
        flip=False,
        transforms=[
            # dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='Resize', img_scale=(1280, 1280), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#######################  pse pipelin  #######################

train_pipeline = train_pipeline
test_pipeline_ctw1500 = test_pipeline_ctw1500

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    shuffle=False,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_ctw1500))

evaluation = dict(interval=10, metric='hmean-iou')
# runner = dict('Attack_Runner')
#TODO
# optimizer_config = dict(grad_clip=None, type='Attack_Optimize