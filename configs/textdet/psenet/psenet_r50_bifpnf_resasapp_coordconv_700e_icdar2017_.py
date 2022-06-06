_base_ = [
    '../../_base_/schedules/schedule_sgd_600e.py',
    '../../_base_/runtime_10e.py',
    '../../_base_/det_models/psenet_r50_fpnf.py',
    '../../_base_/det_datasets/icdar2017.py',
    '../../_base_/det_pipelines/psenet_pipeline.py'
]

# model = {{_base_.model_quad}}

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

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
        loss=dict(type='PSELoss'),
        postprocessor=dict(type='PSEPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)


model = model_quad

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_icdar2015 = {{_base_.test_pipeline_icdar2015}}

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline_icdar2015))

evaluation = dict(interval=10, metric='hmean-iou')

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[200, 400, 600])
# total_epochs = 600
total_epochs = 700
