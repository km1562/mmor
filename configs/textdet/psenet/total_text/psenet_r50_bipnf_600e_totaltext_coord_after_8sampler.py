_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_adam_step_600e.py',
    '../../../_base_/det_models/psenet_r50_fpnf.py',
    '../../../_base_/det_datasets/totaltext.py',
    '../../../_base_/det_pipelines/psenet_pipeline.py'
]


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
        use_asf='',
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        in_channels=[256],
        out_channels=7,
        use_resasapp=False,
        use_coordconv=True,
        loss=dict(type='PSELoss'),
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
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
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


model = model_poly

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
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

