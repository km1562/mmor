_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/schedules/schedule_adam_step_600e.py',
    '../../../_base_/det_models/psenet_r50_bifpnf.py',  #bifpn，以后不要这样了，直接在本地覆盖掉
    '../../../_base_/det_datasets/ctw1500.py',
    '../../../_base_/det_pipelines/psenet_pipeline.py'
]

model_poly = dict(
    type='PSENet',
    # backbone=dict(
    #     type='mmdet.ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    #     norm_eval=True,
    #     style='caffe'),
    backbone=dict(
        type='timm_backbone',
        # model_name='efficientnet_b3',
        model_name='efficientnet_b6',
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3, 4),
        init_cfg=dict(),
    ),
    neck=dict(
        type='SingleBiFPN',
        # in_channels=[256, 512, 1024, 2048],
        # in_channels=[32, 56, 160, 448],
        in_channels=[40, 72, 200, 576],
        # out_channels=256,  #这里本来是256的，然后是256 * 4 = 1024，改成512->2048了
        out_channels=512,  #这里本来是256的，然后是256 * 4 = 1024，改成512->2048了
        fusion_type='concat'),
    bbox_head=dict(
        type='PSEHead',
        in_channels=[512],
        out_channels=7,
        use_coordconv=True,
        use_resasapp=False,
        use_resasapp_add_255=True,
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


model = model_poly

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

total_epochs = 800

data = dict(
    samples_per_gpu=2,
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

evaluation = dict(interval=5, metric='hmean-iou')
