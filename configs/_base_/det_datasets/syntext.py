# _base_ = [
#     '../../_base_/det_pipelines/psenet_pipeline.py'
# ]
# train_pipeline = {{_base_.train_pipeline}}
# test_pipeline_ctw1500 = {{_base_.test_pipeline_ctw1500}}

dataset_type = 'IcdarDataset'
data_root = 'data/synthtext'

train = dict(
    type='TextDetDataset',
    img_prefix=f'{data_root}/imgs',
    ann_file=f'{data_root}/instances_training.lmdb',
    loader=dict(
        type='LmdbLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=None,
    test_mode=True)

test = dict(
        type='TextDetDataset',
    img_prefix=f'{data_root}/imgs',
    ann_file=f'{data_root}/instances_training.lmdb',
        loader=dict(
            type='LmdbLoader',
            repeat=1,
            parser=dict(
                type='LineJsonParser',
                keys=['file_name', 'height', 'width', 'annotations'])),
        pipeline=None,
        test_mode=True)

train_list = [train]

test_list = [test]
