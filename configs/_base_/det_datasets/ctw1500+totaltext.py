dataset_type = 'IcdarDataset'
data_root_1 = 'data/ctw1500'

train_1 = dict(
    type=dataset_type,
    ann_file=f'{data_root_1}/instances_training.json',
    img_prefix=f'{data_root_1}/imgs',
    pipeline=None)

test_1 = dict(
    type=dataset_type,
    ann_file=f'{data_root_1}/instances_test.json',
    img_prefix=f'{data_root_1}/imgs',
    pipeline=None)

data_root_2 = 'data/totaltext'

train_2 = dict(
    type=dataset_type,
    ann_file=f'{data_root_2}/instances_training.json',
    img_prefix=f'{data_root_2}/imgs',
    pipeline=None)

test_2 = dict(
    type=dataset_type,
    ann_file=f'{data_root_2}/instances_test.json',
    img_prefix=f'{data_root_2}/imgs',
    pipeline=None)

train_list = [train_1, train_2]

test_list = [test_1]

