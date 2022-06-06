# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset

from mmocr import digit_version
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.utils import get_root_logger

from apex import amp

from mmcv.runner import Runner, RUNNERS


from mmcv.runer.hooks.optimizer import OptimizerHook
from mmcv.runner.hook import HOOKS

@HOOKS.register_module()
class Attack_OptimizerHook(Hook):
    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        # runner.optimizer.step()


@RUNNERS.register_module()
class Attack_Runner(Runner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        # self._max_iters = 30
        self.call_hook('before_train_epoch')
        time. sleep(2)  # Prevent possible deadlock during epoch transition

        self.iter = 30
        for i, data_batch in enumerate(self.data_loader):  #所有图片

            self._inner_iter = i
            attack_train(data_batch)
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def attck_train(self, data_batch):
        pertur = torch.zeros(img.shape)
        for j in len(self.iter):
            self.call_hook('before_train_iter')

            pertur.requires_grad = True
            perturbation = pertur.cuda()
            img = data_batch['img'].data[0]  # img的数据

            attact_img = img + pertur
            data_batch['img']._data = attact_img

            # 每次iteration更新
            self.run_iter(data_batch, train_mode=True, **kwargs)
            if runner.outputs['loss'] < 0.05 or j == self.iter - 1:
                img = data_batch['img'].data[0]
                data_list = data_batch['img_metas'].data[0]
                for data_meta in data_list:
                    filename = data_mata['filename']
                    filename.replace('training', 'attac_training')
                    import os
                    (dir,file) = os.path.split(filename)
                    if not os.path.exists(dir):
                        os.mkdirs(dir)
                    import cv2
                    cv2.imwrite(filename, img.astype(int))
                    break

            print("cost :" ,runner.outputs['loss'])
            self.call_hook('after_train_iter')
            pertur = pertur - alpha * grad.sign()
            pertur.detach()


    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    if self.epoch == 0:
                        self.perturbation = None

                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')





def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(
            seed=cfg.get('seed'),
            drop_last=False,
            dist=distributed,
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # find_unused_parameters = cfg.get('find_unused_parameters', True)
        print("find_unused_parameters: ", find_unused_parameters)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    print(fp16_cfg)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    #apex
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_samples_per_gpu = (cfg.data.get('val_dataloader', {})).get(
            'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
        if val_samples_per_gpu > 1:
            # Support batch_size > 1 in test for text recognition
            # by disable MultiRotateAugOCR since it is useless for most case
            cfg = disable_text_recog_aug_test(cfg)
            cfg = replace_image_to_tensor(cfg)

        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_loader_cfg = {
            **loader_cfg,
            **dict(shuffle=False, drop_last=False),
            **cfg.data.get('val_dataloader', {}),
            **dict(samples_per_gpu=val_samples_per_gpu)
        }

        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)

        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed. If the seed is None, it will be replaced by a
    random number, and then broadcasted to all processes.

    Args:
        seed (int, Optional): The seed.
        device (str): The device where the seed will be put on.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()