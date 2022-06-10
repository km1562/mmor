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

from mmcv.runner.hooks.optimizer import OptimizerHook, HOOKS
# from mmcv.runner.hooks import HOOKS
from mmcv.runner.utils import get_host_info
from mmcv.runner.epoch_based_runner import *
from mmcv.image.misc import tensor2imgs

VAR=0.5
std = np.array([58.395, 57.12, 57.375])
mean = np.array([123.675, 116.28, 103.53])

def pert_map(pert, eps):
    return (1/(1+torch.exp(pert)) -0.5) *2 * eps


# def tensor_to_image(t):
#     img = t.squeeze()
#     img = img.permute(1,2,0)  #这里需要换嘛
#
#     img = img * torch.tensor(std,device=img.device) + torch.tensor(mean,device=img.device)
#     img *= 255.0
#     img = img.detach().cpu().numpy()
#     return img


def tensor2imgs(tensor, mean=None, std=None, to_rgb=True):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1.
        mean (tuple[float], optional): Mean of images. If None,
            (0, 0, 0) will be used for tensor with 3-channel,
            while (0, ) for tensor with 1-channel. Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            (1, 1, 1) will be used for tensor with 3-channel,
            while (1, ) for tensor with 1-channel. Defaults to None.
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            For the tensor with 1 channel, it must be False. Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    channels = tensor.size(1)
    assert channels in [1, 3]
    if mean is None:
        mean = (0, ) * channels
    if std is None:
        std = (1, ) * channels
    assert (channels == len(mean) == len(std) == 3) or \
        (channels == len(mean) == len(std) == 1 and not to_rgb)

    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().detach().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs

@HOOKS.register_module()
class Attack_OptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        # runner.optimizer.zero_grad()
        runner.model.zero_grad()
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
    #TODO 决定存储在哪里
    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead',
            DeprecationWarning)
        super().__init__(*args, **kwargs)

        # sefl.out_file = self.m
        self.alpha=0.2

        self.eps = 15 / 255 / VAR
        # self.


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


        for i, data_batch in enumerate(self.data_loader):  #所有图片

            self._inner_iter = i
            self.attack_train(data_batch, **kwargs)

            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def attack_train(self, data_batch, **kwargs):
        img = data_batch['img'].data[0]  # img的数据
        img_meta = data_batch['img_metas'].data[0][0]
        pertur = torch.zeros(img.shape, device=img.device)
        iter = 30


        for j in range(iter):
            self.call_hook('before_train_iter')

            # if j == 0:
            #     print(f"origin img{img}", img)
            #     ori_img = img
            pertur.requires_grad = True
            # perturbation = pertur.cuda()
            pertur_map = pert_map(pertur, self.eps)
            # print(pertur_map)
            attact_img = img + pertur_map
            data_batch['img']._data = attact_img

            # 每次iteration更新
            self.run_iter(data_batch, train_mode=True, **kwargs)

            if self.outputs['loss'] < 0.05 or j == iter - 1:
                img = data_batch['img'].data
                data_meata_list = data_batch['img_metas'].data[0]
                for idx, data_meta in enumerate(data_meata_list):  #

                    filename = data_meta['filename']
                    if filename.find("test") != -1:
                        filename = filename.replace('test', self.model_name + 'attac_testing' + str(VAR))
                    elif filename.find("training") != -1:
                        filename = filename.replace('training', +self.model_name + 'attac_training_' + str(VAR))

                    import os
                    (dir,file) = os.path.split(filename)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    import cv2

                    save_ing = img[idx]
                    # print(f"iter 30 {img}", img)
                    save_img = tensor2imgs(
                        save_ing.unsqueeze(0), **img_meta.get('img_norm_cfg', {}))[0]
                    cv2.imwrite(filename, save_img.astype(int))
                    print(f"{file} write success in {filename}" )
                    # sleep(60)
                    # return

            # if j == iter - 1:
            #     print(f"第{j}次 cost : {self.outputs['loss']}" ,j, self.outputs['loss'])

            self.call_hook('after_train_iter')
            pertur = pertur - self.alpha * pertur.grad.sign()
            # print(pertur)
            # print(pertur.grad)
            pertur = pertur.detach()


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
    elif "runner" in cfg:
        cfg.runner = {
            'type': cfg.runner,
            'max_epochs': cfg.total_epochs
        }
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