__all__ = ['LearnPipelineV2']


import os
import sys
import time
import torch
import shutil
import random
import numpy as np
from pathlib import Path
from collections import OrderedDict
from .. import distributed as run_dist
from runner_master import runner
from torchvision.transforms import Compose as TransformCompose
from runner_master.runner import network_initializers
from runner_master.runner import models
from runner_master.runner.transforms import image as transforms_image
from runner_master.runner.transforms import extra as transforms_extra
from runner_master.runner.transforms import batch as transforms_batch
from runner_master.runner.data import imglists
from runner_master.runner.data import filereaders
from runner_master.runner.data import datasets
from runner_master.runner import schedulers
from runner_master.runner.optimizers import param_groups
from runner_master.runner import optimizers
from runner_master.runner.pipelines import rulers

class LearnPipelineV2:

    def __init__(self, config=None, rank=None, world_size=None):
        if None in (config, rank, world_size):
            self.config, self.rank, self.world_size = runner.setup(verbose=True)
        else:
            self.config, self.rank, self.world_size = config, rank, world_size

        self.total_sub_epoch = 0
        self.total_step = 0
        self.total_sample = 0

        self.get_network()
        self.get_loader()
        self.get_optimizer()
        self.get_scheduler()
        self.get_ruler()
        self.get_recorder()
        self.resume()
        self.get_writer()

    ################################ Prepare Resources ################################

    def get_network(self):
        self.network_dict = OrderedDict()
        for name in self.config.network.names:
            conf = self.config.network[name]
            lib = models
            for i, token in enumerate(conf.name.split('.')):
                lib = getattr(lib, token)
            network = lib(**conf.kwargs)
            network.cuda()
            torch.cuda.synchronize()
            for init_name, init_kwargs in conf.init.items():
                init_kwargs = dict(init_kwargs)
                use = init_kwargs.pop('use', True)
                if not use:
                    continue
                lib = network_initializers
                for i, token in enumerate(init_name.split('.')):
                    lib = getattr(lib, token)
                lib(network=network, **init_kwargs)
            self.network_dict[name] = network
            if self.rank == 0 and self.config.network.verbose:
                print('network [{}]'.format(name))
                print(network)

    def get_loader(self):
        self.loader_dict = OrderedDict()
        self.transform_batch_dict = OrderedDict()
        DataLoader = {
            'torch': torch.utils.data.DataLoader,
            'runner': runner.data.DataLoader,
        }[self.config.data.loader_class]
        for name in self.config.data.names:
            conf = self.config.data[name]
            transform_image = TransformCompose([
                getattr(transforms_image, name)(**kwargs)
                for name, kwargs in conf.transform_image.items()
            ])
            transform_extra = TransformCompose([
                getattr(transforms_extra, name)(**kwargs)
                for name, kwargs in conf.transform_extra.items()
            ])
            transform_collate = TransformCompose([
                getattr(transforms_batch, name)(**kwargs)
                for name, kwargs in conf.transform_collate.items()
            ])
            transform_batch = TransformCompose([
                getattr(transforms_batch, name)(**kwargs)
                for name, kwargs in conf.transform_batch.items()
            ])
            self.transform_batch_dict[name] = transform_batch

            if ('local_batch_size' in conf) == ('global_batch_size' in conf):
                raise RuntimeError('Should provide one and only one of [local_batch_size] and [global_batch_size].')
            batch_size = conf.get('local_batch_size', None) or (conf.global_batch_size // self.world_size)

            if not conf.use_infolist:
                lib = imglists
                for i, token in enumerate(conf.imglist_type.split('.')):
                    lib = getattr(lib, token)
                imglist = lib(conf.imglist_path)
                lib = filereaders
                for i, token in enumerate(conf.reader.split('.')):
                    lib = getattr(lib, token)
                reader = lib()
                lib = datasets
                for i, token in enumerate(conf.dataset.name.split('.')):
                    lib = getattr(lib, token)
                dataset = lib(imglist=imglist,
                    root=conf.root,
                    reader=reader,
                    transform_image=transform_image,
                    transform_extra=transform_extra,
                    skip_broken=conf.skip_broken,
                    **conf.dataset.kwargs,
                )
            else:
                lib = filereaders
                for i, token in enumerate(conf.reader.split('.')):
                    lib = getattr(lib, token)
                reader = lib()
                dataset = runner.data.datasets.build_mixed_dataset(
                    infolist=conf.imglist_path,
                    reader=reader,
                    transform_image=transform_image,
                    transform_extra=transform_extra,
                    skip_broken=conf.skip_broken,
                    default_imglist_type=conf.imglist_type,
                    default_dataset_type=conf.dataset.name,
                    **conf.dataset.kwargs,
                )
            if conf.num_sub_epochs == 1:
                sampler = runner.data.samplers.DistributedSampler(
                    dataset,
                    shuffle=conf.shuffle,
                    psudo_index=-1,
                )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=conf.num_workers,
                sampler=sampler,
                pin_memory=conf.pin_memory,
                drop_last=conf.drop_last,
                collate_fn=transform_collate,
            )
            self.loader_dict[name] = loader
            if self.rank == 0 and self.config.data.verbose:
                print('transform_image [{}]'.format(name))
                print(transform_image)
                print('transform_extra [{}]'.format(name))
                print(transform_extra)
                print('transform_batch [{}]'.format(name))
                print(transform_batch)
                print('loader [{}]'.format(name))
                print(loader)

    def get_scheduler(self):
        self.scheduler_dict = OrderedDict()
        unit_lengths = {
            'step': 1,
            'sub_epoch': len(self.loader_dict['train']),
            'epoch': len(self.loader_dict['train']) * self.config.num_sub_epochs,
        }
        for name in self.config.scheduler.names:
            conf = self.config.scheduler[name]
            lib = schedulers
            for i, token in enumerate(conf.name.split('.')):
                lib = getattr(lib, token)
            scheduler = lib(unit_length=unit_lengths[conf.unit_length], **conf.kwargs)
            self.scheduler_dict[name] = scheduler
            if self.rank == 0 and self.config.scheduler.verbose:
                print('scheduler [{}]'.format(name))
                print(scheduler)

    def get_optimizer(self):
        self.optimizer_dict = OrderedDict()
        for name in self.config.optimizer.names:
            network = self.network_dict[name]
            conf = self.config.optimizer[name]
            lib = param_groups
            for i, token in enumerate(conf.param_groups.split('.')):
                lib = getattr(lib, token)
            param_group = lib(network, **conf.kwargs)
            lib = optimizers
            for i, token in enumerate(conf.name.split('.')):
                lib = getattr(lib, token)
            optimizer = lib(param_group)
            if any(param.dtype == torch.float16 for param in network.parameters()):
                optimizer = runner.half.HalfOptimizer(optimizer, **conf.half_kwargs)
            runner.optimizers.check_optimizer(optimizer)
            self.optimizer_dict[name] = optimizer
            if self.rank == 0 and self.config.optimizer.verbose:
                print('optimizer [{}]'.format(name))
                print(optimizer)

    def get_ruler(self):
        self.fast_metric_names_dict = OrderedDict()
        self.slow_metric_names_dict = OrderedDict()
        self.tensor_cache_dict = OrderedDict()
        for name in self.config.metric.names:
            conf = self.config.metric[name]
            fast, slow = runner.utils.divide_metric_names(self.config.metric.speeds, conf.metric_names)
            self.fast_metric_names_dict[name] = fast
            self.slow_metric_names_dict[name] = slow
            self.tensor_cache_dict[name] = runner.utils.Rank0TensorCache(conf.cache_size)
            if len(slow) > 0 and name == 'train':
                print('Warning: using slow metrics for training is very slow')
        lib = rulers
        for i, token in enumerate(self.config.metric.ruler.split('.')):
            lib = getattr(lib, token)
        self.ruler = lib(self.config.metric)

    def get_recorder(self):
        self.recorder_dict = OrderedDict()
        for name in self.config.record.names:
            conf = self.config.record[name]
            recorder = runner.record.RecordManager(
                window_size=conf.window_size,
                summary_mode='mean',
                sync_cuda=conf.sync_cuda,
            )
            self.recorder_dict[name] = recorder

    def get_writer(self):
        pass
        # if self.rank == 0:
        #     self.tb_writer = runner.utils.init_tensorboard('..', self.total_sub_epoch)

    def add_scalar(self, tag, value, step):
        pass
        # self.tb_writer.add_scalar(tag, value, step)

    ################################ Resume Resources ################################

    def resume(self):
        path = self.config.resume
        if path == '' or not Path(path).is_file():
            if self.rank == 0:
                print('nothing to resume')
        else:
            if self.rank == 0:
                print('resume from [{}]'.format(path))
            ckpt = torch.load(path, map_location='cuda:{}'.format(torch.cuda.current_device()))
            for name, network in self.network_dict.items():
                network.load_state_dict(ckpt['network'][name])
            for name, loader in self.loader_dict.items():
                if hasattr(loader.sampler, 'load_state_dict'):
                    loader.sampler.load_state_dict(ckpt['sampler'][name])
            for name, optimizer in self.optimizer_dict.items():
                if 'milestones' in ckpt['optimizer'][name]:
                    del ckpt['optimizer'][name]['milestones']
                optimizer.load_state_dict(ckpt['optimizer'][name])
            for name, scheduler in self.scheduler_dict.items():
                scheduler.load_state_dict(ckpt['scheduler'][name])
            for name, recorder in ckpt['recorder'].items():
                self.recorder_dict[name] = recorder
            self.ruler.load_state_dict(ckpt['ruler'])
            self.total_sub_epoch = ckpt['total_sub_epoch']
            self.total_step = ckpt['total_step']
            self.total_sample = ckpt['total_sample']
            if 'rng_state' in ckpt:
                rng_state = ckpt['rng_state']
                random.setstate(rng_state['random'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch'].cpu())
                torch.cuda.set_rng_state(rng_state['torch.cuda'].cpu())
            self.extra_resume(ckpt)

    def extra_resume(self, ckpt):
        pass

    ################################ Running Pipelines ################################

    def run(self):
        num_total_sub_epochs = self.config.num_epochs * self.config.num_sub_epochs
        while self.total_sub_epoch < num_total_sub_epochs:
            # count from 0
            self.sub_epoch = self.total_sub_epoch % self.config.num_sub_epochs
            self.epoch = self.total_sub_epoch // self.config.num_sub_epochs

            # count from 1
            self.total_sub_epoch += 1
            self.sub_epoch += 1
            self.epoch += 1

            self.train_epoch()
            self.multi_eval_epoch()

            # mark an anchor into tensorboard for later resuming
            if self.rank == 0:
                runner.utils.anchor_tensorboard(self.tb_writer, self.total_sub_epoch)

            self.save()

    def set_network(self, name):
        for network_name, network in self.network_dict.items():
            network.train(name == 'train')

    def train_epoch(self):
        network = self.network_dict['main']
        self.set_network('train')
        self.train_dataiter = iter(self.loader_dict['train'])
        transform_batch = self.transform_batch_dict['train']
        optimizer = self.optimizer_dict['main']
        lr_scheduler = self.scheduler_dict['lr']
        recorder = self.recorder_dict['train']
        fast_metric_names = self.fast_metric_names_dict['train']
        slow_metric_names = self.slow_metric_names_dict['train']
        tensor_cache = self.tensor_cache_dict['train'] if len(slow_metric_names) > 0 else None

        for self.train_step in range(1, len(self.train_dataiter) + 1):
            with recorder.record_time('batch'):
                self.curr_lr = lr_scheduler.step(self.total_step)
                self.total_step += 1
                report_plain = self.train_step == 1 or self.train_step % self.config.report.plain_freq == 0
                report_curve = self.train_step == 1 or self.train_step % self.config.report.curve_freq == 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.curr_lr
                with recorder.record_time('loader'):
                    batch = next(self.train_dataiter)
                batch = transform_batch(batch)
                fast_metrics = self.train_one_step(network, batch, optimizer, fast_metric_names, tensor_cache)
                for name, metric in fast_metrics.items():
                    recorder.record_value(name, metric.item(), group='metric')
                if (report_plain or report_curve) and len(slow_metric_names) > 0:
                    slow_metrics = self.ruler(slow_metric_names, **tensor_cache.cat())
                    for name, metric in slow_metrics.items():
                        recorder.record_value(name, metric.item(), window_size=1, group='metric')
                if self.rank == 0 and report_plain:
                    self.train_report_plain()
                if self.rank == 0 and report_curve:
                    self.train_report_curve()
        del self.train_dataiter  # 不delete的话worker不会关闭，会影响后面的计算速度
        run_dist.sync_bn_stat(network)
        torch.cuda.synchronize()
        del batch
        torch.cuda.empty_cache()

    def train_one_step(self, network, batch, optimizer, fast_metric_names, tensor_cache):
        self.total_sample += batch['data'].size(0) * self.world_size
        optimizer.zero_grad()
        logit = network(batch['data'])
        fast_metrics = self.ruler(
            fast_metric_names,
            mul=1 / self.world_size,
            logit=logit,
            label=batch['label'],
            soft_label=batch['soft_label'],
        )
        if hasattr(optimizer, 'backward'):
            optimizer.backward(fast_metrics[self.config.metric.loss])
        else:
            fast_metrics[self.config.metric.loss].backward()
        run_dist.sync_grad_sum(network)
        run_dist.all_reduce_sum(list(fast_metrics.values()))
        optimizer.step()
        if tensor_cache is not None:
            tensor_cache.add(logit=logit, label=batch['label'])
        return fast_metrics

    def train_report_plain(self):
        recorder = self.recorder_dict['train']
        msg = [
            time.strftime('[%Y-%m-%d-%H:%M:%S]'),
            '[{}]'.format(self.config.run_name),
            'epoch [{}/{}]'.format(self.epoch, self.config.num_epochs),
        ]
        if self.config.num_sub_epochs > 1:
            msg.extend([
                'sub_epoch [{}/{}]'.format(self.sub_epoch, self.config.num_sub_epochs),
            ])
        msg.extend([
            'step [{}/{}]'.format(self.train_step, len(self.train_dataiter)),
            'lr [{:.02e}]'.format(self.curr_lr),
        ])
        for key, value in sorted(recorder.items()):
            msg.append('{} [{:.03f}]'.format(key, value))
        for key, value in sorted(recorder.items('metric')):
            msg.append('{} [{:.04f}]'.format(key, value))
        print(', '.join(msg))

    def train_report_curve(self):
        recorder = self.recorder_dict['train']
        for key, value in sorted(recorder.items()):
            self.add_scalar('time/{}_time'.format(key), value, self.total_step)
        for key, value in sorted(recorder.items('metric')):
            self.add_scalar('train-metric-step/{}'.format(key), value, self.total_step)
            self.add_scalar('train-metric-sample/{}'.format(key), value, self.total_sample)
            self.add_scalar('train-metric-lr/{}'.format(key), value, self.curr_lr * self.config.report.lr_axis_factor)
        self.add_scalar('other/lr-step', self.curr_lr, self.total_step)
        self.add_scalar('other/lr-sample', self.curr_lr, self.total_sample)

    def multi_eval_epoch(self):
        for eval_name in self.config.eval_names:
            with torch.no_grad():
                self.eval_epoch(eval_name)
        curr_metrics = OrderedDict(self.recorder_dict['eval'].items('curr-metric', group_must_exist=False))
        best_metrics = OrderedDict(self.recorder_dict['eval'].items('best-metric', group_must_exist=False))
        self.curr_is_best_dict = OrderedDict()
        for name in self.config.metric.critical_metric_names:
            self.curr_is_best_dict[name] = curr_metrics[name] == best_metrics[name]

        if self.rank == 0:
            msg = []
            for eval_name in self.config.eval_names:
                for name in self.config.metric[eval_name].metric_names:
                    full_name = '{}_{}'.format(eval_name, name)
                    curr = curr_metrics[full_name]
                    best = best_metrics[full_name]
                    msg.append('{} [{:.04f} ({:.04f})]'.format(full_name, curr, best))
                    lr_axis = self.curr_lr * self.config.report.lr_axis_factor
                    self.add_scalar('{}-metric-sub-epoch-curr/{}'.format(eval_name, name), curr, self.total_sub_epoch)  # noqa: E501
                    self.add_scalar('{}-metric-sub-epoch-best/{}'.format(eval_name, name), best, self.total_sub_epoch)  # noqa: E501
                    self.add_scalar('{}-metric-epoch-curr/{}'.format(eval_name, name), curr, self.epoch)
                    self.add_scalar('{}-metric-epoch-best/{}'.format(eval_name, name), best, self.epoch)
                    self.add_scalar('{}-metric-lr-curr/{}'.format(eval_name, name), curr, lr_axis)
                    self.add_scalar('{}-metric-lr-best/{}'.format(eval_name, name), best, lr_axis)
                    self.add_scalar('other/lr-sub-epoch', self.curr_lr, self.total_sub_epoch)
                    self.add_scalar('other/lr-epoch', self.curr_lr, self.epoch)
            print('\n'.join(msg))

    def eval_epoch(self, eval_name):
        if len(self.config.metric[eval_name].metric_names) == 0:
            print('no metric to evaluate for [{}]'.format(eval_name), flush=True)
            return
        network = self.network_dict['main']
        self.set_network(eval_name)
        eval_dataiter = iter(self.loader_dict[eval_name])
        transform_batch = self.transform_batch_dict[eval_name]
        recorder = self.recorder_dict['eval']
        fast_metric_names = self.fast_metric_names_dict[eval_name]
        slow_metric_names = self.slow_metric_names_dict[eval_name]
        tensor_cache = self.tensor_cache_dict[eval_name] if len(slow_metric_names) > 0 else None

        total_fast_metrics = OrderedDict()
        for name in ['num_samples'] + self.config.metric[eval_name].metric_names:
            total_fast_metrics[name] = 0.0

        for step in range(1, len(eval_dataiter) + 1):
            with recorder.record_time('batch', group=eval_name):
                batch = next(eval_dataiter)
                batch = transform_batch(batch)
                fast_metrics, num_samples = self.eval_one_step(network, batch, fast_metric_names, tensor_cache)
                for name, metric in fast_metrics.items():
                    total_fast_metrics[name] += metric.item() * num_samples
                total_fast_metrics['num_samples'] += num_samples
            if self.rank == 0 and step % self.config.report.plain_freq == 0:
                msg = [
                    time.strftime('[%Y-%m-%d-%H:%M:%S]'),
                    '[{}]'.format(self.config.run_name),
                    '[{}] eval step [{}/{}]'.format(eval_name, step, len(eval_dataiter)),
                ]
                for key, value in sorted(recorder.items(eval_name)):
                    msg.append('{} [{:.03f}]'.format(key, value))
                print(', '.join(msg))
        del eval_dataiter  # 不delete的话worker不会关闭，会影响后面的计算速度
        del batch
        torch.cuda.empty_cache()

        curr_metrics = OrderedDict()
        if len(fast_metric_names) > 0:
            fast_metric_tensor = torch.Tensor(list(total_fast_metrics.values())).cuda()
            run_dist.all_reduce_sum([fast_metric_tensor])
            fast_metric_tensor = fast_metric_tensor[1:] / fast_metric_tensor[0].item()
            curr_metrics.update(zip(fast_metric_names, fast_metric_tensor))
        if len(slow_metric_names) > 0:
            total_slow_metrics = self.ruler(slow_metric_names, **tensor_cache.cat())
            tensor_cache.reset()
            curr_metrics.update(total_slow_metrics)
        for name, value in curr_metrics.items():
            polar = self.config.metric.polars[name]
            name = '{}_{}'.format(eval_name, name)
            value = value.item()
            recorder.record_value(name, value, 'inf', polar, group='best-metric')
            recorder.record_value(name, value, 1, group='curr-metric')

    def eval_one_step(self, network, batch, fast_metric_names, tensor_cache):
        psudo = batch['psudo']
        num_samples = (1 - psudo).sum().item()
        logit = network(batch['data']).detach()
        if num_samples > 0 and len(fast_metric_names) > 0:
            fast_metrics = self.ruler(
                fast_metric_names,
                logit=logit[psudo == 0],
                label=batch['label'][psudo == 0],
                soft_label=batch['soft_label'][psudo == 0],
            )
        else:
            fast_metrics = OrderedDict()
        if tensor_cache is not None:
            tensor_cache.add(psudo, logit=logit, label=batch['label'])
        return fast_metrics, num_samples

    def save(self):
        if self.rank == 0:
            ckpt = {
                'epoch': self.epoch,
                'sub_epoch': self.sub_epoch,
                'total_sub_epoch': self.total_sub_epoch,
                'total_step': self.total_step,
                'total_sample': self.total_sample,

                'curr_metric': dict(),
                'best_metric': dict(),

                'network': OrderedDict(),
                'sampler': OrderedDict(),
                'optimizer': OrderedDict(),
                'scheduler': OrderedDict(),
                'recorder': OrderedDict(),
                'ruler': self.ruler.state_dict(),
            }

            for name, value in self.recorder_dict['eval'].items('curr-metric', group_must_exist=False):
                ckpt['curr_metric'][name] = value
            for name, value in self.recorder_dict['eval'].items('best-metric', group_must_exist=False):
                ckpt['best_metric'][name] = value

            for name, network in self.network_dict.items():
                ckpt['network'][name] = network.state_dict()
            for name, loader in self.loader_dict.items():
                if hasattr(loader.sampler, 'state_dict'):
                    ckpt['sampler'] = loader.sampler.state_dict()
            for name, optimizer in self.optimizer_dict.items():
                ckpt['optimizer'][name] = optimizer.state_dict()
            for name, scheduler in self.scheduler_dict.items():
                ckpt['scheduler'][name] = scheduler.state_dict()
            for name, recorder in self.recorder_dict.items():
                ckpt['recorder'][name] = recorder

            ckpt['rng_state'] = {
                'random': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch.cuda': torch.cuda.get_rng_state(),
            }

            self.extra_save(ckpt)

            # prepare save name, e.g. 120-001-001
            history_save_names = []
            for epoch in range(1, self.epoch + 1):
                for sub_epoch in range(1, self.config.num_sub_epochs + 1):
                    if epoch <= self.epoch and sub_epoch <= self.sub_epoch:
                        history_save_names.append('{:03d}-{:03d}-{:03d}.pth'.format(
                            epoch, sub_epoch, self.config.num_sub_epochs,
                        ))

            # checkpoint: contains the sufficient states for resuming, including but not limited to network paramsters
            subdir = 'checkpoint'
            self.save_subdir(subdir, ckpt, history_save_names[-1])
            self.clear_subdir(subdir, history_save_names)
            # state_dict: only contains the paramsters of each network
            for network_name, state_dict in ckpt['network'].items():
                subdir = 'state_dict_{}'.format(network_name)
                self.save_subdir(subdir, state_dict, history_save_names[-1])
                self.clear_subdir(subdir, history_save_names)

    def save_subdir(self, subdir, obj, save_name):
        fulldir = Path.cwd().parent / subdir
        fulldir.mkdir(parents=True, exist_ok=True)
        curr_path = fulldir / save_name
        latest_path = fulldir / 'latest.pth'

        torch.save(obj, str(curr_path))
        print('add diskfile [{}]'.format(curr_path.absolute()))
        shutil.copy(str(curr_path), str(latest_path))
        print('add diskfile [{}]'.format(latest_path.absolute()))


        for metric_name, curr_is_best in self.curr_is_best_dict.items():
            if curr_is_best:
                best_path = fulldir / 'best_{}.pth'.format(metric_name)
                shutil.copy(str(curr_path), str(best_path))
                print('add diskfile [{}]'.format(best_path.absolute()))

    def clear_subdir(self, subdir, history_save_names):
        fulldir = Path.cwd().parent / subdir
        max_save_history = self.config.get('save', {}).get('max_save_history', None)
        if max_save_history is None:
            return
        if max_save_history < 0:
            raise RuntimeError('invalid value for max_save_history: {}'.format(max_save_history))
        end_index = max(0, len(history_save_names) - max_save_history)
        for save_name in history_save_names[:end_index]:
            path = fulldir / save_name
            if path.is_file():
                print('del diskfile [{}]'.format(path))
                os.remove(str(path))

    def extra_save(self, ckpt):
        pass
