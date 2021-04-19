import torch
from collections import OrderedDict
from runner_master import runner
import runner_master.runner.distributed as run_dist
from ..baseline.reweight_patch import ReWeightLearnPipeline
from torchvision.transforms import Compose as TransformCompose
from runner_master.runner.transforms import image as transforms_image
from runner_master.runner.transforms import extra as transforms_extra
from runner_master.runner.transforms import batch as transforms_batch
from runner_master.runner.data import datasets
from runner_master.runner.data import imglists
from runner_master.runner.data import filereaders

def ScalingConf(conf, t):
    conf = torch.pow(conf, t).float()
    return conf

@runner.patch_pipeline('OfflineSelfContainedConfPipeline')
class OfflineSelfContainedConfPipeline(ReWeightLearnPipeline):

    def train_epoch(self):
        network = self.network_dict['main']
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
                #######################################################################################################
                #                                 labeler preprocessing                                               #
                #######################################################################################################
                with torch.no_grad():
                    score = batch['pseudo_label']

                    # Get pseudo labels
                    if self.config.SCC_setting.label_sharpenT == 0:  # one-hot pseudo labels
                        pseudo = torch.argmax(score, dim=1).long()
                        batch['pseudo_label'] = torch.zeros(score.size(0),
                                                            score.size(1)).scatter_(1,pseudo.view(-1, 1),1).float()
                        del pseudo
                    elif self.config.SCC_setting.label_sharpenT == 1:
                        batch['pseudo_label'] = score
                    else:
                        batch['pseudo_label'] = torch.pow(score, 1 / self.config.SCC_setting.label_sharpenT)

                    # Get confidence scores
                    if self.config.SCC_setting.conf_power != 1:  # confidence calibration, otherwise no change
                        batch['conf_score'] = ScalingConf(batch['conf_score'], self.config.SCC_setting.conf_power)
                    # remove intermediate var
                    del score

                self.set_network('train')

                # get final pseudo label
                conf_policy = self.config.SCC_setting.conf_policy
                if conf_policy.startswith('ConstantConf'):
                    conf_constant = float(conf_policy.split('_')[1])
                    batch['soft_label'] = conf_constant * batch['soft_label'] \
                                          + (1 - conf_constant) * batch['pseudo_label']
                elif conf_policy.startswith('ConfvsOneMinusConf'):
                    batch['soft_label'] = batch['conf_score'].view(-1, 1) * batch['soft_label'] \
                                          + (1 - batch['conf_score']).view(-1, 1) * batch['pseudo_label']
                elif conf_policy.startswith('Distillation'):
                    batch['soft_label'] = batch['pseudo_label'].clone()
                else:
                    raise RuntimeError('Unknown Confidence Policy.')
                #changed the position of transform_batch
                batch['sample_weight'] = batch['sample_weight'].float()
                batch = transform_batch(batch)
                #######################################################################################################
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

            lib = imglists
            for i, token in enumerate(conf.imglist_type.split('.')):
                lib = getattr(lib, token)
            imglist = lib(conf.imglist_path)
            lib = filereaders
            for i, token in enumerate(conf.reader.split('.')):
                lib = getattr(lib, token)
            reader = lib()
            if ('local_batch_size' in conf) == ('global_batch_size' in conf):
                raise RuntimeError('Should provide one and only one of [local_batch_size] and [global_batch_size].')
            batch_size = conf.get('local_batch_size', None) or (conf.global_batch_size // self.world_size)
            if conf.dataset.name == 'PseudoDataset':
                lib = datasets
                for i, token in enumerate(conf.dataset.name.split('.')):
                    lib = getattr(lib, token)
                dataset = lib(
                    conf.pseudo_root,
                    imglist,
                    conf.root,
                    reader,
                    transform_image,
                    transform_extra,
                    skip_broken=conf.skip_broken,
                    **conf.dataset.kwargs,
                )
            else:
                lib = datasets
                for i, token in enumerate(conf.dataset.name.split('.')):
                    lib = getattr(lib, token)
                dataset = lib(
                    imglist,
                    conf.root,
                    reader,
                    transform_image,
                    transform_extra,
                    skip_broken=conf.skip_broken,
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
