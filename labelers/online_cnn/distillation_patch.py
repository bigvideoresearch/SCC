import time
import torch
from collections import OrderedDict
from runner_master import runner
import runner_master.runner.distributed as run_dist
from runner_master.runner.pipelines import LearnPipelineV2
from torchvision.transforms import Compose as TransformCompose
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
import numpy as np

class DistillationPipeline(LearnPipelineV2):

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
            if conf.dataset.name == 'ReadPseudoDatasetV2':
                lib = datasets
                for i, token in enumerate(conf.dataset.name.split('.')):
                    lib = getattr(lib, token)
                dataset = lib(
                    conf.dataset.name,
                    imglist=imglist,
                    root=conf.root,
                    pseudo_root=conf.pseudo_root,
                    reader=reader,
                    transform_image=transform_image,
                    transform_extra=transform_extra,
                    skip_broken=conf.skip_broken,
                    **conf.dataset.kwargs,
                )
            else:
                lib = datasets
                for i, token in enumerate(conf.dataset.name.split('.')):
                    lib = getattr(lib, token)
                dataset = lib(
                    conf.dataset.name,
                    imglist=imglist,
                    root=conf.root,
                    reader=reader,
                    transform_image=transform_image,
                    transform_extra=transform_extra,
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


    def train_one_step(self, network, batch, optimizer, fast_metric_names, tensor_cache):
        self.total_sample += batch['data'].size(0) * self.world_size
        with torch.no_grad():
            # label calibration
            score = batch['pseudo_label']
            if self.config.distillation.label_sharpenT == 0: # one-hot pseudo labels
                pseudo = torch.argmax(score, dim=1).long()
                batch['pseudo_label'] = torch.zeros(score.size(0), score.size(1)).scatter_(1, pseudo.view(-1,1), 1)
                del pseudo
            elif self.config.distillation.label_sharpenT == 1:
                pass
            else:
                batch['pseudo_label'] = torch.pow(score, 1/self.config.distillation.label_sharpenT).float()
            del score

        optimizer.zero_grad()
        logit = network(batch['data'])

        fast_metrics = self.ruler(
            fast_metric_names,
            mul=1 / self.world_size,
            logit=logit,
            label=batch['label'],
            soft_label=batch['soft_label'],
            sample_weight=batch['sample_weight'],
            pseudo_label=batch['pseudo_label'],
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

runner.patch_pipeline('DistillationPipeline', DistillationPipeline)
