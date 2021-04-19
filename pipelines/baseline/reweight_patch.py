import torch
import runner_master.runner as runner
import runner_master.runner.distributed as run_dist
from runner_master.runner.pipelines import LearnPipelineV2
from collections import OrderedDict
from torchvision.transforms import Compose as TransformCompose


@runner.patch_pipeline('ReWeightLearnPipeline')
class ReWeightLearnPipeline(LearnPipelineV2):

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
                batch['sample_weight'] = batch['sample_weight'].float()
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

        if self.config.metric.loss.startswith('rew'):
            fast_metrics = self.ruler(
                fast_metric_names,
                mul=1 / self.world_size,
                logit=logit,
                label=batch['label'],
                soft_label=batch['soft_label'],
                sample_weight=batch['sample_weight'],
            )
        else:
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
