import torch
import time
from pathlib import Path
from runner_master import runner
import runner_master.runner.distributed as run_dist
from collections import OrderedDict
from runner_master.runner import models
from runner_master.runner import network_initializers
import numpy as np
import os

from networks.resnet_modify import modify


class ExtractFeaturePipeline(runner.pipelines.EvalPipelineV2):

    def get_network(self):
        self.network_dict = OrderedDict()
        for name in self.config.network.names:
            conf = self.config.network[name]
            lib = models
            for i, token in enumerate(conf.name.split('.')):
                lib = getattr(lib, token)
            network = lib(**conf.kwargs)
            if conf.modify.use:
                network = modify(network, conf.modify)
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

    def eval_epoch(self, eval_name):
        feature_network = self.network_dict['feature_net']
        label_network = self.network_dict['label_net']

        self.set_network('eval')

        eval_dataiter = iter(self.loader_dict[eval_name])
        transform_batch = self.transform_batch_dict[eval_name]
        recorder = self.recorder_dict[eval_name]
        fast_metric_names = self.fast_metric_names_dict[eval_name]
        slow_metric_names = self.slow_metric_names_dict[eval_name]
        tensor_cache = runner.utils.Rank0TensorCache()

        total_fast_metrics = OrderedDict()
        for name in ['num_samples'] + self.config.metric[eval_name].metric_names:
            total_fast_metrics[name] = 0.0

        score_list = []
        feature_list = []
        nline_list = []

        for step in range(1, len(eval_dataiter) + 1):
            with recorder.record_time('batch', group=eval_name):
                batch = next(eval_dataiter)
                batch = transform_batch(batch)
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                soft_label = batch['soft_label'].cuda()
                psudo = batch['psudo'].cuda()
                sample_weight = batch['sample_weight'].cuda()
                image_name = batch['image_name']
                feature = feature_network(data).detach()
                logit = label_network(data).detach()
                score = torch.nn.functional.sigmoid(logit)
                conf = torch.sum(soft_label * score, dim=1)
                pseudo = torch.argmax(logit, dim=1)
                correct_rate = torch.sum(pseudo == label)
                feature = feature[psudo == 0].cpu()
                score = score[psudo == 0].cpu()
                label = label[psudo == 0].cpu()
                sample_weight = sample_weight[psudo == 0].cpu()
                conf = conf[psudo == 0].cpu()
                image_name = [n for n, p in zip(image_name, psudo.tolist()) if p == 0]
                ############### get features and scores ###################################
                num_samples = (1 - psudo).sum().item()
                for idx in range(num_samples):
                    score_list.append(score[idx].tolist())
                    feature_list.append(feature[idx].tolist())
                    newline = image_name[idx] + ' {"label": ' + str(label[idx].item()) + ', "sample_weight": ' + str(
                        sample_weight[idx].item()) + ', "conf_score": ' + str(conf[idx].item()) + '}\n'
                    nline_list.append(newline)
                ############################################################################
                fast_metrics, num_samples = self.eval_one_step(label_network, batch, fast_metric_names, tensor_cache)
                for name, metric in fast_metrics.items():
                    total_fast_metrics[name] += metric.item() * num_samples
                total_fast_metrics['num_samples'] += num_samples
            if self.rank == 0 and step % self.config.report.plain_freq == 0:
                msg = [
                    time.strftime('[%Y-%m-%d-%H:%M:%S]'),
                    '[{}]'.format(self.config.run_name),
                    '[{}] eval step [{}/{}]'.format(eval_name, step, len(eval_dataiter)),
                    'correct rate [{}/{}]'.format(correct_rate, num_samples),
                ]
                for key, value in sorted(recorder.items(eval_name)):
                    msg.append('{} [{:.03f}]'.format(key, value))
                print(', '.join(msg))
        del eval_dataiter  # 不delete的话worker不会关闭，会影响后面的计算速度
        del batch
        torch.cuda.empty_cache()

        if self.rank == 0:
            print('#########starting save features and imglist...############')
        save_root = self.config.save_root
        file_name = self.config.save_filename
        nlist_name = file_name + '_imglist.txt'
        feature_name = file_name + '_feature.npy'
        score_name = file_name + '_score.npy'
        save_dir = Path(save_root)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save('{}/feature_rank_{}.npy'.format(save_root, self.rank), np.array(feature_list))
        np.save('{}/score_rank_{}.npy'.format(save_root, self.rank), np.array(score_list))
        with open('{}/nlist_rank_{}.txt'.format(save_root, self.rank), 'w') as f:
            f.writelines(nline_list)
        ############### Merge ###################
        run_dist.barrier()
        if self.rank == 0:
            score_tensor = None
            feature_tensor = None
            for i in range(self.world_size):
                with open('{}/nlist_rank_{}.txt'.format(save_root, i), 'r') as reader, \
                        open('{}/{}'.format(save_root, nlist_name), 'a+') as writer:
                    for line in reader.readlines():
                        writer.write(line)
                os.remove('{}/nlist_rank_{}.txt'.format(save_root, i))

                temp_feature = torch.from_numpy(np.load('{}/feature_rank_{}.npy'.format(save_root, i)))
                temp_score = torch.from_numpy(np.load('{}/score_rank_{}.npy'.format(save_root, i)))
                if score_tensor is None:
                    score_tensor = temp_score
                else:
                    score_tensor = torch.cat((score_tensor, temp_score), dim=0)
                os.remove('{}/score_rank_{}.npy'.format(save_root, i))

                if feature_tensor is None:
                    feature_tensor = temp_feature
                else:
                    feature_tensor = torch.cat((feature_tensor, temp_feature), dim=0)
                os.remove('{}/feature_rank_{}.npy'.format(save_root, i))
            np.save('{}/{}'.format(save_root, feature_name), feature_tensor.float())
            np.save('{}/{}'.format(save_root, score_name), score_tensor.float())
        ######################################################

    def multi_eval_epoch(self):
        for eval_name in self.config.eval_names:
            with torch.no_grad():
                self.eval_epoch(eval_name)


runner.patch_pipeline('ExtractFeaturePipeline', ExtractFeaturePipeline)
