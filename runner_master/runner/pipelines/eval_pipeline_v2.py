__all__ = ['EvalPipelineV2']


import sys
import torch
from runner_master import runner
from . learn_pipeline_v2 import LearnPipelineV2


class EvalPipelineV2(LearnPipelineV2):

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
        self.get_ruler()
        self.get_recorder()

    def run(self):
        self.multi_eval_epoch()

    def multi_eval_epoch(self):
        for eval_name in self.config.eval_names:
            with torch.no_grad():
                self.eval_epoch(eval_name)
        if self.rank == 0:
            msg = []
            curr_metrics = {k: v for k, v in self.recorder_dict['eval'].items('curr-metric')}
            for eval_name in self.config.eval_names:
                for name in self.config.metric[eval_name].metric_names:
                    full_name = '{}_{}'.format(eval_name, name)
                    msg.append('{} [{:.04f}]'.format(full_name, curr_metrics[full_name]))
            print(', '.join(msg))
