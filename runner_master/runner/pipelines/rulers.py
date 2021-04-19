__all__ = ['Ruler', 'ClassifyRuler']


import torch
from .. import metrics
import torch.nn.functional as F
from collections import OrderedDict
from .. import distributed as run_dist


class Ruler:

    def __init__(self, conf):
        self.conf = conf
        self.world_size = run_dist.get_world_size()
        self.rank = run_dist.get_rank()

    def __call__(self, metric_names, mul=1, **data):
        self.results = OrderedDict()
        cache = dict()
        for name in metric_names:
            if name in cache:
                self.results[name] = cache.pop(name) * mul
                continue
            f_name = self.conf.get(name, {}).get('use', name)
            f_kwargs = self.conf.get(name, {}).get('kwargs', {})
            f = getattr(self, f_name)
            value = f(**f_kwargs, **data)
            if isinstance(value, dict):
                value_dict = value
                for key, value in value_dict.items():
                    if key == name:
                        self.results[key] = value * mul
                    else:
                        cache[key] = value
            else:
                self.results[name] = value * mul
        results = self.results
        del self.results
        return results

    def state_dict(self):
        return None

    def load_state_dict(self, state):
        pass


class ClassifyRuler(Ruler):

    def ce(self, **data):
        return F.cross_entropy(data['logit'], data['label'])

    def sce(self, **data):
        return metrics.soft_cross_entropy(data['logit'], data['soft_label'])

    def bce(self, **data):
        sum_bce = F.binary_cross_entropy_with_logits(data['logit'], data['label'], reduction='sum')
        return sum_bce / data['logit'].size(0)  # 沿batch_size维度mean，沿class维度sum

    def top1(self, **data):
        return metrics.topk_accuracy(data['logit'], data['label'], 1)

    def top5(self, **data):
        return metrics.topk_accuracy(data['logit'], data['label'], 5)

    def mapscore(self, **data):
        # 只有rank0传进的data非None
        import pdb; pdb.set_trace()
        if self.rank == 0:
            # 按chunksize沿类维度分块计算，否则爆显存
            score = metrics.mapscore(data['logit'], data['label'], chunksize=100)
        else:
            score = torch.zeros(1).cuda()
        run_dist.broadcast([score], 0)
        return score
