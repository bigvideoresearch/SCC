import torch
import torch.nn.functional as F

import runner_master.runner as runner
import runner_master.runner.distributed as run_dist
from runner_master.runner.pipelines.rulers import ClassifyRuler
from runner_master.runner.metrics import topk_accuracy


class ReweightRuler(ClassifyRuler):

    def aol_bce(self, **data):
        sum_bce = F.binary_cross_entropy_with_logits(data['logit'], data['soft_label'], reduction='none')
        return (sum_bce.sum(dim=1)).mean()

    def rew_aol_bce(self, **data):
        losses = F.binary_cross_entropy_with_logits(data['logit'], data['soft_label'], reduction='none')
        return (losses.sum(dim=1) * data['sample_weight'].type_as(losses)).mean()

    def avg_top1(self, **data):
        if self.rank == 0:
            accu = topk_accuracy(data['logit'], data['label'], 1, class_average=True)
        else:
            accu = torch.zeros(1)
        accu = accu.cuda()
        run_dist.broadcast([accu], 0)
        return accu

    def avg_top5(self, **data):
        if self.rank == 0:
            accu = topk_accuracy(data['logit'], data['label'], 5, class_average=True)
        else:
            accu = torch.zeros(1)
        accu = accu.cuda()
        run_dist.broadcast([accu], 0)
        return accu

    def top1(self, **data):
        return topk_accuracy(data['logit'], data['label'], 1)

    def top5(self, **data):
        return topk_accuracy(data['logit'], data['label'], 5)


runner.patch_ruler('ReweightRuler', ReweightRuler)
