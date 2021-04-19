import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy(outputs, labels, T=None):
    if T is not None:
        return cross_entropy(outputs / T, labels / T) * T ** 2

    if outputs.shape != labels.shape: # This is the case when labels are one-hot
        return F.cross_entropy(outputs, labels,size_average=True)
    else:
        return -torch.mean(torch.sum(torch.mul(F.log_softmax(outputs, dim=1),
                             F.softmax(labels, dim=1)),dim=1))
