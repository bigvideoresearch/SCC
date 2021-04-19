import torch
from torch import nn
import torch.nn.functional as F

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)
        return x
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class GeM(nn.Module):

    def __init__(self, eps=1e-6, p=1.0):
        super(GeM, self).__init__()
        #self.p = nn.Parameter(torch.ones(()))
        self.p = p
        self.eps = eps

    def forward(self, x):
        # print('gem.py 1 x.shape=', x.shape, flush=True)
        _, _, H, W = x.size()
        x = x.clamp(min=self.eps)
        x = F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        # print('gem.py 2 x.shape=', x.shape, flush=True)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def modify(net, conf):
    modes = conf.method
    for mode in modes:
        if mode in ('gem'):
            net.avgpool = GeM(p=conf.gem.p)
        elif mode in ('use_backbone'):
            net.fc = Identity()
        elif mode in ('l2n'):
            net.fc = L2N()
        elif mode in ('projector'):
            net.fc = nn.Sequential(
                nn.Linear(conf.projector.n_features, conf.projector.n_features),
                nn.ReLU(inplace=True),
                nn.Linear(conf.projector.n_features, conf.projector.projection_dim)
                )
        elif mode in ('projector_conv'):
            net.view = nn.Sequential()
            net.dropout = nn.Sequential()
            net.fc = nn.Sequential(
                nn.Conv2d(conf.projector.n_features, conf.projector.n_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(conf.projector.n_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(conf.projector.n_features, conf.projector.projection_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(conf.projector.projection_dim)
                )
        else:
            raise RuntimeError('unknown mode: [{}]'.format(mode))
    return net
    
