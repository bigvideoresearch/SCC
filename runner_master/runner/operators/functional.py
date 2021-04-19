__all__ = ['Functional']


from torch import nn


class Functional(nn.Module):

    def __init__(self, f):
        super(Functional, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)
