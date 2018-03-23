import torch as th
import torchvision as tv
from torch import nn
import torch.functional as F

class GroupNormalization(nn.Module):

    def __init__(self, G, train = False, eps=1e-5):

        super().__init__()
        self.gamma = nn.Parameter(th.randn(1))
        self.beta = nn.Parameter(th.randn(1))
        self.G = G
        self.eps = eps
        self.mean = 0
        self.var = 0
        self.train = train

    def forward(self, x):

        if self.train == True:

            N, C, H, W = x.size().data
            x = x.view(N, self.G, C//self.G, H, W)

            mean = th.mean(x, dim=[2, 3, 4], keepdim=True)
            var = th.var(x, dim=[2, 3, 4], keepdim=True)

            x = (x-mean) / th.sqrt(var+self.eps)

            x = x.view(N, C, H, W)

            return self.gamma * x + self.beta

        if self.train == False:
            pass

