import torch as th
import torchvision as tv
from torch import nn
import torch.functional as F

class GroupNormalization(nn.Module):

    def __init__(self, channelNum, G,  eps=1e-5):

        super().__init__()
        self.channelNum = channelNum
        self.gamma = nn.Parameter(th.randn(self.channelNum, 1, 1))
        self.beta = nn.Parameter(th.randn(self.channelNum, 1, 1))
        self.G = G
        self.eps = eps


    def forward(self, x):

            N, C, H, W = x.size()

            x = x.view(N, self.G, -1)

            mean = x.mean(dim=2, keepdim=True)
            var = x.std(dim=2, keepdim=True)

            x = (x-mean) / th.sqrt(var+self.eps)

            x = x.view(N, C, H, W)

            out = self.gamma * x + self.beta

            return out


