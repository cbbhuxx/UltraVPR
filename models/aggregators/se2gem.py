import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class se2gem(nn.Module):
    """
    The se2gem removes the normalization layer that follows the input in CosPlace.
    CosPlace aggregation layer as implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py
    Args:
        in_dim: number of channels of the input
        out_dim: dimension of the output descriptor 
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x): 
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        # norms = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
        # x = x / norms
        return x

if __name__ == '__main__':
    x = torch.randn(4, 256, 10, 10)
    m = se2gem(256, 256)
    r = m(x)
    print(r.shape)