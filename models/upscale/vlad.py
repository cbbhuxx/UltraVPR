import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class vlad(nn.Module):
    def __init__(self, num_classes=125, dim=512, vladv2=False):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.centroids = nn.Parameter(torch.rand(num_classes, dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        N, D = x.shape[:2]

        vlad = torch.zeros([N, self.num_classes, D], dtype=x.dtype, layout=x.layout, device=x.device)
        self.centroids.data = self.centroids.data.to(device)

        gem_des_flatten = x.view(N, D, -1)
        feature_x = gem_des_flatten.permute(0, 2, 1)
        dist_matrix = torch.pairwise_distance(feature_x, self.centroids.unsqueeze(0))
        assign = F.softmax(1/dist_matrix, dim=1)
        for C in range(self.num_classes):
            residual = gem_des_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                        self.centroids[C:C + 1, :].expand(gem_des_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= assign[:, C:C + 1].unsqueeze(2).unsqueeze(3)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def init_params(self, clsts):
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        print("Load the classification centers completed.")


def main():
    x = torch.randn(96, 128)

    upscaling = vlad(31, dim=128)
    output = upscaling(x)

    print(output.shape)


if __name__ == '__main__':
    main()