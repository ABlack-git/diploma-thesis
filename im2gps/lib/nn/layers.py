from .enum import NNEnum
import functional as f
import torch.nn as nn


class Descriptors2Weights(nn.Module):
    def __init__(self, m: int, dist_type=NNEnum.L2_DIST):
        super().__init__()
        self.m = m
        self.dist_type: NNEnum = dist_type

    def forward(self, q, descs):
        dists = f.distance_from_query(q, descs, dist_type=self.dist_type)
        return f.dist2weights(dists, self.m, dist_type=self.dist_type)

    def __repr__(self):
        return self.__class__.__name__ + f'(m={self.m}, metric={self.dist_type.name})'


class KDE(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, weights, coordinates):
        """
        :param weights: BxN
        :param coordinates: BxNx2
        :return:
        """
        return f.kde(weights, coordinates, self.sigma)

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma})'


class HaversineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, ground_true):
        return f.haversine_loss(predicted, ground_true)
