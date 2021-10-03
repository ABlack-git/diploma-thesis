import torch
import torch.nn as nn

import im2gps.core.nn.functional as f
from im2gps.core.nn.enum import NNEnum


class L2NormalizationLayer(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2NormalizationLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return f.l2_normalization(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f'(eps={self.eps})'


class Descriptors2Weights(nn.Module):
    def __init__(self, m: float, trainable=True, dist_type=NNEnum.L2_DIST):
        super().__init__()
        self.m = nn.Parameter(data=torch.tensor(m, dtype=torch.float), requires_grad=trainable)
        if isinstance(dist_type, NNEnum):
            self.dist_type: NNEnum = dist_type
        elif isinstance(dist_type, str):
            self.dist_type = NNEnum(dist_type)
        else:
            raise ValueError(f"Unknown type for parameter dist_type: {type(dist_type)}")

    def forward(self, q, descs):
        dists = f.distance_from_query(q, descs, dist_type=self.dist_type)
        return f.dist2weights(dists, self.m, dist_type=self.dist_type)

    def __repr__(self):
        return self.__class__.__name__ + f'(m={self.m}, metric={self.dist_type.value})'


class KDE(nn.Module):
    def __init__(self, sigma, trainable=True):
        super().__init__()
        self.sigma = nn.Parameter(data=torch.tensor(sigma), requires_grad=trainable)

    def forward(self, weights, coordinates):
        """
        :param weights: BxN
        :param coordinates: BxNx2
        :return:
        """
        pdf = f.kde(weights, coordinates, self.sigma)
        if self.training:
            return pdf
        else:
            # get points with max prob in coordinates
            pass

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma={self.sigma})'


class HaversineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, ground_true):
        return f.haversine_loss(predicted, ground_true)


class KDELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pdf, coord_space, true_coords):
        return f.kde_loss(pdf, coord_space, true_coords)
