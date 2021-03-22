import torch.nn as nn
from layers import Descriptors2Weights, KDE


class Network(nn.Module):
    def __init__(self, *features, m, dist_type, transform_only=True):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.transform_only = transform_only
        self.d2w = Descriptors2Weights(m, dist_type)
        self.kde = KDE()

    def forward(self, *inputs, **kwargs):
        descriptors = self.features(inputs[0])

        if not self.transform_only:
            q = self.features(inputs[1])
            weights = self.d2w(q, descriptors)
            out = self.kde(weights, descriptors, inputs[2])
