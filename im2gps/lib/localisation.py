import logging
import numpy as np
import typing as tp
from enum import Enum
from scipy.stats import multivariate_normal
from im2gps.lib.index import IndexType

log = logging.getLogger(__name__)


class LocalisationType(Enum):
    NN = ('1nn',)
    KDE = ('kde',)
    AVG = ('avg',)

    def __init__(self, type_str):
        self.type_str = type_str


def _kde(nn_coordinates: np.ndarray, weights: np.ndarray, sigma) -> tp.List[float]:
    tmp = np.array([weights[j] * multivariate_normal.pdf(nn_coordinates, mean=mu, cov=sigma ** 2 * np.eye(2))
                    for j, mu in enumerate(nn_coordinates)])
    pdf = np.sum(tmp, axis=0)
    return nn_coordinates[np.argmax(pdf)].tolist()


def _weighted_average(coordinates: np.ndarray, weights: tp.Union[np.ndarray, None] = None):
    if weights is None:
        q, k, _ = coordinates.shape
        weights = np.ones((q, k))
    w = weights / np.expand_dims(np.sum(weights, axis=1), axis=1)
    return np.sum(coordinates * w[:, :, np.newaxis], axis=1)


def l2_similarity_weights(dists: np.ndarray, m):
    eps = 0.0001
    return (1 / dists + eps) ** m


def cosine_similarity_weights(dists: np.ndarray, m):
    return (dists+1)**m


def localise_by_knn(coordinates: np.ndarray, loc_type, index_type, dist=None, **kwargs) -> tp.List[tp.List[float]]:
    locations = []
    if loc_type == LocalisationType.NN.type_str:
        locations = coordinates.squeeze().tolist()
    elif loc_type == LocalisationType.KDE.type_str:
        assert 'sigma' in kwargs and 'm' in kwargs, "parameters sigma and m should be be provided for kde"
        assert dist is not None, "dist should not be none"
        assert dist.shape == coordinates.shape[:2], f"shapes of dist and indices are different"
        for nn_coords, knn_dists in zip(coordinates, dist):
            weights = _get_weights(knn_dists, index_type, kwargs['m'])
            loc = _kde(np.array(nn_coords), weights, kwargs['sigma'])
            locations.append(loc)
    elif loc_type == LocalisationType.AVG.type_str:
        assert 'avg_type' in kwargs, 'avg_type should be provided'
        assert 'm' in kwargs, 'parameter m should be provided'

        if kwargs['avg_type'] == 'weighted':
            weights = _get_weights(dist, index_type, kwargs['m'])
        elif kwargs['avg_type'] == 'regular':
            weights = None
        else:
            raise ValueError(f'Unknown type of avg_type, {kwargs["avg_type"]}')

        locations = _weighted_average(np.array(coordinates), weights=weights).tolist()
    else:
        raise ValueError(f'Unknown localisation type {loc_type}')
    return locations


def _get_weights(dist, index_type, m):
    if index_type == IndexType.L2_INDEX:
        weights = l2_similarity_weights(dist, m)
    elif index_type == IndexType.COSINE_INDEX:
        weights = cosine_similarity_weights(dist, m)
    else:
        raise ValueError(f'Unknown index type: {index_type}')
    return weights
