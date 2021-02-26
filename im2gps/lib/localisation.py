import logging
import numpy as np
import typing as tp
from enum import Enum
from KDEpy import NaiveKDE
from im2gps.lib.index import IndexType

log = logging.getLogger(__name__)


class LocalisationType(Enum):
    NN = ('1nn',)
    KDE = ('kde',)
    AVG = ('avg',)

    def __init__(self, type_str):
        self.type_str = type_str


def __get_grid(centers, sigma, bw=3, n=5) -> np.ndarray:
    """
    Creates local grid for each point in centers.
    :param centers: Nx2 array of input points
    :param sigma: sigma of distributions
    :param bw: bandwidth, number of sigmas to use for grid per axes
    :param n: number of points per sigma
    :return: N*(2 * bw * n + 1) * (2 * bw * n + 1) points, because (2 * bw * n + 1) * (2 * bw * n + 1) grid is created
    for each point in centers.
    """
    tl = centers - (bw * sigma, -bw * sigma)  # transform centers of a grid to top left corner
    # 2 * bw * n + 1 is number of points on each axes of a grid
    x_space = np.linspace(tl[:, 0], tl[:, 0] + (2 * bw * sigma), num=2 * bw * n + 1).T
    y_space = np.linspace(tl[:, 1], tl[:, 1] - (2 * bw * sigma), num=2 * bw * n + 1).T

    positions = []
    for x, y in zip(x_space, y_space):
        xx, yy = np.meshgrid(x, y)
        positions.append(np.vstack([xx.ravel(), yy.ravel()]).T)

    return np.concatenate(positions)


def _kde(nn_coordinates, weights, sigma):
    kde: NaiveKDE = NaiveKDE(kernel='gaussian', bw=sigma).fit(data=nn_coordinates, weights=weights)
    points = __get_grid(nn_coordinates, sigma)
    y = kde.evaluate(points)
    return points[np.argmax(y)]


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
    return (dists + 1) ** m


def localise_by_knn(coordinates: np.ndarray, loc_type, index_type, dist=None, **kwargs) -> tp.List[tp.List[float]]:
    locations = []
    if loc_type == LocalisationType.NN.type_str:
        locations = coordinates.squeeze().tolist()
    elif loc_type == LocalisationType.KDE.type_str:
        assert 'sigma' in kwargs and 'm' in kwargs, "parameters sigma and m should be be provided for kde"
        assert dist is not None, "dist should not be none"
        assert dist.shape == coordinates.shape[:2], f"shapes of dist and indices are different"
        log.info("Starting KDE")
        for nn_coords, knn_dists in zip(coordinates, dist):
            weights = _get_weights(knn_dists, index_type, kwargs['m'])
            loc = _kde(np.array(nn_coords), weights, kwargs['sigma'])
            locations.append(loc)
    elif loc_type == LocalisationType.AVG.type_str:
        assert 'avg_type' in kwargs, 'avg_type should be provided'
        assert 'm' in kwargs, 'parameter m should be provided'
        log.info("Starting AVG")
        if kwargs['avg_type'] == 'weighted':
            weights = _get_weights(dist, index_type, kwargs['m'])
        elif kwargs['avg_type'] == 'regular':
            weights = None
        else:
            raise ValueError(f'Unknown type of avg_type, {kwargs["avg_type"]}')

        locations = _weighted_average(np.array(coordinates), weights=weights).tolist()
    else:
        raise ValueError(f'Unknown localisation type {loc_type}')
    log.info("Localisation done")
    return locations


def _get_weights(dist, index_type, m):
    if index_type == IndexType.L2_INDEX:
        weights = l2_similarity_weights(dist, m)
    elif index_type == IndexType.COSINE_INDEX:
        weights = cosine_similarity_weights(dist, m)
    else:
        raise ValueError(f'Unknown index type: {index_type}')
    return weights
