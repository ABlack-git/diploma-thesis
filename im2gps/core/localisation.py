import time

import logging
import numpy as np
import typing as t
from enum import Enum
from KDEpy import NaiveKDE
from multiprocessing import Pool

import im2gps.utils as utils
from im2gps.core.index import IndexType, Index, IndexBuilder, IndexConfig

log = logging.getLogger(__name__)


class LocalisationType(Enum):
    NN = 'nn'
    KDE = 'kde'
    AVG = 'avg'
    WEIGHTED_AVG = 'w_avg'


def get_grid(centers, sigma, bw=3, n=3) -> np.ndarray:
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


def _weighted_average(coordinates: np.ndarray, weights: t.Union[np.ndarray, None] = None):
    if weights is None:
        q, k, _ = coordinates.shape
        weights = np.ones((q, k))
    w = weights / np.expand_dims(np.sum(weights, axis=1), axis=1)
    return np.sum(coordinates * w[:, :, np.newaxis], axis=1)


def l2_similarity_weights(dists: np.ndarray, m):
    eps = 0.0001
    return (1 / (dists + eps)) ** m


def cosine_similarity_weights(dists: np.ndarray, m):
    return (dists + 1) ** m


class LocalisationModel:
    def __init__(self, localisation_type, index: t.Union[Index, IndexConfig], sigma=0.1, m=1, k=1, num_workers=0):
        self.sigma: float = sigma
        self.m: int = m
        self.k = k
        self.localisation_type = localisation_type
        self.num_workers = num_workers
        self._index: t.Optional[Index] = None
        self._index_config: t.Optional[IndexConfig] = None

        if isinstance(index, IndexConfig):
            self._index_config = index
        elif isinstance(index, Index):
            self._index = index
        else:
            raise ValueError(
                f"Unexpected type of index {type(self._index)}. Index should be instance of Index or IndexConfig")

        self._trained: bool = False
        self._coordinate_map = None

    def fit(self, ids, coordinates, descriptors=None):
        """
        :param data: dict with coordinates, ids and descriptors
        :return:
        """
        if not self._trained:
            log.debug("Fitting localisation model...")
            self._coordinate_map = LocalisationModel.compute_coordinate_map(ids, coordinates)
            if self._index_config is not None:
                if self._index_config.index_dir is not None:
                    log.debug("Index directory was provided, will load index from disk")
                    self._load_index()
                else:
                    log.debug("Building index...")
                    self._index = IndexBuilder(self._index_config, descriptors, ids).build()
            elif self._index is not None:
                log.debug(f"Using provided index {repr(self._index)}")
            else:
                raise ValueError("Index config and index was both None")
            self._trained = True
        else:
            log.warning("Calling fit() function, but model is trained. Ignoring.")

    @staticmethod
    def compute_coordinate_map(ids, coordinates):
        return {photo_id: coordinate for photo_id, coordinate in zip(ids, coordinates)}

    def fit_from_coord_map(self, coordinate_map):
        if not self._trained:
            self._coordinate_map = coordinate_map

            if self._index_config is not None and self._index_config.index_dir is not None:
                self._load_index()
            else:
                raise ValueError(f"Index config should be not None and have index directory provided: "
                                 f"{self._index_config}")

    def predict(self, data: np.ndarray) -> np.ndarray:
        log.debug(f"Searching {self.k} nearest neighbours...")
        dists, ids = self._index.search(data, self.k)
        coordinates_list = []
        for neighbours in ids:
            neighbour_coords = []
            for neighbour_id in neighbours:
                neighbour_coords.append(self._coordinate_map[neighbour_id])
            coordinates_list.append(neighbour_coords)
        coordinates = np.array(coordinates_list)
        return self._localise(coordinates, dists)

    def _localise(self, coordinates, dists) -> np.ndarray:
        if self.localisation_type == LocalisationType.NN:
            log.debug(f"Running {LocalisationType.NN.value} localisation")
            return coordinates[:, 0].squeeze()

        elif self.localisation_type == LocalisationType.KDE:
            log.debug(f"Running {LocalisationType.KDE.value} localisation")
            start = time.time()
            locations = self._kde_localisation(coordinates, dists)
            end = time.time()
            log.debug(f"Finished {LocalisationType.KDE.value} localisation in {end - start} seconds")
            return np.array(locations)

        elif self.localisation_type == LocalisationType.AVG or self.localisation_type == LocalisationType.WEIGHTED_AVG:
            if self.localisation_type == LocalisationType.WEIGHTED_AVG:
                log.debug(f"Running {LocalisationType.WEIGHTED_AVG.value} localisation")
                weights = self._get_weights(dists)
            else:
                log.debug(f"Running {LocalisationType.AVG.value} localisation")
                weights = None
            return _weighted_average(coordinates, weights)

        else:
            raise ValueError(f'Unknown localisation type {self.localisation_type}')

    def _get_weights(self, dist):
        if self._index.index_type == IndexType.L2_INDEX:
            weights = l2_similarity_weights(dist, self.m)
        elif self._index.index_type == IndexType.COSINE_INDEX:
            weights = cosine_similarity_weights(dist, self.m)
        else:
            raise ValueError(f'Unknown index type: {self._index.index_type}')
        return weights

    def _kde(self, nn_coordinates, weights):
        kde: NaiveKDE = NaiveKDE(kernel='gaussian', bw=self.sigma).fit(data=nn_coordinates, weights=weights)
        points = get_grid(nn_coordinates, self.sigma)
        y = kde.evaluate(points)
        return points[np.argmax(y)]

    def _kde_step(self, step_data):
        nn_coords, nn_dists = step_data
        weights = self._get_weights(nn_dists)
        return self._kde(nn_coords, weights)

    def _kde_localisation(self, coords, dists):
        zipped_data = zip(coords, dists)
        if self.num_workers == 0:
            locations = []
            for data in zipped_data:
                locations.append(self._kde_step(data))
        else:
            log.debug(f"Using {self.num_workers} number of workers to run kde")
            self._unload_index()
            with Pool(processes=self.num_workers) as pool:
                chunk_size = round(coords.shape[0] / self.num_workers)
                locations = [res for res in pool.imap(self._kde_step, zipped_data, chunk_size)]
            # self._load_index()

        return locations

    def _unload_index(self):
        log.debug(f"Unloading index, memory usage: {utils.get_memory_usage()}")
        del self._index.index
        log.debug(f"Index unloaded, memory usage: {utils.get_memory_usage()}")

    def _load_index(self):
        log.debug(f"Loading index, memory usage: {utils.get_memory_usage()}")
        del self._index
        self._index = None
        self._index = IndexBuilder(self._index_config).build()
        log.debug(f"Index loaded, memory usage: {utils.get_memory_usage()}")

    def __repr__(self):
        return f"LocalisationModel(localisation_type={self.localisation_type}, sigma={self.sigma}, m={self.m}, " \
               f"k={self.k})"
