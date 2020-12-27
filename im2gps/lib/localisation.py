import logging
import numpy as np
import typing as tp
import faiss
from scipy.stats import multivariate_normal
from im2gps.data.descriptors import DescriptorsTable

log = logging.getLogger(__name__)


def __batch_range(size, step):
    start = 0
    end = 0
    while end < size - 1:
        end = start + step - 1
        if end > size - 1:
            end = size - 1
        yield start, end
        start = end + 1


def _kde(nn_coordinates: np.ndarray, distances: np.ndarray, sigma, m) -> tp.List[float]:
    weights = (1 / distances) ** m
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


def build_index(data: DescriptorsTable, batch_size=50000, gpu_enabled=False, gpu_id=0) -> faiss.IndexFlatL2:
    log.debug("Creating flatL2 index")
    index = faiss.IndexFlatL2(data.desc_shape)
    if gpu_enabled:
        log.debug(f"Creating GPU index with gpu_id={gpu_id}")
        resource = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resource, gpu_id, index)

    log.debug("Adding data to index...")
    for start, end in __batch_range(len(data), batch_size):
        batch = data.get_descriptors_by_range(start, end + 1, field='descriptor').astype('float32')
        index.add(batch)
    log.debug(f"Index created, number of vectors in index {index.ntotal}")
    return index


def find_knn(queries: np.ndarray, index: faiss.IndexFlatL2, k: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    return index.search(queries, k)


def localise_by_knn(coordinates: np.ndarray, loc_type, dist=None, **kwargs) -> tp.List[tp.List[float]]:
    locations = []
    if loc_type == '1nn':
        locations = coordinates.squeeze().tolist()
    elif loc_type == 'kde':
        assert 'sigma' in kwargs and 'm' in kwargs, "sigma and m should be be provided for kde"
        assert dist is not None, "dist should not be none"
        assert dist.shape == coordinates.shape[:2], f"shapes of dist and indices are different"
        for nn_coords, knn_dists in zip(coordinates, dist):
            loc = _kde(np.array(nn_coords), knn_dists, kwargs['sigma'], kwargs['m'])
            locations.append(loc)
    elif loc_type == 'avg':
        assert 'avg_type' in kwargs, 'avg_type should be provided'
        if kwargs['avg_type'] == 'weighted':
            weights = (1 / dist)
        elif kwargs['avg_type'] == 'regular':
            weights = None
        else:
            raise ValueError(f'Unknown type of avg_type, {kwargs["avg_type"]}')
        locations = _weighted_average(np.array(coordinates), weights=weights).tolist()
    return locations
