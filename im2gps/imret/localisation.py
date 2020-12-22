import torch
import logging
import numpy as np
import typing as tp
from im2gps.data.descriptors import DescriptorsTable

log = logging.getLogger(__name__)


def __compute_distance_numpy(query: np.ndarray, base: np.ndarray) -> np.ndarray:
    return np.linalg.norm(query - base, axis=2).squeeze()


def __compute_distance_torch(query: np.ndarray, base: np.ndarray, device: torch.device) -> np.ndarray:
    t_query = torch.from_numpy(query).float().to(device)
    t_base = torch.from_numpy(base).float().to(device)
    norm = torch.norm(t_query - t_base, dim=2).squeeze()
    if norm.device.type == 'cuda':
        return norm.cpu().numpy()
    else:
        return norm.numpy()


def _compute_distances(query: np.ndarray, base: np.ndarray, use_torch: bool = True,
                       device: torch.device = None) -> np.ndarray:
    """
    Computes distances between query and base.
    :param query: array of shape D or NxD, where N is number of queries, D is dimension of descriptor vector
    :type query: np.ndarray
    :param base: array of shape D or MxD, where M is number of descriptors, D is the dimension of descriptors
    :type base: np.ndarray
    :param use_torch: True/False, determines whether to use PyTorch for computations
    :type use_torch: bool
    :param device: torch device, use together with use_torch parameter
    :type device: torch.device
    :return: returns NxM array of distances. If N or M equals 1 singelton dimension will be removed.
    :rtype: np.ndarray
    """
    assert query.ndim < 3, f"Query array shape was {query.shape}, but required is NxD"
    assert base.ndim < 3, f"Base array shape was {base.shape}, but required is MxD"
    if query.ndim == 1:
        q_n = 1
        q_d, = query.shape
    else:
        q_n, q_d = query.shape

    if base.ndim == 1:
        b_m = 1
        b_d, = base.shape
    else:
        b_m, b_d = base.shape

    if use_torch:
        dev = torch.device('cpu') if device is None else device
        return __compute_distance_torch(query.reshape((q_n, 1, q_d)), base.reshape((b_m, b_d)), dev)
    else:
        return __compute_distance_numpy(query.reshape((q_n, 1, q_d)), base.reshape((b_m, b_d)))


def __batch_range(size, step):
    start = 0
    end = 0
    while end < size - 1:
        end = start + step - 1
        if end > size - 1:
            end = size - 1
        yield start, end
        start = end + 1


def _localize_by_1nn(queries: tp.Union[np.ndarray, DescriptorsTable], database: DescriptorsTable,
                     query_batch_size=100, db_batch_size=1000, use_torch: bool = True,
                     device: torch.device = None) -> np.ndarray:
    """

    :param queries:
    :type queries:
    :param database:
    :type database:
    :return: numpy array of size N, where each element is tuple of longitude and latitude, in that order
    :rtype: np.ndarray
    """

    if isinstance(queries, np.ndarray):
        num_queries = queries.shape[0]
    elif isinstance(queries, DescriptorsTable):
        num_queries = len(queries)
    else:
        raise ValueError(f"queries should be of type either np.ndarray or DescriptorTable, was {type(queries)}")

    distances = np.full((num_queries,), np.inf)
    localisation = np.full((num_queries,), np.nan, dtype=[('longitude', 'f8'), ('latitude', 'f8')])
    for q_batch_start, q_batch_end in __batch_range(num_queries, query_batch_size):
        if isinstance(queries, np.ndarray):
            q_batch = queries[q_batch_start:q_batch_end + 1]
        else:
            q_batch = queries.get_descriptors_by_range(q_batch_start, q_batch_end + 1, field='descriptor')

        for d_batch_start, d_batch_end in __batch_range(len(database), db_batch_size):
            batch_descriptors = database.get_descriptors_by_range(d_batch_start, d_batch_end + 1)
            d_batch = np.array([desc.descriptor for desc in batch_descriptors])
            batch_distances = _compute_distances(q_batch, d_batch, use_torch, device)
            # get indices of database descriptor with min distance from query
            min_indices = np.argmin(batch_distances, axis=1)
            # get min distance values
            min_vals = np.take_along_axis(batch_distances, np.expand_dims(min_indices, axis=1), axis=1).squeeze()

            # check if current distance of queries is greater than min values
            dist_batch_slice = distances[q_batch_start:q_batch_end + 1]
            dist_condition = dist_batch_slice > min_vals

            # update locations where current distance of queries is greater than min values

            desc_locations = np.array([(batch_descriptors[i].lon, batch_descriptors[i].lat) for i in min_indices],
                                      dtype=[('longitude', 'f8'), ('latitude', 'f8')])
            loc_batch_slice = localisation[q_batch_start:q_batch_end + 1]
            loc_batch_slice[:] = np.where(dist_condition, desc_locations, loc_batch_slice)

            # update distance where current distance of queries is greater than min values
            dist_batch_slice[:] = np.where(dist_condition, min_vals, dist_batch_slice)

    return localisation


def _kde():
    pass


def localise_knn_kde(queries: tp.Union[np.ndarray, DescriptorsTable], database: DescriptorsTable,
                     k, sigma, m, query_batch_size=100, db_batch_size=1000, use_torch: bool = True,
                     device: torch.device = None):
    if isinstance(queries, np.ndarray):
        num_queries = queries.shape[0]
    elif isinstance(queries, DescriptorsTable):
        num_queries = len(queries)
    else:
        raise ValueError(f"queries should be of type either np.ndarray or DescriptorTable, was {type(queries)}")
    distances = np.full((num_queries, k), np.inf)
    localisation = np.full((num_queries,), np.nan, dtype=[('longitude', 'f8'), ('latitude', 'f8')])
    for q_batch_start, q_batch_end in __batch_range(num_queries, query_batch_size):
        if isinstance(queries, np.ndarray):
            q_batch = queries[q_batch_start:q_batch_end + 1]
        else:
            q_batch = queries.get_descriptors_by_range(q_batch_start, q_batch_end + 1, field='descriptor')

        for d_batch_start, d_batch_end in __batch_range(len(database), db_batch_size):
            batch_descriptors = database.get_descriptors_by_range(d_batch_start, d_batch_end + 1)
            d_batch = np.array([desc.descriptor for desc in batch_descriptors])
            batch_distances = _compute_distances(q_batch, d_batch, use_torch, device)
            # for each query sort distances and return original indices (only k first indices)
            k_min_indices = np.argsort(batch_distances, axis=1)[:, :k]
            k_min_distances = np.take_along_axis(batch_distances, k_min_indices, axis=1)
            # get indices of database descriptor with min distance from query
            min_indices = np.argmin(batch_distances, axis=1)
            # get min distance values
            min_vals = np.take_along_axis(batch_distances, np.expand_dims(min_indices, axis=1), axis=1).squeeze()

            # check if current distance of queries is greater than min values
            dist_batch_slice = distances[q_batch_start:q_batch_end + 1]
            dist_condition = dist_batch_slice > min_vals

            # update locations where current distance of queries is greater than min values

            desc_locations = np.array([(batch_descriptors[i].lon, batch_descriptors[i].lat) for i in min_indices],
                                      dtype=[('longitude', 'f8'), ('latitude', 'f8')])
            loc_batch_slice = localisation[q_batch_start:q_batch_end + 1]
            loc_batch_slice[:] = np.where(dist_condition, desc_locations, loc_batch_slice)

            # update distance where current distance of queries is greater than min values
            dist_batch_slice[:] = np.where(dist_condition, min_vals, dist_batch_slice)

    return localisation


# q1 = np.array([1, 2, 3])
# q2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# d = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
# diff_numpy = _compute_distances(q2, d, use_torch=True)
# print(diff_numpy)

# def test_numpy(q_size, b_size):
#     queries = np.random.random((q_size, 2048))
#     base = np.random.random((b_size, 2048))
#     norms = _compute_distances(queries, base)
#
#
# def test_torch(q_size, b_size):
#     queries = np.random.random((q_size, 2048))
#     base = np.random.random((b_size, 2048))
#     norms = _compute_distance_torch(queries, base)
#
#
# import timeit
# import matplotlib.pyplot as plt
#
# x = []
# y = []
# z = []
# for i in range(100, 2100, 100):
#     x.append(i)
#     y.append(timeit.timeit(f"test_numpy({i}, 10)", setup="from __main__ import test_numpy", number=10) / 10)
#
# for i in range(100, 2100, 100):
#     z.append(timeit.timeit(f"test_torch({i}, 10)", setup="from __main__ import test_torch", number=10) / 10)
#
# plt.plot(x, y)
# plt.plot(x, z)
# plt.show()
from im2gps.imret.metric import localization_accuracy

test_file = '/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_test_q.h5'
val_file = '/Users/zakharca/Documents/Study/thesis/descriptors/512_flickr_descriptor_val_q.h5'
with DescriptorsTable(test_file, 2048) as test_descriptors, \
        DescriptorsTable(val_file, 2048) as val_descriptors:
    localizations = _localize_by_1nn(test_descriptors, val_descriptors, use_torch=True, device=torch.device('cuda:1'),
                                     query_batch_size=100, db_batch_size=10000)
    # localizations = _localize_by_1nn(test_descriptors, val_descriptors, use_torch=False, db_batch_size=100)
    gt_locs = [(desc.lon, desc.lat) for desc in test_descriptors.iterrows()]
    accuracy = localization_accuracy(gt_locs, localizations.tolist())
    print(accuracy)
