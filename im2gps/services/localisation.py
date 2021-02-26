import logging
import numpy as np
import typing as t
import im2gps.lib.localisation as loc
import im2gps.lib.metric as metric
from im2gps.lib.index import IndexBuilder, Index, IndexType
from im2gps.data.descriptors import DescriptorsTable, Descriptor
from im2gps.data.flickr_repo import FlickrPhoto

log = logging.getLogger(__name__)


def test_localization(path_to_db, test_q_path, k, gpu_enabled, gpu_id, loc_type, index_type: IndexType, **kwargs):
    with DescriptorsTable(path_to_db, 2048) as db, \
            DescriptorsTable(test_q_path, 2048) as test_q:
        index = IndexBuilder(db, index_type=index_type, gpu_enabled=gpu_enabled, gpu_id=gpu_id).build()
        queries = test_q.get_descriptors_by_range(0, len(test_q))
        queries_array = np.array([q.descriptor for q in test_q]).astype('float32')
        locations = localise(index, db, queries_array, k, loc_type, index_type, **kwargs)
        return compute_statistics(queries, locations)


def get_image_density_at_query_loc(file_path):
    with DescriptorsTable(file_path, 2048) as test_q:
        queries = test_q.get_descriptors_by_range(0, len(test_q) - 1)
        locations = [[desc.lon, desc.lat] for desc in queries]
        densities = [FlickrPhoto.count_photos_in_radius(coords, 1) for coords in locations]
    return densities


def localise(index: Index, db: DescriptorsTable, queries: np.ndarray, k, loc_type, index_type, **kwargs):
    log.info(f"Finding {k} nearest neighbours of queries")
    preprocessed = kwargs['preprocessed'] if 'preprocessed' in kwargs else False
    dists, indices = index.search(queries, k, preprocessed=preprocessed)
    knn_coords = indices_to_coords(indices, db)
    localizations = loc.localise_by_knn(knn_coords, loc_type, index_type, dist=dists, **kwargs)
    return localizations


def compute_statistics(queries: t.List[Descriptor], locations):
    q_true_loc = [[desc.lat, desc.lon] for desc in queries]
    geo_dist_error = metric.compute_geo_distance(q_true_loc, locations)
    accuracy = metric.localization_accuracy(geo_dist_error)
    error = metric.avg_errors(geo_dist_error)
    return accuracy, error, geo_dist_error


def indices_to_coords(indices, db: DescriptorsTable):
    knn_coords = []
    for knn_indices in indices:
        nn_coords = []
        for i in knn_indices:
            desc = db[i]
            nn_coords.append([desc.lat, desc.lon])
        knn_coords.append(nn_coords)
    return np.array(knn_coords)
