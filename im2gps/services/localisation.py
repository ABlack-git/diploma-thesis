import logging
import numpy as np
import im2gps.lib.localisation as loc
import im2gps.lib.metric as metric
from im2gps.data.descriptors import DescriptorsTable
from im2gps.data.flickr_repo import FlickrPhoto

log = logging.getLogger(__name__)


def test_localization(path_to_db, test_q_path, k, gpu_enabled, gpu_id, loc_type, **kwargs):
    with DescriptorsTable(path_to_db, 2048) as db, \
            DescriptorsTable(test_q_path, 2048) as test_q:
        index = loc.build_index(db, gpu_enabled=gpu_enabled, gpu_id=gpu_id)
        return run_test(index, db, test_q, k, loc_type, **kwargs)


def run_test(index, db, queries, k, loc_type, **kwargs):
    queries = queries.get_descriptors_by_range(0, len(queries) - 1)
    queries_arr = np.array([q.descriptor for q in queries]).astype('float32')

    log.info(f"Finding {k} nearest neighbours of queries")
    dists, indices = loc.find_knn(queries_arr, index, k)
    knn_coords = __coords_from_indices(indices, db)

    log.info(f"Performing {loc_type} localization")
    localizations = loc.localise_by_knn(knn_coords, loc_type, dist=dists, **kwargs)
    q_true_loc = [[desc.lat, desc.lon] for desc in queries]
    geo_dist_error = metric.compute_geo_distance(q_true_loc, localizations)
    accuracy = metric.localization_accuracy(geo_dist_error)
    error = metric.avg_errors(geo_dist_error)
    return accuracy, error, geo_dist_error


def get_image_density_at_query_loc(file_path):
    with DescriptorsTable(file_path, 2048) as test_q:
        queries = test_q.get_descriptors_by_range(0, len(test_q) - 1)
        locations = [[desc.lon, desc.lat] for desc in queries]
        densities = [FlickrPhoto.count_photos_in_radius(coords, 1) for coords in locations]
    return densities


def __coords_from_indices(indices, db):
    knn_coords = []
    for knn_indices in indices:
        nn_coords = []
        for i in knn_indices:
            desc = db[i]
            nn_coords.append([desc.lat, desc.lon])
        knn_coords.append(nn_coords)
    return np.array(knn_coords)
