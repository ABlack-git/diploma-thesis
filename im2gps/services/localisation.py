import numpy as np
import im2gps.lib.localisation as loc
import im2gps.lib.metric as metric
from im2gps.data.descriptors import DescriptorsTable


def test_localization(path_to_db, test_q_path, k, gpu_enabled, gpu_id, loc_type):
    with DescriptorsTable(path_to_db, 2048) as db, \
            DescriptorsTable(test_q_path, 2048) as test_q:
        index = loc.build_index(db, gpu_enabled=gpu_enabled, gpu_id=gpu_id)
        queries = test_q.get_descriptors_by_range(0, len(test_q) - 1)
        queries_arr = np.array([q.descriptor for q in queries]).astype('float32')
        dists, indices = loc.find_knn(queries_arr, index, k)
        localizations = loc.localise_by_knn(indices, db, loc_type)
        q_true_loc = [(desc.lat, desc.lon) for desc in queries]
        geo_dist = metric.compute_geo_distance(q_true_loc, localizations)
        accuracy = metric.localization_accuracy(geo_dist)
        error = metric.avg_errors(geo_dist)
    return accuracy, error
