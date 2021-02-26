import logging
import numpy as np
import im2gps.lib.localisation as loc
import im2gps.lib.metric as metric
from im2gps.lib.index import IndexBuilder, Index, IndexType
from im2gps.data.descriptors import DescriptorsTable
from im2gps.data.flickr_repo import FlickrPhoto

log = logging.getLogger(__name__)


def test_localization(path_to_db, test_q_path, k, gpu_enabled, gpu_id, loc_type, index_type: IndexType, **kwargs):
    with DescriptorsTable(path_to_db, 2048) as db, \
            DescriptorsTable(test_q_path, 2048) as test_q:
        index = IndexBuilder(db, index_type=index_type, gpu_enabled=gpu_enabled, gpu_id=gpu_id).build()
        db_coords = np.array([[desc.lat, desc.lon] for desc in db])
        q_coords = np.array([[desc.lat, desc.lon] for desc in test_q])
        queries_array = np.array([q.descriptor for q in test_q]).astype('float32')
        locations = localise(index, db_coords, queries_array, k, loc_type, index_type, **kwargs)
        return compute_statistics(q_coords, locations)


def get_image_density_at_query_loc(output_path, start_from, save_every=25000):
    import pandas as pd
    import os
    from tqdm import tqdm
    from dataclasses import make_dataclass

    file_count = (start_from + 1) // save_every
    base_name = os.path.basename(output_path)
    dir_name = os.path.dirname(output_path)
    file_path = os.path.join(dir_name, base_name.split(".")[0] + "-{}." + base_name.split(".")[1])
    Density = make_dataclass("Density",
                             [("photo_id", int), ("density_10m", int), ("density_100m", int), ("density_500m", int)])
    densities = []
    pbar = tqdm(total=FlickrPhoto.objects.count(), position=1, initial=start_from)
    pbar_logger = tqdm(total=0, position=2, bar_format='{desc}')
    pbar_logger.set_description_str(f"Processing batch number {file_count}. "
                                    f"Output file {file_path.format(file_count)}")

    cursor = FlickrPhoto.objects.order_by('date_upload') \
        .skip(start_from).only('photo_id', 'geo.coords').timeout(False)

    for i, photo in enumerate(cursor):
        coords = photo.geo.coords['coordinates']
        density_10m = FlickrPhoto.count_photos_in_radius(coords, 0.01)
        density_100m = FlickrPhoto.count_photos_in_radius(coords, 0.1)
        density_500m = FlickrPhoto.count_photos_in_radius(coords, 0.5)
        densities.append(Density(photo.photo_id, density_10m, density_100m, density_500m))
        if (i + 1) % save_every == 0:
            pd.DataFrame(densities).to_csv(file_path.format(file_count))
            densities = []
            file_count += 1
            pbar_logger.set_description_str(f"Processing batch number {file_count}. "
                                            f"Output file {file_path.format(file_count)}")
        pbar.update(1)

    pbar.close()
    del cursor

    pd.DataFrame(densities).to_csv(file_path.format(file_count))
    pbar_logger.set_description_str("Finished last batch")


def localise(index: Index, db_coords: np.ndarray, queries: np.ndarray, k, loc_type, index_type: IndexType, **kwargs):
    log.info(f"Finding {k} nearest neighbours of queries")
    preprocessed = kwargs['preprocessed'] if 'preprocessed' in kwargs else False
    dists, indices = index.search(queries, k, preprocessed=preprocessed)
    knn_coords = indices_to_coords(indices, db_coords)
    localizations = loc.localise_by_knn(knn_coords, loc_type, index_type, dist=dists, **kwargs)
    return localizations


def compute_statistics(q_coords: np.ndarray, locations: np.ndarray):
    geo_dist_error = metric.compute_geo_distance(q_coords, locations)
    accuracy = metric.localization_accuracy(geo_dist_error)
    error = metric.avg_errors(geo_dist_error)
    return accuracy, error, geo_dist_error


def indices_to_coords(indices, db: np.ndarray):
    return db[indices]
