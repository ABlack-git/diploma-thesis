import json
import os
import pandas as pd
from tqdm import tqdm
from dataclasses import make_dataclass, dataclass
from mongoengine.context_managers import switch_collection

from im2gps.data.flickr_repo import FlickrPhoto
from im2gps.data.descriptors import DatasetEnum


def get_image_density_at_query_loc(output_path, start_from, save_every=25000):
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
        coords = photo.geo.coordinates
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


def density_near_query(output_path, dataset: DatasetEnum, ids_file):
    with open(ids_file, 'r') as f:
        ids = json.load(f)

    ds_ids = ids[dataset.value]

    density_records = []
    for photo_id in tqdm(ds_ids):
        q_photo = FlickrPhoto.objects(photo_id=photo_id).first()

        coords = q_photo.geo.coordinates
        thresholds = {
            "density_10m": 0.01,
            "density_100m": 0.1,
            "density_500m": 0.5,
            "density_1km": 1,
        }
        DensityRecord = make_dataclass("DensityRecord", [("photo_id", int),
                                                         ("density_10m", int),
                                                         ("density_100m", int),
                                                         ("density_500m", int),
                                                         ("density_1km", int)])
        densities = dict()
        with switch_collection(FlickrPhoto, "flickr.db1") as DbFlickrPhoto:
            for density_type, radius in thresholds.items():
                density = DbFlickrPhoto.count_photos_in_radius(coords, radius)
                densities[density_type] = density
        density_records.append(DensityRecord(photo_id, **densities))

    df = pd.DataFrame(density_records)
    df.to_csv(output_path, index=False)
