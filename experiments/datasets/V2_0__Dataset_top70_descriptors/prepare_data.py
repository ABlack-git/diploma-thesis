import json
import numpy as np
import os
import tqdm
from mongoengine import connect

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.core.index import IndexConfig, Index
from im2gps.services.index import get_index
from im2gps.core.metric import compute_geo_distance

PATH_TO_DS = "/home/andrew/Documents/study/thesis/thesis-src/experiments/datasets/V2_0__Dataset_top70_descriptors"


def prepare_data(dataset: DatasetEnum, index: Index):
    cursor = MongoDescriptor.objects(dataset=dataset)
    pbar = tqdm.tqdm(cursor, total=cursor.count(), desc=f"Processing {dataset} dataset")

    new_dataset = []
    dataset_info = {}
    for query in pbar:  # type: MongoDescriptor
        dists, ids = index.search(np.expand_dims(query.descriptor, axis=0), 70)
        dataset_object = {
            "query": query.photo_id,
            "neighbours": []
        }
        dataset_info[query.photo_id] = {
            "neighbours": [],
            "num_neighbours": 0
        }

        for neighbour_id, dist in zip(ids[0, :], dists[0, :]):
            dataset_object['neighbours'].append(int(neighbour_id))

            neighbour_doc: MongoDescriptor = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE,
                                                                     photo_id=neighbour_id).first()

            geo_dist = compute_geo_distance(np.array([query.coordinates]), np.array([neighbour_doc.coordinates]))

            dataset_info[query.photo_id]["neighbours"].append(
                {
                    "id": int(neighbour_id),
                    "geo_dist": float(geo_dist[0]),
                    "desc_dist": float(dist)
                }
            )
        new_dataset.append(dataset_object)

    return new_dataset, dataset_info


def save_file(obj, file_name):
    path = os.path.join(PATH_TO_DS, file_name)
    print(f"Saving file to {path}")
    with open(path, "w") as f:
        json.dump(obj, f)


def main():
    print("Building index")
    index_config = IndexConfig.from_path("/home/andrew/Documents/study/thesis/indices/l2_512_index")
    index = get_index(index_config)

    datasets = {
        DatasetEnum.VALIDATION_QUERY: "val_ds{}.json",
        DatasetEnum.TRAIN_QUERY: "train_ds{}.json"
    }

    for ds_type, filename in datasets.items():
        print(f"Starting to process {ds_type} dataset")
        dataset, dataset_info = prepare_data(ds_type, index)
        save_file(dataset, filename.format(""))
        save_file(dataset_info, filename.format("_info"))


if __name__ == '__main__':
    connect(db="im2gps", host="localhost", port=51998)
    main()
