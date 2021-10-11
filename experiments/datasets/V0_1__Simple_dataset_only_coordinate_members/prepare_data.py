import json
import numpy as np
import os
import tqdm
from mongoengine import connect

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.core.metric import compute_geo_distance

PATH_TO_DS = "/home/andrew/Documents/study/thesis/thesis-src/experiments/datasets/V0_1__Simple_dataset_only_coordinate_members"


def prepare_data(dataset: DatasetEnum):
    cursor = MongoDescriptor.objects(dataset=dataset)
    pbar = tqdm.tqdm(cursor, total=cursor.count(), desc=f"Processing {dataset} dataset")

    new_dataset = []
    dataset_info = {}
    for query in pbar:  # type: MongoDescriptor

        dataset_object = {
            "query": query.photo_id,
            "neighbours": []
        }
        dataset_info[query.photo_id] = {
            "neighbours": [],
            "num_neighbours": 0
        }

        neighbours = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE, coords__near=query.coordinates)[:50]
        for neighbour in neighbours:  # type: MongoDescriptor
            dataset_object['neighbours'].append(neighbour.photo_id)
            geo_dist = compute_geo_distance(np.array([query.coordinates]), np.array([neighbour.coordinates]))
            dist = np.linalg.norm(query.descriptor - neighbour.descriptor)

            dataset_info[query.photo_id]["neighbours"].append(
                {
                    "id": neighbour.photo_id,
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

    datasets = {
        DatasetEnum.VALIDATION_QUERY: "val_ds{}.json",
        DatasetEnum.TRAIN_QUERY: "train_ds{}.json"
    }

    for ds_type, filename in datasets.items():
        print(f"Starting to process {ds_type} dataset")
        dataset, dataset_info = prepare_data(ds_type)
        save_file(dataset, filename.format(""))
        save_file(dataset_info, filename.format("_info"))


if __name__ == '__main__':
    connect(db="im2gps", host="localhost", port=51998)
    main()
