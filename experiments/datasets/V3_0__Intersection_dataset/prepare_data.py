import json
import numpy as np
import os
import tqdm
from mongoengine import connect

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.core.index import IndexConfig, Index
from im2gps.services.index import get_index
from im2gps.core.metric import compute_geo_distance

PATH_TO_DS = "/home/andrew/Documents/study/thesis/thesis-src/experiments/datasets/V3_0__Intersection_dataset"


def prepare_data(dataset: DatasetEnum, index: Index):
    cursor = MongoDescriptor.objects(dataset=dataset)
    pbar = tqdm.tqdm(cursor, total=cursor.count(), desc=f"Processing {dataset} dataset")

    new_dataset = []
    dataset_info = {}
    count = 0
    for query in pbar:  # type: MongoDescriptor
        dataset_object = {
            "query": query.photo_id,
            "neighbours": [],
            "target_id": -1
        }
        dataset_info[query.photo_id] = {
            "neighbours": [],
            "num_neighbours": 0,
            "target_id": -1
        }

        d_dists, d_neighbours_ids = index.search(np.expand_dims(query.descriptor, axis=0), 100)
        c_neighbours = [desc_doc for desc_doc in MongoDescriptor.objects(dataset=DatasetEnum.DATABASE,
                                                                         coords__near=query.coordinates)[:500]]

        n_descriptors = np.array([desc_doc.descriptor for desc_doc in c_neighbours])

        c_neighbours_ids_set = {neighbour.photo_id for neighbour in c_neighbours}
        d_neighbours_ids_set = set(d_neighbours_ids[0])

        intersection = d_neighbours_ids_set.intersection(c_neighbours_ids_set)

        desc_dists = np.linalg.norm(query.descriptor - n_descriptors, axis=1)

        candidates = []

        if len(intersection) == 0:
            # find closest in d-space among c_neighbours
            target_neighbour_idx = np.argmin(desc_dists)
            target_neighbour = c_neighbours[target_neighbour_idx].photo_id
            candidates.append(target_neighbour)
            candidates.extend(d_neighbours_ids[0][:50].tolist())

            count = 0
            for neighbour in c_neighbours:
                if neighbour.photo_id == target_neighbour:
                    continue
                candidates.append(neighbour.photo_id)
                count += 1
                if count == 49:
                    break
        else:
            # find closest in c-space among intersection
            candidates.extend(list(intersection))
            intersection_length = len(intersection)
            max_neighbours = 100 - intersection_length
            max_d_neighbours = round(max_neighbours / 2)
            max_c_neighbours = max_neighbours - max_d_neighbours

            count = 0
            for d_id in d_neighbours_ids[0].tolist():
                if d_id in intersection:
                    continue
                candidates.append(d_id)
                count += 1
                if count == max_d_neighbours:
                    break

            target_neighbour = -1
            for n in c_neighbours:
                if n.photo_id in intersection:
                    target_neighbour = n.photo_id
                    break
            count = 0
            for neighbour in c_neighbours:
                if neighbour.photo_id in intersection:
                    continue
                candidates.append(neighbour.photo_id)
                count += 1
                if count == max_c_neighbours:
                    break
        dataset_object['target_id'] = target_neighbour
        for neighbour_id in candidates:
            dataset_object['neighbours'].append(int(neighbour_id))

            # add info
            neighbour_doc: MongoDescriptor = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE,
                                                                     photo_id=neighbour_id).first()
            geo_dist = compute_geo_distance(np.array([query.coordinates]), np.array([neighbour_doc.coordinates]))
            desc_dist = np.linalg.norm(query.descriptor - neighbour_doc.descriptor)
            dataset_info[query.photo_id]["neighbours"].append(
                {
                    "id": int(neighbour_id),
                    "geo_dist": float(geo_dist[0]),
                    "desc_dist": float(desc_dist)
                }
            )

        new_dataset.append(dataset_object)
    print(count)
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
