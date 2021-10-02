import os
import json
import numpy as np
import tqdm
from mongoengine import connect

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.core.index import IndexConfig
from im2gps.services.index import get_index
from im2gps.core.metric import compute_geo_distance

PATH_TO_DS = "/home/andrew/Documents/study/thesis/thesis-src/experiments/datasets/V1_0__Simple_dataset"

connect(db="im2gps", host="localhost", port=51998)

print("Getting queries")
train_cursor = MongoDescriptor.objects(dataset=DatasetEnum.VALIDATION_QUERY)
train_count = train_cursor.count()

print("Building index")
index_config = IndexConfig.from_path("/home/andrew/Documents/study/thesis/indices/l2_512_index")
index = get_index(index_config)

train_dataset = []
train_info = {}

pbar = tqdm.tqdm(train_cursor, total=train_count, desc="Processing train query")
for i, train_query in enumerate(pbar):
    neighbours = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE, coords__near=train_query.coordinates)[:25]

    dists, ids = index.search(np.expand_dims(train_query.descriptor, axis=0), 25)
    neighbour_ids = set()
    train_object = {
        "query": train_query.photo_id,
        "neighbours": []
    }

    train_info[train_query.photo_id] = {"neighbours": [],
                                        "num_neighbours": 0}
    for neighbour in neighbours:
        train_object['neighbours'].append(neighbour.photo_id)
        neighbour_ids.add(neighbour.photo_id)
        geo_dist = compute_geo_distance(np.array([train_query.coordinates]), np.array([neighbour.coordinates]))
        desc_dist = np.linalg.norm(train_query.descriptor - neighbour.descriptor)

        train_info[train_query.photo_id]['neighbours'].append({
            "id": neighbour.photo_id,
            "geo_dist": float(geo_dist[0]),
            "desc_dist": float(desc_dist)
        })

    count = 0
    for n_id, desc_dist in zip(ids[0, :], dists[0, :]):
        if n_id not in neighbour_ids:
            train_object['neighbours'].append(int(n_id))
            neighbour_ids.add(int(n_id))
            count += 1

            desc_doc = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE, photo_id=n_id).first()
            geo_dist = compute_geo_distance(np.array([train_query.coordinates]), np.array([desc_doc.coordinates]))
            train_info[train_query.photo_id]['neighbours'].append({
                "id": int(n_id),
                "geo_dist": float(geo_dist[0]),
                "desc_dist": float(desc_dist)
            })

    if count < 25:
        added_count = 0
        random_docs = MongoDescriptor.objects(dataset=DatasetEnum.DATABASE, coords__near=train_query.coordinates)[25:50]
        for doc in random_docs:
            if added_count >= (25 - count):
                del random_docs
                break
            if doc.photo_id in neighbour_ids:
                continue
            train_object['neighbours'].append(doc.photo_id)
            neighbour_ids.add(doc.photo_id)
            geo_dist = compute_geo_distance(np.array([train_query.coordinates]), np.array([doc.coordinates]))
            desc_dist = np.linalg.norm(train_query.descriptor - doc.descriptor)

            train_info[train_query.photo_id]['neighbours'].append({
                "id": doc.photo_id,
                "geo_dist": float(geo_dist[0]),
                "desc_dist": float(desc_dist)
            })
            added_count += 1

    train_info[train_query.photo_id]["num_neighbours"] = len(train_info[train_query.photo_id]["neighbours"])
    train_dataset.append(train_object)

TRAIN_DS = "train_ds.json"
VAL_DS = "val_ds.json"
TRAIN_DS_INFO = "train_ds_info.json"
VAL_DS_INFO = "val_ds_info.json"
# print("pickling dataset")
# with open(os.path.join(PATH_TO_DS, "val_ds.pickle"), "wb") as f:
#     pickle.dump(train_dataset, f)
# print("pickling dataset info")
# with open(os.path.join(PATH_TO_DS, "val_ds_info.pickle"), "wb") as f:
#     pickle.dump(train_info, f)
print("saving dataset to json")
with open(os.path.join(PATH_TO_DS, VAL_DS), "w") as f:
    json.dump(train_dataset, f)
print("saving dataset info to json")
with open(os.path.join(PATH_TO_DS, VAL_DS_INFO), "w") as f:
    json.dump(train_info, f)
