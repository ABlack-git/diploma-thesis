import hydra
import json
import logging
import numpy as np
from mongoengine import connect

from im2gps.data.flickr_repo import FlickrPhoto
from im2gps.conf.config import Config

log = logging.getLogger(__name__)


def _count_frequencies(data: dict):
    owners = []
    frequencies = []
    total_count = sum(data.values())
    for k, v in data.items():
        owners.append(k)
        frequencies.append(float(v) / float(total_count))
    return owners, frequencies


def _sample(data: dict, total_count, threshold):
    size = 0
    authors = []
    owners, frequencies = _count_frequencies(data)
    while data and size < total_count * threshold:
        owner_sample = np.random.choice(owners, 1, p=frequencies, replace=False)[0]
        authors.append(owner_sample)
        size = size + data[owner_sample]
        data.pop(owner_sample)
        owners, frequencies = _count_frequencies(data)
    return authors


def _split():
    pipeline = [
        {"$group": {"_id": "$owner_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    owners_cnt = {doc['_id']: doc['count'] for doc in FlickrPhoto.objects.aggregate(pipeline)}
    total_count = sum(owners_cnt.values())
    train_authors = _sample(owners_cnt, total_count, 0.95)
    train = [photo.photo_id for photo in FlickrPhoto.objects(owner_name__in=train_authors).only('photo_id')]
    train_q_authors = _sample(owners_cnt, total_count, 0.03)
    train_q = [photo.photo_id for photo in FlickrPhoto.objects(owner_name__in=train_q_authors).only('photo_id')]
    test_q_authors = _sample(owners_cnt, total_count, 0.01)
    test_q = [photo.photo_id for photo in FlickrPhoto.objects(owner_name__in=test_q_authors).only('photo_id')]
    val_q_authors = _sample(owners_cnt, total_count, 0.01)
    val_q = [photo.photo_id for photo in FlickrPhoto.objects(owner_name__in=val_q_authors).only('photo_id')]
    datasets = {'train': train, 'train_q': train_q, 'test_q': test_q, 'val_q': val_q}
    log.info(f"train size: {len(train)}, train_q size: {len(train_q)}, test_q size: {len(test_q)}, "
             f"val_q size: {len(val_q)}")
    with open('datasets.json', 'w') as f:
        json.dump(datasets, f)


@hydra.main(config_path='../../conf', config_name='config')
def main(cfg: Config):
    connect(db=cfg.data.db.database, host=cfg.data.db.host, port=cfg.data.db.port)
    _split()


if __name__ == '__main__':
    main()
