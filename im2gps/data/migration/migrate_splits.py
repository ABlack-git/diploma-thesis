import json
import logging

from tqdm import tqdm
from im2gps.data.flickr_repo import FlickrPhoto

log = logging.getLogger(__name__)

collections_map = {
    "train": "flickr.db1",
    "train_q": "flickr.db1.train_q",
    "test_q": "flickr.db1.test_q",
    "val_q": "flickr.db1.val_q"
}


def migrate_splits(splits_file, collection_map=None):
    if collection_map is None:
        collection_map = collections_map
    with open(splits_file, 'r') as f:
        splits: dict = json.load(f)
    splits_bar = tqdm(splits.items(), position=1)

    for split_name, ids in splits.items():
        splits_bar.set_postfix({"split": split_name})
        for photo_id in tqdm(ids):
            photo = FlickrPhoto.objects.get(photo_id=photo_id)
            photo.switch_collection(collection_map[split_name], keep_created=False)
            photo.save()
