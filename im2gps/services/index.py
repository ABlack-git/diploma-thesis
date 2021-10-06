import os
import pickle
import logging

from im2gps.data.descriptors import MongoDescriptor, DatasetEnum
from im2gps.core.index import IndexConfig, IndexBuilder, INDEX_CLASS_FILE, INDEX_FILE_NAME, Index

log = logging.getLogger(__name__)


def create_and_save_index(index_config: IndexConfig):
    log.info("Reading data from db...")
    ids, coordinates, descriptors = MongoDescriptor.get_data_as_arrays(DatasetEnum.DATABASE)
    log.info("Finished reading data from db")

    log.info("Building index...")
    index = IndexBuilder(index_config, descriptors, ids).build()
    log.info(f"Index built: {repr(index)}")

    class_path = os.path.join(index_config.index_dir, INDEX_CLASS_FILE)
    log.info(f"Saving class file to {class_path}")
    with open(class_path, "wb") as f:
        pickle.dump(index_config, f)

    index_path = os.path.join(index_config.index_dir, INDEX_FILE_NAME)
    log.info(f"Saving index file to {index_path}")
    index.write_index(index_path)


def get_index(index_config: IndexConfig) -> Index:
    if index_config.index_dir is None:
        log.info("Getting training data")
        ids, coordinates, descriptors = MongoDescriptor.get_data_as_arrays(DatasetEnum.DATABASE)
        log.info("Finished getting training data")
        index = IndexBuilder(index_config, descriptors, ids).build()
    else:
        index = IndexBuilder(index_config).build()

    return index
