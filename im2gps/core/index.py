import os
import faiss
import logging
import numpy as np
import pickle
from enum import Enum
from dataclasses import dataclass
from typing import Union

log = logging.getLogger(__name__)

INDEX_FILE_NAME = "index.if"
INDEX_CLASS_FILE = "index.class"


class IndexType(Enum):
    L2_INDEX = "L2"
    COSINE_INDEX = "cosine"


@dataclass
class IndexConfig:
    index_dir: str = None
    index_type: IndexType = IndexType.L2_INDEX
    gpu_enabled: bool = False
    gpu_id: int = -1

    @classmethod
    def from_path(cls, path):
        with open(os.path.join(path, INDEX_CLASS_FILE), 'rb') as f:
            index_config = pickle.load(f)

        index_config.index_dir = path
        return index_config


class Index:
    def __init__(self, index, index_type, gpu_enabled, gpu_id):
        self.index: faiss.Index = index
        self.index_type: IndexType = index_type
        self.gpu_enabled = gpu_enabled
        self.gpu_id = gpu_id

    def search(self, queries: np.ndarray, k: int, preprocessed=False):
        if queries.dtype != "float32":
            queries = queries.astype("float32")
        if self.index_type == IndexType.COSINE_INDEX and not preprocessed:
            faiss.normalize_L2(queries)
            dist, ind = self.index.search(queries, k)
        else:
            dist, ind = self.index.search(queries, k)
        return dist, ind

    def add_with_ids(self, data, ids):
        if data.dtype != "float32":
            data = data.astype("float32")
        if self.index_type is IndexType.COSINE_INDEX:
            faiss.normalize_L2(data)
        self.index.add_with_ids(data, ids)

    def write_index(self, path):
        faiss.write_index(self.index, path)

    def __repr__(self):
        return f"Index(index_type={self.index_type}, gpu_enabled={self.gpu_enabled}, gpu_id={self.gpu_id}, " \
               f"index={repr(self.index)})"


class IndexBuilder:
    def __init__(self, index_config: IndexConfig, data: Union[np.ndarray, None] = None,
                 ids: Union[np.ndarray, None] = None,
                 index_dimension=None):
        self.index_type: IndexType = index_config.index_type
        self.gpu_enabled: bool = index_config.gpu_enabled
        self.gpu_id: int = index_config.gpu_id
        self.index_dir = index_config.index_dir
        self.data: Union[np.ndarray, None] = data
        self.ids: Union[np.ndarray, None] = ids
        self.__index: faiss.Index = None
        self.__index_dimension = index_dimension
        self.__load_index = False
        self.__build_empty_index = False

    def __validate_params(self):
        assert self.index_type in IndexType, f"Unknown index type: {self.index_type}"
        if self.gpu_enabled:
            assert self.gpu_id >= 0, f"gpu_id should be provided"

        if self.index_dir is not None:
            assert os.path.isdir(self.index_dir), f"Index directory should " \
                                                  f"point to existing directory. {self.index_dir}"

            assert os.path.isfile(os.path.join(self.index_dir, INDEX_FILE_NAME)), \
                f"{INDEX_FILE_NAME} should exist under provided directory {self.index_dir}"
            self.__load_index = True
        elif self.data is None and self.ids is None:
            assert self.__index_dimension is not None, "Index dimension should be provided, when data is not"
            assert isinstance(self.__index_dimension, int), f"Index dimension should be an int, " \
                                                            f"was {type(self.__index_dimension)}"
            self.__build_empty_index = True
        else:
            assert isinstance(self.data, np.ndarray), "data should be instance of np.ndarray"
            assert isinstance(self.ids, np.ndarray), "ids should be instance of np.ndarray"
            assert len(self.data) == len(self.ids), "number of data vectors and ids should be the same"

    def __load_index_from_disk(self):
        with open(os.path.join(self.index_dir, INDEX_CLASS_FILE), 'rb') as f:
            index_class = pickle.load(f)
            if index_class.index_type == self.index_type:
                pass
            else:
                if self.index_type is None:
                    log.debug(f"Index type {index_class.index_type} will be loaded from disk")
                else:
                    log.warn(f"{self.index_type} index type was provided, but {index_class.index_type} index type "
                             f"will be loaded from disk")

                self.index_type = index_class.index_type
        index_path = os.path.join(self.index_dir, INDEX_FILE_NAME)
        log.debug(f"Loading index from {index_path}")
        self.__index = faiss.read_index(index_path)

    def __set_index(self):
        if self.__load_index:
            self.__load_index_from_disk()
        elif self.__build_empty_index:
            self.__build_index(self.__index_dimension)
        else:
            self.__build_index(self.data.shape[1])

    def __build_index(self, index_dimension):
        if self.index_type is IndexType.L2_INDEX:
            log.debug("Building L2 index")
            index = faiss.IndexFlatL2(index_dimension)
        elif self.index_type is IndexType.COSINE_INDEX:
            log.debug("Building cosine index")
            index = faiss.IndexFlatIP(index_dimension)
        else:
            raise ValueError(f"Unknown index type {self.index_type}")
        self.__index = faiss.IndexIDMap(index)

    def __move_to_gpu(self):
        if self.gpu_enabled:
            log.debug(f"Moving index to gpu with id: {self.gpu_id}")

            # two lines below are workaround for a faiss bug, where memmory is used on the gpu0 even when gpu_id!=0
            # for more info see https://github.com/facebookresearch/faiss/issues/1651
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

            resource = faiss.StandardGpuResources()
            self.__index = faiss.index_cpu_to_gpu(resource, self.gpu_id, self.__index)

    def __add_data_to_index(self):
        if not self.__load_index and not self.__build_empty_index:
            log.debug("Adding data to index...")
            if self.data.dtype != "float32":
                self.data = self.data.astype("float32")
            if self.index_type is IndexType.COSINE_INDEX:
                faiss.normalize_L2(self.data)
            self.__index.add_with_ids(self.data, self.ids)

    def build(self) -> Index:
        self.__validate_params()
        self.__set_index()
        self.__move_to_gpu()
        self.__add_data_to_index()
        log.debug(f"Index created, total number of vectors: {self.__index.ntotal}")
        return Index(self.__index, self.index_type, self.gpu_enabled, self.gpu_id)
