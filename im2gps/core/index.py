import os
import faiss
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass

log = logging.getLogger(__name__)


class IndexType(Enum):
    L2_INDEX = "L2"
    COSINE_INDEX = "cosine"


@dataclass
class IndexConfig:
    index_type: IndexType = IndexType.L2_INDEX
    gpu_enabled: bool = False
    gpu_id: int = -1


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

    def __repr__(self):
        return f"Index(index_type={self.index_type}, gpu_enabled={self.gpu_enabled}, gpu_id={self.gpu_id}, " \
               f"index={repr(self.index)})"


class IndexBuilder:
    def __init__(self, data: np.ndarray, ids: np.ndarray, index_config: IndexConfig):
        self.index_type: IndexType = index_config.index_type
        self.gpu_enabled: bool = index_config.gpu_enabled
        self.gpu_id: int = index_config.gpu_id
        self.data: np.ndarray = data
        self.ids: np.ndarray = ids
        self.__index: faiss.Index = None

    def __validate_params(self):
        assert self.index_type in IndexType, f"Unknown index type: {self.index_type}"
        assert isinstance(self.data, np.ndarray), "data should be instance of np.ndarray"
        assert len(self.data) == len(self.ids), "number of data vectors and ids should be the same"
        if self.gpu_enabled:
            assert self.gpu_id >= 0, f"gpu_id should be provided"

    def __set_index(self):
        if self.index_type is IndexType.L2_INDEX:
            log.debug("Building L2 index")
            index = faiss.IndexFlatL2(self.data.shape[1])
        elif self.index_type is IndexType.COSINE_INDEX:
            log.debug("Building cosine index")
            index = faiss.IndexFlatIP(self.data.shape[1])
        else:
            raise ValueError(f"Unknown index type {self.index_type}")
        self.__index = faiss.IndexIDMap(index)

    def __move_to_gpu(self):
        if self.gpu_enabled:
            log.debug(f"Moving index to gpu with id: {self.gpu_id}")

            # two lines below are workaround for a faiss bug, where memmory is used on the gpu0 even when gpu_id!=0
            # for more info see https://github.com/facebookresearch/faiss/issues/1651
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

            resource = faiss.StandardGpuResources()
            self.__index = faiss.index_cpu_to_gpu(resource, self.gpu_id, self.__index)

    def __add_data_to_index(self):
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
