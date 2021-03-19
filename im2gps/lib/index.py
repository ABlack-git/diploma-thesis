import faiss
import logging
import im2gps.utils as utils
import numpy as np
from enum import Enum
from im2gps.data.descriptors import DescriptorsTable

log = logging.getLogger(__name__)


class IndexType(Enum):
    L2_INDEX = ('l2',)
    COSINE_INDEX = ('cosine',)

    def __init__(self, type_str):
        self.type_str = type_str


class Index:
    def __init__(self, index, index_type):
        self.index: faiss.Index = index
        self.index_type: IndexType = index_type

    def search(self, queries: np.ndarray, k: int, preprocessed=False):
        if self.index_type == IndexType.COSINE_INDEX and not preprocessed:
            faiss.normalize_L2(queries)
            dist, ind = self.index.search(queries, k)
        else:
            dist, ind = self.index.search(queries, k)
        return dist, ind


class IndexBuilder:
    def __init__(self, data, index_type=None, gpu_enabled=False, gpu_id=-1, batch_size=50000):
        self.index_type: IndexType = index_type
        self.gpu_enabled: bool = gpu_enabled
        self.gpu_id: int = gpu_id
        self.batch_size: int = batch_size
        self.data: DescriptorsTable = data

        self.__index: faiss.Index = None

    def __validate_params(self):
        assert self.index_type in IndexType, f"Unknown index type: {self.index_type}"
        assert self.batch_size > 0
        assert isinstance(self.data, DescriptorsTable), "data should be instance of DescriptorsTable"
        if self.gpu_enabled:
            assert self.gpu_id >= 0, f"gpu_id should be provided"

    def __set_index(self):
        if self.index_type is IndexType.L2_INDEX:
            log.debug("Building L2 index")
            self.__index = faiss.IndexFlatL2(self.data.desc_shape)
        elif self.index_type is IndexType.COSINE_INDEX:
            log.debug("Building cosine index")
            self.__index = faiss.IndexFlatIP(self.data.desc_shape)
        else:
            raise ValueError(f"Unknown index type {self.index_type}")

    def __move_to_gpu(self):
        if self.gpu_enabled:
            log.debug(f"Moving index to gpu with id: {self.gpu_id}")
            resource = faiss.StandardGpuResources()
            self.__index = faiss.index_cpu_to_gpu(resource, self.gpu_id, self.__index)

    def __add_data_to_index(self):
        for start, end in utils.batch_range(len(self.data), self.batch_size):
            log.debug(f"Adding vectors from {start} to {end} to index")
            batch = self.data.get_descriptors_by_range(start, end + 1, field='descriptor').astype('float32')
            if self.index_type is IndexType.COSINE_INDEX:
                faiss.normalize_L2(batch)
            self.__index.add(batch)

    def build(self) -> Index:
        self.__validate_params()
        self.__set_index()
        self.__move_to_gpu()
        self.__add_data_to_index()
        log.debug(f"Index created, total number of vectors: {self.__index.ntotal}")
        return Index(self.__index, self.index_type)
