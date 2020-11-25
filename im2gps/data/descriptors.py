import tables as tb
import numpy as np
import logging

from dataclasses import dataclass, fields
from typing import List, Union, Generator

log = logging.getLogger(__name__)


@dataclass()
class Descriptor:
    photo_id: int = None
    lat: float = None
    lon: float = None
    descriptor: np.array = None


class DescriptorsTable:
    def __init__(self, file_path: str, descriptor_shape: int, flush_every=1000):
        self.file_path: str = file_path
        self.desc_shape: int = descriptor_shape
        self._unflushed = 0
        self._flush_every = flush_every
        self.file: tb.File
        self.group: tb.Group
        self.table: tb.Table
        self.__init()

    def __init(self):
        class _DescriptorTable(tb.IsDescription):
            photo_id = tb.Int64Col()
            lat = tb.Float64Col()
            lon = tb.Float64Col()
            descriptor = tb.Float64Col(shape=(self.desc_shape,))

        log.debug(f"Opening descriptor file {self.file_path}")
        self.file = tb.open_file(self.file_path, mode='a', title="Descriptor file")
        if "/descriptors" in self.file:
            self.group = self.file.root.descriptors
        else:
            self.group = self.file.create_group(self.file.root, "descriptors")
        if "/descriptors/descriptors" in self.file:
            self.table = self.file.root.descriptors.descriptors
        else:
            self.table = self.file.create_table(self.group, "descriptors", _DescriptorTable)
        log.debug("File opened successfully")

    def iterrows(self) -> Generator[Descriptor, None, None]:
        for row in self.table.iterrows():
            desc = Descriptor()
            for field in fields(Descriptor):
                setattr(desc, field.name, row[field.name])
            yield desc

    def add(self, descriptors: Union[List[Descriptor], Descriptor]):
        if isinstance(descriptors, Descriptor):
            self.__add_descriptor_to_table(descriptors)
        else:
            for desc in descriptors:
                self.__add_descriptor_to_table(desc)
        if (self._unflushed + 1) % self._flush_every == 0:
            self.table.flush()
            self._unflushed = 0

    def __add_descriptor_to_table(self, desc: Descriptor):
        row = self.table.row
        for field in fields(Descriptor):
            row[field.name] = getattr(desc, field.name)
        row.append()
        self._unflushed = self._unflushed + 1

    def close(self):
        if self._unflushed > 0:
            self.table.flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
