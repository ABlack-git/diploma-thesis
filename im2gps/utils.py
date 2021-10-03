import os
import typing as t
import numpy as np
import psutil
from collections import deque


class Singelton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singelton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def change_order_of_lat_and_lot(list_of_coords: t.List[t.Tuple[float, float]]) -> t.List[t.Tuple[float, float]]:
    return [(y, x) for x, y in list_of_coords]


def create_output_folders(path: str, with_filename=False):
    if with_filename:
        path = os.path.dirname(path)
    if path is None or not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)


def batch_range(size, step):
    start = 0
    end = 0
    while end < size - 1:
        end = start + step - 1
        if end > size - 1:
            end = size - 1
        yield start, end
        start = end + 1


def normalise_vector(v, axis=-1, order=2):
    norm = np.atleast_1d(np.linalg.norm(v, order, axis))
    norm[norm == 0] = 1
    return v / np.expand_dims(norm, axis)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss * 0.000000001


class Stats:
    def __init__(self, maxlen=10):
        self._current = 0
        self._average = 0
        self._moving_average = 0

        self._total_count = 0
        self._sum = 0
        self._queue = deque(maxlen=maxlen)

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        self._current = value
        self._sum += value
        self._queue.append(value)
        self._total_count += 1

        self._average = self._sum / self._total_count
        self._moving_average = sum(self._queue) / len(self._queue)

    @property
    def avg(self):
        return self._average

    @property
    def sma(self):
        return self._moving_average

    @property
    def total_count(self):
        return self._total_count

    @property
    def sum(self):
        return self._sum
