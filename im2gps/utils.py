import os
import typing as t


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
