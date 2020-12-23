import typing as tp
import numpy as np
from haversine import haversine_vector, Unit


def localization_accuracy(gt_locations: tp.List[tp.Tuple[float, float]],
                          pred_locations: tp.List[tp.Tuple[float, float]],
                          thresholds: tp.Dict[str, float] = None, latitude_first: bool = False):
    if thresholds is None:
        thresholds = {
            'street_level': 1.0,
            'neighborhood_level': 5.0,
            'city_level': 20.0,
            'county_level': 50.0
        }
    if not latitude_first:
        gt_locations = [(lat, lon) for lon, lat in gt_locations]
        pred_locations = [(lat, lon) for lon, lat in pred_locations]
    dists: np.ndarray = haversine_vector(gt_locations, pred_locations, Unit.KILOMETERS)
    accuracy = {}
    for level, dist in thresholds.items():
        accuracy[level] = np.sum(dists <= dist) / dists.size
    return accuracy
