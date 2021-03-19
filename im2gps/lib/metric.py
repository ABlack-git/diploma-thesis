import typing as t
import numpy as np
from haversine import haversine_vector, Unit

dist_thresholds = {
    '1m': 0.001,
    '5m': 0.005,
    '10m': 0.01,
    '100m': 0.1,
    '500m': 0.5,
    '1km': 1.0,
    '5km': 5.0,
    '10km': 10.0
}
error_thresholds = {
    '0m to 1m': (0, 0.001),
    '1m to 5m': (0.001, 0.005),
    '5m to 10m': (0.005, 0.01),
    '10m to 100m': (0.01, 0.1),
    '100m to 500m': (0.1, 0.5),
    '500m to 1km': (0.5, 1.0),
    '1km to 5km': (1.0, 5.0),
    '5km to 10km': (5.0, 10.0),
    '10km+': (10.0, float('inf'))
}


def localization_accuracy(dists: np.ndarray, thresholds: t.Dict[str, float] = None):
    if thresholds is None:
        thresholds = dist_thresholds
    accuracy = {}
    for level, dist in thresholds.items():
        accuracy[level] = np.sum(dists <= dist) / dists.size
    return accuracy


def avg_errors(dists: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = error_thresholds
    error = {}
    for level, limits in thresholds.items():
        left, right = limits
        dists_in_range = dists[np.where(np.logical_and(dists >= left, dists < right))]
        error[level] = np.average(dists_in_range)
    return error


def compute_geo_distance(true_locations: np.ndarray,
                         predicted_locations: np.ndarray):
    return haversine_vector(true_locations, predicted_locations, Unit.KILOMETERS)
